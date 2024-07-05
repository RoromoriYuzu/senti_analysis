"""
情感分析
读取 Comments.csv 文件，对评论进行情感分析，将结果保存到 Comments_with_sentiment.csv 文件中、
使用之前训练好的逻辑回归模型和神经网络模型进行情感分析
对于每条评论，计算逻辑回归模型和神经网络模型的预测概率，然后将两者的概率进行加权平均，得到最终的概率
如果两者的概率差值在0.2之内，则认为是中性评论，否则根据概率大小判断是好评还是差评
差评：-1，中性：0，好评：1
"""

import os

from tqdm import tqdm

os.environ["TRANSFORMERS_CACHE"] = "./data/cache/"
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
from joblib import load


class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(768, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def get_bert_features(texts, tokenizer, bert_model, device, max_length=512):
    encoded_texts = tokenizer.batch_encode_plus(
        texts,
        max_length=max_length,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoded_texts['input_ids'].to(device)
    attention_masks = encoded_texts['attention_mask'].to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_masks)

    return outputs.last_hidden_state[:, 0, :]


def predict(features, lr_clf, sentiment_classifier):
    features_np = features.cpu().numpy()
    lr_probabilities = lr_clf.predict_proba(features_np)
    with torch.no_grad():
        nn_predictions = sentiment_classifier(features).squeeze()
    nn_probabilities = torch.stack([1 - nn_predictions, nn_predictions], dim=1).cpu().numpy()
    return lr_probabilities, nn_probabilities


def combine_probabilities(lr_prob, nn_prob, lr_weight=0.4, nn_weight=0.6):
    lr_weight, nn_weight = lr_weight / (lr_weight + nn_weight), nn_weight / (lr_weight + nn_weight)
    combined_prob = lr_prob * lr_weight + nn_prob * nn_weight
    return combined_prob


# 确定情感标签
def get_sentiment_label(combined_prob):
    x, y = combined_prob
    if abs(x - y) <= 0.2:  # 如果差值在0.2之内，则为中性评论
        return 0
    elif x > y:  # 如果x（差评的概率）更大，则为差评
        return -1
    else:  # 如果y（好评的概率）更大，则为好评
        return 1


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载BERT模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    bert_model = BertModel.from_pretrained('bert-base-chinese').to(device)
    bert_model.eval()

    # 加载逻辑回归模型
    lr_clf = load('./output/LogisticModel.joblib')

    # 加载神经网络模型
    nn_model = SentimentClassifier().to(device)
    nn_model.load_state_dict(torch.load('./output/NNModel.pt', map_location=device))
    nn_model.eval()

    # 从CSV文件加载评论数据
    comments_df = pd.read_csv('./output/Comments.csv')

    # 批量大小
    batch_size = 64

    # 分批处理数据
    combined_probabilities_list = []
    sentiment_labels = []
    lr_probabilities_list = []
    nn_probabilities_list = []

    for i in tqdm(range(0, len(comments_df), batch_size), desc='情感分析'):
        batch_texts = comments_df['评论'][i:i + batch_size].tolist()

        # 获取BERT特征
        bert_features = get_bert_features(batch_texts, tokenizer, bert_model, device, max_length=512)

        # 预测
        lr_probabilities, nn_probabilities = predict(bert_features, lr_clf, nn_model)

        for j in range(bert_features.size(0)):
            combined_probabilities = combine_probabilities(lr_probabilities[j], nn_probabilities[j], lr_weight=0.6,
                                                           nn_weight=0.4)
            combined_probabilities_list.append(combined_probabilities)
            sentiment_labels.append(get_sentiment_label(combined_probabilities))
            lr_probabilities_list.append(lr_probabilities[j])
            nn_probabilities_list.append(nn_probabilities[j])

    # 添加新的列到数据框
    comments_df['lr_probabilities'] = lr_probabilities_list
    comments_df['nn_probabilities'] = nn_probabilities_list
    comments_df['combined_probabilities'] = combined_probabilities_list
    comments_df['sentiment'] = sentiment_labels

    # 设置时间戳列为整数
    comments_df['评论时间'] = comments_df['评论时间'].astype(int)
    # 保存新的CSV文件
    comments_df.to_csv('./output/Comments_with_sentiment.csv', index=False)

    print("处理完成，结果已保存到Comments_with_sentiment.csv文件中")
