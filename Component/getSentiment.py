import os

# 环境变量设置可以移到系统环境中，或者在程序开始时设置一次
os.environ["TRANSFORMERS_CACHE"] = "./data/cache/"
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
from joblib import load


# 加载模型和tokenizer
def load_model_and_tokenizer(device):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    bert_model = BertModel.from_pretrained('bert-base-chinese').to(device)
    bert_model.eval()
    return tokenizer, bert_model


# 情感分类器定义可以移到单独的文件中
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


# 编码和获取BERT特征的函数
def encode_and_get_bert_features(texts, tokenizer, bert_model, device):
    encoded_texts = tokenizer.batch_encode_plus(
        texts,
        max_length=512,
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


# 预测函数
def predict(text, tokenizer, bert_model, lr_clf, nn_model, device):
    bert_features = encode_and_get_bert_features([text], tokenizer, bert_model, device)
    features_np = bert_features.cpu().numpy()
    lr_probabilities = lr_clf.predict_proba(features_np)
    with torch.no_grad():
        nn_predictions = nn_model(bert_features).squeeze()
    nn_probabilities = torch.tensor([1 - nn_predictions, nn_predictions]).unsqueeze(0).cpu().numpy()
    return lr_probabilities, nn_probabilities


# 组合概率函数
def combine_probabilities(lr_prob, nn_prob, lr_weight=0.6, nn_weight=0.4):
    return lr_prob * lr_weight + nn_prob * nn_weight


# 确定情感标签
def get_sentiment_label(combined_prob):
    label = 0 if abs(combined_prob[0][0] - combined_prob[0][1]) <= 0.2 else (
        -1 if combined_prob[0][0] > combined_prob[0][1] else 1)
    return label


def print_results(text, lr_probabilities, nn_probabilities, combined_probabilities, sentiment_labels):
    print(f"评论: {text}")
    print(f"逻辑回归模型预测概率: {lr_probabilities}")
    print(f"神经网络模型预测概率: {nn_probabilities}")
    print(f"综合概率: {combined_probabilities}")
    print(f"情感标签: {sentiment_labels}")


# 获取情感
def get_sentiment(text):
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    tokenizer, bert_model = load_model_and_tokenizer(device)
    # 加载逻辑回归和神经网络模型
    lr_clf = load('./output/LogisticModel.joblib')
    nn_model = SentimentClassifier().to(device)
    nn_model.load_state_dict(torch.load('./output/NNModel.pt', map_location=device))
    nn_model.eval()

    lr_probabilities, nn_probabilities = predict(text, tokenizer, bert_model, lr_clf, nn_model, device)
    combined_prob = combine_probabilities(lr_probabilities, nn_probabilities)
    sentiment_label = get_sentiment_label(combined_prob)
    results = {
        'lr_probabilities': lr_probabilities.flatten().tolist(),
        'nn_probabilities': nn_probabilities.flatten().tolist(),
        'combined_prob': combined_prob.flatten().tolist(),
        'sentiment_label': sentiment_label
    }
    return results
