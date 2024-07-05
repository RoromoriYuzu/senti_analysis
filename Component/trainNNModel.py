"""
使用BERT和构建的神经网络进行情感分析
读取SentimentAnnotatedComments.csv文件，该文件包含评论和情感标签
使用BERT对评论进行向量化，训练模型
保存模型为sentiment_classifier.pt
输出结果为准确率、F1分数、精确度、均方误差和召回率
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from Component import bertEncode


def train_neural_network_model(file_path, output_path='./output/sentiment_classifier.pt', epochs=70):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 加载和预处理数据
    df = pd.read_csv(file_path)
    comments = df['comment'].values
    sentiments = (df['IntegratedSentiment'] >= 0.5).astype(int)

    encoded_comments = bertEncode.encode_text(comments, device)

    # 将数据集拆分为训练集和验证集
    train_features, val_features, train_labels, val_labels = train_test_split(
        encoded_comments, sentiments, test_size=0.3, random_state=42
    )

    # 如果标签是Pandas Series对象，则重置索引
    if isinstance(train_labels, pd.Series):
        train_labels = train_labels.reset_index(drop=True)
    if isinstance(val_labels, pd.Series):
        val_labels = val_labels.reset_index(drop=True)

    # 构建PyTorch数据集和数据加载器
    class SentimentDataset(Dataset):
        def __init__(self, embeddings, sentiments):
            self.embeddings = embeddings
            self.sentiments = sentiments

        def __len__(self):
            return len(self.sentiments)

        def __getitem__(self, idx):
            return self.embeddings[idx], self.sentiments[idx]

    train_dataset = SentimentDataset(train_features, train_labels)
    val_dataset = SentimentDataset(val_features, val_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    # 构建神经网络模型
    class SentimentClassifier(nn.Module):
        def __init__(self):
            super(SentimentClassifier, self).__init__()
            self.fc1 = nn.Linear(768, 50)  # BERT base产生768维度的嵌入
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(50, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x

    model = SentimentClassifier()
    model = model.to(device)  # 将模型移动到CUDA

    # 训练模型
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环
    num_epochs = epochs  # epoch数量,多了会过拟合
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            total_loss += loss.item()

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.4f}')

        # 验证步骤
        model.eval()
        val_loss = 0
        val_outputs = []
        val_labels = []
        with torch.no_grad():
            for batch in val_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                val_loss += loss.item()

                val_outputs.extend(outputs.squeeze().tolist())
                val_labels.extend(labels.tolist())

        val_loss /= len(val_dataloader)

        # 计算并打印准确率、F1分数、精确度和召回率
        val_outputs = [1 if output >= 0.5 else 0 for output in val_outputs]
        accuracy = accuracy_score(val_labels, val_outputs)
        f1 = f1_score(val_labels, val_outputs)
        precision = precision_score(val_labels, val_outputs)
        recall = recall_score(val_labels, val_outputs)

        # 打印所有的指标
        print(f'Validation Loss: {val_loss:.4f}, '
              f'Accuracy: {accuracy:.4f}, '
              f'F1 Score: {f1:.4f}, '
              f'Precision: {precision:.4f}, '
              f'Recall: {recall:.4f}')

    # 保存模型
    torch.save(model.state_dict(), output_path)
    return output_path
