"""
使用BERT模型和逻辑回归进行情感分析
读取SentimentAnnotatedComments.csv文件，该文件包含评论和情感标签
使用BERT模型和分词器对评论进行向量化，训练逻辑回归模型
输出结果为准确率、F1分数、精确度、均方误差和召回率
将模型保存为joblib文件
"""

import os
import Component.bertEncode as bertEncode
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, precision_score, recall_score

import torch
from joblib import dump


def train_logistic_model(file_path, output_path='./output/lr_clf.joblib', max_iter=2000):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(file_path)

    # 将连续的情感值转换为二分类标签
    sentiments = (df['IntegratedSentiment'] >= 0.5).astype(int)
    comments = df['comment'].values

    encoded_comments = bertEncode.encode_text(comments, device)
    # 训练测试集划分
    X_train, X_test, y_train, y_test = train_test_split(
        encoded_comments, sentiments, test_size=0.3, random_state=42
    )

    # 训练逻辑回归模型
    lr_clf = LogisticRegression(max_iter=max_iter)  # 增加迭代次数以确保收敛
    lr_clf.fit(X_train, y_train)

    # 保存模型
    dump(lr_clf, output_path)

    # 模型评估
    y_pred = lr_clf.predict(X_test)

    # 计算并打印准确率、F1分数、精确度、均方误差和召回率
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f'Accuracy: {accuracy:.4f}, '
          f'F1 Score: {f1:.4f}, '
          f'Precision: {precision:.4f}, '
          f'Recall: {recall:.4f}'
          f'Mean Squared Error: {mse:.4f}')
    return output_path
