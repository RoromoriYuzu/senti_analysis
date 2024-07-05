from jieba import analyse
import pandas as pd

file_path = './output/Comments_with_sentiment.csv'

# 获取关键词
def get_keyword(topK=10):
    df = pd.read_csv(file_path)
    comments = df['评论'].tolist()
    comments = ''.join(comments)
    keywords = analyse.extract_tags(comments, topK=topK, withWeight=True)
    return keywords


# 获取指定情感的关键词
def get_keyword_by_sentiment(sentiment, topK=10):
    df = pd.read_csv(file_path)
    df = df[df['sentiment'] == sentiment]
    comments = df['评论'].tolist()
    comments = ''.join(comments)
    keywords = analyse.extract_tags(comments, topK=topK, withWeight=True)
    return keywords
