import pandas as pd
import random

file_path = './output/Comments_with_sentiment.csv'

comments = pd.read_csv(file_path)


# 获取评论数量
def get_comment_count():
    return len(comments)


# 获取指定IP的评论数量
def get_comment_count_by_ip(ip):
    return len(comments[comments['IP'] == ip])


# 获取指定IP的评论
def get_comment_by_ip(ip):
    return comments[comments['IP'] == ip].to_dict(orient='records')


#获取指定数量IP列表，以及对应的评论数量
def get_ip_list(top):
    ip_list = comments['IP'].value_counts().head(top).to_dict()
    return ip_list


# 获取IP数量
def get_ip_count():
    return len(comments['IP'].value_counts())


# 获取指定情感的评论数量
def get_comment_count_by_sentiment(sentiment):
    return len(comments[comments['sentiment'] == sentiment])


# 获取指定情感的评论
def get_comment_by_sentiment(sentiment):
    return comments[comments['sentiment'] == sentiment].to_dict(orient='records')


# 获取情感列表，以及对应的评论数量
def get_sentiment_list():
    sentiment_list = comments['sentiment'].value_counts().to_dict()
    return sentiment_list


# 获取指定数量的评论
def get_comments(index, end):
    index = max(0, index)
    end = min(len(comments), end)
    return comments.iloc[index:end].to_dict(orient='records')


# 获取随机评论
def get_random_comments():
    random_int = random.randint(0, len(comments))
    return comments.iloc[random_int].to_dict()


# 获取不同评论时间段评论数量
def get_comment_count_by_time():
    comments['评论时间'] = pd.to_datetime(comments['评论时间'], unit='s')
    comments['year_month'] = comments['评论时间'].dt.to_period('M')
    comment_counts = comments.groupby('year_month').size()
    return comment_counts
