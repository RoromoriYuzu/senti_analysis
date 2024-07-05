import jieba
import numpy as np
import pandas as pd


# 分词和去除停用词的函数
def tokenize_and_remove_stopwords(text, stopwords):
    # 分词
    words = jieba.lcut(text)
    # 去除停用词
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)


def process_data(file_path, output_path='./output/processData.csv', stopwords_path='./data/stop_words.utf8',
                 user_dict_path='./data/user_dict.utf8'):
    # 读取CSV文件
    combined_df = pd.read_csv(file_path)

    # 重命名列名
    combined_df.rename(columns={
        '评论时间': 'time',
        'IP': 'IP',
        '用户ID': 'userId',
        '昵称': 'userNick',
        '评论': 'comment',
        '评分': 'score',

    }, inplace=True)

    # 去除重复值
    combined_df.drop_duplicates(inplace=True)

    # 提取中文字符，去除非中文字符
    combined_df['processComment'] = combined_df['comment'].str.replace(r'[^\u4e00-\u9fa5]', '', regex=True)

    # 压缩连续字符
    combined_df['processComment'] = combined_df['processComment'].str.replace(r'(\w)\1{2,}', r'\1\1', regex=True)

    # 去除前后空白
    combined_df['processComment'] = combined_df['processComment'].str.strip()

    # 去除缺失值
    combined_df.replace('', np.nan, inplace=True)
    combined_df.dropna(inplace=True)

    # 去除长度小于4的评论
    combined_df = combined_df[combined_df['processComment'].str.len() >= 4]

    # 设置时间戳列为整数类型
    combined_df['time'] = combined_df['time'].astype(int)

    # 重置索引
    combined_df.reset_index(drop=True, inplace=True)

    # 打印处理后的DataFrame信息
    print(combined_df.info())
    print(combined_df.head())

    # 加载停用词表
    stopwords = [line.strip() for line in open(stopwords_path, encoding='utf-8').readlines()]
    # 加载自定义词典
    jieba.load_userdict(user_dict_path)

    # 分词并去除停用词
    combined_df['jieba'] = combined_df['processComment'].apply(lambda x: tokenize_and_remove_stopwords(x, stopwords))

    # 再次去除缺失值，因为分词后可能会产生空字符串
    combined_df.replace('', np.nan, inplace=True)
    combined_df.dropna(inplace=True)

    combined_df.to_csv(output_path, index=False, encoding='utf-8')
    return output_path
