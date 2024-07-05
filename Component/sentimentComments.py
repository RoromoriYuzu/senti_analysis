import pandas as pd
from snownlp import SnowNLP
from tqdm import tqdm


def nlp_intensity(text):
    # 通过SnowNLP计算情感强度
    s = SnowNLP(text)
    return s.sentiments


def senticnet_intensity(text, senticNet_data):
    # 通过SenticNet计算情感强度
    seg_list = text.split()
    intensity = 0
    count = 0
    for word in seg_list:
        if word in senticNet_data['CONCEPT'].values:
            count += 1
            intensity += senticNet_data[senticNet_data['CONCEPT'] == word]['POLARITY INTENSITY'].values[0]
    return intensity / count if count != 0 else 0


#  用户评分权重高 1.0，SnowNLP权重为0.6，SenticNet权重为0.4
def integrate_sentiment_scores(row, weight_rating=1.0, weight_snownlp=0.6, weight_senticnet=0.4):
    # 将SnowNLP分数在0-1范围
    snow_score = row['SnowNLP']

    # 将SenticNet分数从-1-1范围转换为0-1范围
    sentic_score = (row['SenticNet'] + 1) / 2

    # 将评分从1-10范围转换为0-1范围
    rating_score = row['score'] / 10

    # 计算加权平均分数
    total_weight = weight_snownlp + weight_senticnet + weight_rating
    weighted_average_score = ((snow_score * weight_snownlp) +
                              (sentic_score * weight_senticnet) +
                              (rating_score * weight_rating)) / total_weight

    return weighted_average_score


def get_sentiment(file_path, output_path='./output/Sentiment_annotated_comments.csv',
                  senticNet_data='./data/senticNet.xlsx'):
    tqdm.pandas()
    df = pd.read_csv(file_path)
    senticNet_data = pd.read_excel(senticNet_data)
    print('使用SnowNLP计算情感强度...')
    df['SnowNLP'] = df['comment'].progress_apply(nlp_intensity)
    print('\n使用SenticNet计算情感强度...')
    df['SenticNet'] = df['jieba'].progress_apply(lambda comment: senticnet_intensity(comment, senticNet_data))
    df['IntegratedSentiment'] = df.apply(integrate_sentiment_scores, axis=1)

    # 调整列的顺序
    df = df[['time', 'IP', 'userId',
             'userNick', 'comment', 'processComment',
             'jieba', 'score', 'SnowNLP',
             'SenticNet', 'IntegratedSentiment']]

    df.to_csv(output_path, index=False)
    return output_path
