"""
抓取评论并预处理训练模型
"""
import Component.comments as comments
import Component.preprocess as preprocess
import Component.sentimentComments as sentimentComments
import Component.trainLogisticModel as trainLogisticModel
import Component.trainNNModel as trainNNModel

if __name__ == '__main__':
    # 抓取老君山评论，对应同程旅游和携程的景点ID分别为 100
    comments_path = comments.get_comments(ly_poi_id=8472, ctrip_poi_id=77806, page_size=50,
                                          output_path='./output/Comments.csv')
    print(f"\033[34m [Log] Comments fetched and saved successfully at {comments_path} \033[0m")

    # 预处理评论数据
    preprocess_path = preprocess.process_data(file_path=comments_path, output_path='./output/ProcessData.csv')
    print(f"\033[34m [Log] Data processed and saved successfully at {preprocess_path} \033[0m")

    # 标注情感
    sentimentComments_path = sentimentComments.get_sentiment(file_path=preprocess_path,
                                                             output_path='./output/SentimentAnnotatedComments.csv')

    # 训练模型
    # 训练逻辑回归模型
    logistic_model_path = trainLogisticModel.train_logistic_model(file_path=sentimentComments_path,
                                                                  output_path='./output/LogisticModel.joblib')
    print(f"\033[34m [Log] Logistic Model trained and saved successfully at {logistic_model_path} \033[0m")

    # 训练神经网络模型
    nn_model_path = trainNNModel.train_neural_network_model(file_path=sentimentComments_path,
                                                            output_path='./output/NNModel.pt')
    print(f"\033[34m [Log] Neural Network Model trained and saved successfully at {nn_model_path} \033[0m")


