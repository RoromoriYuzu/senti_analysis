import os

from tqdm import tqdm

# 设置transformers的缓存目录
os.environ["TRANSFORMERS_CACHE"] = "./data/cache/"


import numpy as np

from transformers import BertTokenizer, BertModel
import torch


def batch_encode(texts, tokenizer, model, device, batch_size=64):
    model.eval()  # 将模型设置为评估模式
    input_ids = []
    attention_masks = []
    features = []

    for i in tqdm(range(0, len(texts), batch_size), desc='批处理编码'):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer.batch_encode_plus(
            batch_texts,
            max_length=512,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_masks = encoded['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_masks)

        # 只取[CLS]的输出
        batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        features.append(batch_features)

    return np.concatenate(features, axis=0)


def encode_text(texts, device, batch_size=64):
    # 加载预训练的BERT模型和分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    model.to(device)
    encoded_comments = batch_encode(texts, tokenizer, model, device=device, batch_size=batch_size)
    return encoded_comments
