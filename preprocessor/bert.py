import torch
import numpy as np
from transformers import BertTokenizer, BertModel

class BertVectorizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        self.model = BertModel.from_pretrained('bert-base-multilingual-uncased')

    def transform(self, documents):
        input_ids = torch.LongTensor([self.tokenizer.encode(doc) for doc in documents])
        _, pooled_output = self.model(input_ids)
        return np.array(pooled_output)
