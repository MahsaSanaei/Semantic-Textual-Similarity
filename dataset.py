import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, labels_to_ids):
        self.len = len(data)
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids
       
    def __len__(self):
        return self.len
             
    def __getitem__(self, index):
        text1 = ' '.join(self.data['premise'].tolist()[index].split())
        text2 = ' '.join(self.data['hypothesis'].tolist()[index].split())
        label = self.data['label'].tolist()[index]
        inputs = self.tokenizer.encode_plus(text1,
                                            text2,
                                            max_length=self.max_len,
                                            padding='max_length',
                                            return_token_type_ids=True,
                                            truncation=True)
        label = self.labels_to_ids[label]
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        return {
             'ids': torch.tensor(ids, dtype=torch.long),
             'mask': torch.tensor(mask, dtype=torch.long),
             'targets': torch.tensor(label, dtype=torch.long)
        }