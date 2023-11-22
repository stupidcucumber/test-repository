import pandas as pd
import torch
from .tokenizer import Tokenizer
from torch.utils.data import Dataset


class BertDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: Tokenizer):
        super().__init__()

        self.tokenizer = tokenizer

        temp_data = [self.tokenizer.tokenize(dataframe.iloc[index]['sentence'], dataframe.iloc[index]['tags']) for index in range(len(dataframe))]

        self.data = pd.DataFrame(temp_data)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        return self.data.iloc[index]['input_ids'], self.data.iloc[index]['attention_mask'], torch.LongTensor(self.data.loc[index]['labels'])