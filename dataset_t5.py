import pandas as pd
import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, csv_file, tokenizer, code_column="body"):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        # self.max_length = max_length
        self.code_column = code_column
        self.device = "cuda"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = self.data.iloc[idx][self.code_column]
        code = code.replace("METHOD_NAME", "<extra_id_0>")
        label = self.data.iloc[idx]['label']
        label = label.replace(" ", "_")
        label = f"<extra_id_0> {label}"
        label_ids = self.tokenizer(label, return_tensors="pt").input_ids.to(self.device)

        input_ids = self.tokenizer(code, return_tensors="pt").input_ids.to(self.device)
        # inputs = self.tokenizer(code, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        # input_ids = inputs['input_ids'].squeeze(0)

        return {
            'input_ids': input_ids,
            'label_ids': label_ids
        }