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
        label_ids = self.tokenizer(f"<extra_id_0> {label}").input_ids

        input_ids = self.tokenizer(code).input_ids
        # inputs = self.tokenizer(code, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        # input_ids = inputs['input_ids'].squeeze(0)

        return {
            'input_ids': input_ids,
            'label_ids': label_ids,
            'label': label
        }


def collate_fn(batch, tokenizer, device):
    input_ids = [item["input_ids"] for item in batch]
    label_ids = [item["label_ids"] for item in batch]
    labels = [item["label"] for item in batch]
    max_length_input = max(len(ids) for ids in input_ids)
    max_length_label = max(len(ids) for ids in label_ids)

    input_ids_padded = []
    label_ids_padded = []

    for (input_ids, label_ids) in zip(input_ids, label_ids):
        padding_input_length = max_length_input - len(input_ids)
        padding_label_length = max_length_label - len(label_ids)

        input_ids_padded.append(input_ids + [tokenizer.pad_token_id] * padding_input_length)
        label_ids_padded.append(label_ids + [tokenizer.pad_token_id] * padding_label_length)

    return {
        "input_ids": torch.tensor(input_ids_padded).to(device),
        "attention_mask": torch.tensor(label_ids_padded).to(device),
        "labels": labels
    }