import json

import pandas as pd
import torch as t
from eval_metrics import calc_metrics
from parse import parse_method
from tqdm import tqdm
from datasets import Dataset, DatasetDict


def from_df(df):
    dataset = []

    for code, label in tqdm(zip(df["code"].tolist(), df["label"].tolist()), total=len(df)):
        method_text = parse_method(code)
        prompt = (
            "<|fim_prefix|>"
            + method_text.replace("METHOD_NAME", "<|fim_suffix|>", 1)
            + "<|fim_middle|>"
        )
        dataset.append({
            "text": prompt,
            "label": str(label).replace(" ", "_"),
        })
    return Dataset.from_list(dataset)

train_df = pd.read_csv("/data/data/train_train.csv")
val_df = pd.read_csv("/data/data/train_val.csv")

dataset_dict = DatasetDict({
    "train": from_df(train_df),
    "validation": from_df(val_df)
})

dataset_dict.push_to_hub("fim-dataset")