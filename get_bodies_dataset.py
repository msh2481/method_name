import pandas as pd

from parse import parse_method

#%%

train_dataset_file = "/data/data/train.csv"
train_dataset_file_extended = "/data/data/train_with_body.csv"
dataset = pd.read_csv(train_dataset_file)
#%%

dataset['body'] = dataset['code'].apply(lambda x: parse_method(x, "METHOD_NAME"))

dataset.to_csv(train_dataset_file_extended)

#%%
