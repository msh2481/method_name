#%%

import pandas as pd
from sklearn.model_selection import train_test_split

#%%
train_dataset_file = "/data/data/train_with_body.csv"
df = pd.read_csv(train_dataset_file)

train_df, val_df = train_test_split(df, test_size=0.10, random_state=42)

train_df.to_csv('train_train.csv', index=False)
val_df.to_csv('train_val.csv', index=False)

#%%