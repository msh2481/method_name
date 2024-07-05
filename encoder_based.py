from transformers import T5ForConditionalGeneration, AutoTokenizer
import re
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial

from dataset_t5 import CSVDataset, collate_fn
from eval_metrics import calc_metrics

#%%
checkpoint = "Salesforce/codet5p-220m"
device = "cuda" # for GPU usage or "cpu" for CPU usage
#%%

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)
collate_fn_tok = partial(collate_fn, tokenizer=tokenizer, device=device)

#%%

val_file = "/data/data/train_val.csv"
val_dataset = CSVDataset(val_file, tokenizer)
val_dl = DataLoader(val_dataset, batch_size=50, shuffle=True, collate_fn=collate_fn_tok)
item = next(iter(val_dl))

#%%

predictions = []
labels = []
idx = 0

for item in tqdm(val_dl):
    outputs = model.generate(item["input_ids"], max_length=10)
    prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    prediction = [re.sub(r"^[^a-zA-Z]*|[^a-zA-Z]*$", "", pred) for pred in prediction]
    prediction = [(pred.split(" ")[0]).lower().split("_") for pred in prediction]
    predictions.extend(prediction)

    labels_splitted = [label.split("_") for label in item["labels"]]
    labels.extend(labels_splitted)
    if idx > 10:
        break
    idx += 1

#%%

calc_metrics(predictions, labels)

#%%

#
# input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
# labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
#
# # the forward function automatically creates the correct decoder_input_ids
# loss = model(input_ids=input_ids, labels=labels).loss
# loss.item()

#%%