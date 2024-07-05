from transformers import T5ForConditionalGeneration, AutoTokenizer
import re
#%%
checkpoint = "Salesforce/codet5p-220m"
device = "cuda" # for GPU usage or "cpu" for CPU usage
#%%

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

#%%

function_body = '''
def <extra_id_0>():
    print("hello_world")
'''

#%%
# inputs = tokenizer.encode("def print_hello_world():<extra_id_0>", return_tensors="pt").to(device)
inputs = tokenizer.encode(function_body, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_length=10)
prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(prediction)
# ==> print "Hello World"
# match = re.search(r'def (\w+)', prediction)
# function_name = match.group(1)
# print(function_name)
#%%

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids

# the forward function automatically creates the correct decoder_input_ids
loss = model(input_ids=input_ids, labels=labels).loss
loss.item()

#%%