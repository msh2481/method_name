import json

import pandas as pd
import torch as t
from eval_metrics import calc_metrics
from parse import parse_method
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GemmaTokenizer


def smart_split(text: str) -> list[str]:
    if text == "METHOD_NAME":
        return ["test"]
    result = []
    buffer = ""
    for c in text:
        if c == "_":
            if buffer:
                result.append(buffer)
            buffer = ""
        elif c.isupper():
            if buffer:
                result.append(buffer)
            buffer += c.lower()
        else:
            buffer += c
    if buffer:
        result.append(buffer)
    return result


def solve_gpt(method_text: str) -> list[str]:
    method_text = parse_method(method_text)
    from openai import OpenAI

    client = OpenAI()
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that predicts method names based on the given method body.",
                },
                {
                    "role": "user",
                    "content": f"Predict the method name for the following Python method:\n\n{method_text}\n\nProvide only the method name without any explanation.",
                },
            ],
        )
        result = completion.choices[0].message.content.strip()
        return smart_split(result)
    except Exception as e:
        print(f"Error occurred: {e}")
        return []


model_name = "google/codegemma-7b"
tokenizer = GemmaTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=t.bfloat16,
).cuda()


def solve_fim(method_text: str) -> list[str]:
    method_text = parse_method(method_text)
    prompt = (
        "<|fim_prefix|>"
        + method_text.replace("METHOD_NAME", "<|fim_suffix|>", 1)
        + "<|fim_middle|>"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs, max_new_tokens=100)
    result = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    suffix = "<|file_separator|>"
    if result.endswith(suffix):
        result = result[: -len(suffix)]
    return smart_split(result)


def solve_dummy(method_text: str) -> list[str]:
    return ["test", "get", "set", "to"]


def solve_decoder(method_text: str) -> list[str]:
    method_text = parse_method(method_text)
    try:
        messages = [
            {
                "role": "system",
                "content": "",
            },
            {
                "role": "user",
                "content": f"Predict the method name (masked with METHOD_NAME) for the following Python method:\n\n{method_text}\n\nProvide only the name, in camel case, without any explanation.",
            },
        ]

        prompt = (
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            + "METHOD_NAME = "
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with t.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        predicted_name = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        predicted_name = predicted_name.split("\n")[-1].split("=")[-1].strip("\"'`. ")
        print(predicted_name)
        print()
        return predicted_name.split("_")
    except Exception as e:
        print(f"Error occurred: {e}")
        return []


def make_submission(input_name: str, output_name: str):
    df = pd.read_csv(input_name)
    ground_truth = []
    predictions = []
    try:
        ground_truth = [x.split() for x in df["label"].tolist()]
        df.drop(columns=["label"], inplace=True)
    except:
        print("Running in test mode, no labels")
    codes = df["code"].tolist()
    labels = []
    with t.no_grad():
        for i in tqdm(range(len(df))):
            p = solve_dummy(codes[i])
            predictions.append(p)
            labels.append(" ".join(p))
    df["label"] = labels

    df.to_csv(output_name, index=False)
    if ground_truth:
        results = calc_metrics(predictions, ground_truth)
        print(json.dumps(results, indent=4))


make_submission("data/test_methods.csv", "submission.csv")
