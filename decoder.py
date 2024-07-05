import pandas as pd
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer


def solve(method_text: str) -> list[str]:
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
        print(result)
        return result.split("_")
    except Exception as e:
        print(f"Error occurred: {e}")
        return []


model_name = "Qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=t.bfloat16,
).cuda()


def solve_hf(method_text: str) -> list[str]:
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


df = pd.read_csv("data/small_train.csv")

for i in range(10):
    x = df["code"][i]
    y = df["label"][i]
    p = solve_hf(x)
    print(p, "vs", y)
