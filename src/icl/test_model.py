import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    print("=" * 60)
    print("ROBUST-ICL-BANDIT")
    print("TESTING LOCAL ICL MODEL")
    print("=" * 60)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    print("\nModel loaded successfully.")

    prompt = """You are an AG-News text classification system.

Classify the following news article into exactly one category:

World
Sports
Business
Sci/Tech

Article:
The company announced a new smartphone with an advanced processor and improved battery life.

Answer with only the category name.
"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    print("\nGenerating prediction...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    print("\n" + "=" * 60)
    print("MODEL RESPONSE:")
    print(response.strip())
    print("=" * 60)


if __name__ == "__main__":
    main()