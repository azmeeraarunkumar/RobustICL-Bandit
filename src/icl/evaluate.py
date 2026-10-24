import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sampler import load_partition, sample_demonstrations
from prompt import build_icl_prompt


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

LABEL_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    return tokenizer, model


def predict(model, tokenizer, prompt):

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

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


def main():

    print("=" * 70)
    print("ROBUST-ICL-BANDIT")
    print("FIRST END-TO-END ICL TEST")
    print("=" * 70)

    # Load one noisy partition
    partition_id = 6

    print(f"\nUsing noisy partition: {partition_id}")

    partition = load_partition(
        partition_id,
        noisy=True
    )

    # Select 4 demonstrations
    demonstrations = sample_demonstrations(
        partition,
        num_shots=4,
        seed=42
    )

    # Select a separate query
    query_index = 100

    query_text = partition[query_index]["text"]
    true_label = partition[query_index]["label"]

    print("\nTrue label:")
    print(LABEL_NAMES[true_label])

    # Build ICL prompt
    prompt = build_icl_prompt(
        demonstrations,
        query_text
    )

    print("\n" + "=" * 70)
    print("ICL PROMPT")
    print("=" * 70)

    print(prompt)

    # Load model
    tokenizer, model = load_model()

    # Predict
    print("\nGenerating prediction...")

    prediction = predict(
        model,
        tokenizer,
        prompt
    )

    print("\n" + "=" * 70)
    print("MODEL PREDICTION")
    print("=" * 70)

    print(prediction)

    print("\n" + "=" * 70)
    print("TRUE LABEL")
    print("=" * 70)

    print(LABEL_NAMES[true_label])

    print("\n" + "=" * 70)
    print("END-TO-END ICL TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()