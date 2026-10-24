
import csv
import os
import time

import torch
from datasets import load_from_disk
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

# Partition number -> expected noise percentage
NOISE_LEVELS = {
    1: 0,
    2: 10,
    3: 20,
    4: 30,
    5: 40,
    6: 50,
    7: 40,
    8: 30,
    9: 20,
    10: 10,
}

NUM_SHOTS = 4
NUM_QUERIES = 100
DEMONSTRATION_SEED = 42
EVAL_PATH = "data/processed/eval_set"
RESULTS_PATH = "results/partition_evaluation.csv"


def load_model():
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def predict(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
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


def normalize_prediction(response):
    response = response.lower().strip()

    # Match complete category names first.
    if "world" in response:
        return 0
    if "sports" in response or "sport" in response:
        return 1
    if "business" in response:
        return 2
    if "sci/tech" in response:
        return 3
    if "science" in response or "technology" in response:
        return 3

    return None


def evaluate_partition(partition_id, tokenizer, model, eval_data):
    print("\n" + "=" * 65, flush=True)
    print(
        f"Evaluating partition {partition_id} "
        f"(expected noise: {NOISE_LEVELS[partition_id]}%)",
        flush=True,
    )
    print("=" * 65, flush=True)

    partition = load_partition(partition_id, noisy=True)
    queries = eval_data.select(range(min(NUM_QUERIES, len(eval_data))))

    correct = 0
    invalid = 0
    start_time = time.time()

    for query_number, example in enumerate(queries, start=1):
        demonstrations = sample_demonstrations(
            partition,
            num_shots=NUM_SHOTS,
            seed=DEMONSTRATION_SEED + query_number,
        )

        prompt = build_icl_prompt(
            demonstrations,
            example["text"],
        )

        response = predict(model, tokenizer, prompt)
        predicted_label = normalize_prediction(response)
        true_label = int(example["label"])

        if predicted_label is None:
            invalid += 1
        elif predicted_label == true_label:
            correct += 1

        print(
            f"Partition {partition_id} | "
            f"Query {query_number}/{len(queries)} | "
            f"Prediction: {response} | "
            f"Correct: {predicted_label == true_label}",
            flush=True,
        )

    total = len(queries)
    accuracy = correct / total if total else 0.0
    elapsed = time.time() - start_time

    result = {
        "partition": partition_id,
        "noise_percent": NOISE_LEVELS[partition_id],
        "queries": total,
        "shots": NUM_SHOTS,
        "correct": correct,
        "invalid_predictions": invalid,
        "accuracy_percent": round(accuracy * 100, 2),
        "elapsed_seconds": round(elapsed, 2),
    }

    print("\nPartition summary:", flush=True)
    for key, value in result.items():
        print(f"{key}: {value}", flush=True)

    return result


def main():
    print("=" * 65)
    print("ROBUST-ICL-BANDIT: ALL-PARTITION EVALUATION")
    print("=" * 65)

    if not os.path.isdir(EVAL_PATH):
        raise FileNotFoundError(
            f"Evaluation set not found: {EVAL_PATH}"
        )

    eval_data = load_from_disk(EVAL_PATH)
    if len(eval_data) == 0:
        raise ValueError("The evaluation set is empty.")

    num_queries = min(NUM_QUERIES, len(eval_data))
    print(f"Evaluation queries per partition: {num_queries}")
    print(f"Shots per query: {NUM_SHOTS}")
    print("Partitions to evaluate: 1–10")

    tokenizer, model = load_model()
    results = []

    try:
        for partition_id in range(1, 11):
            result = evaluate_partition(
                partition_id,
                tokenizer,
                model,
                eval_data,
            )
            results.append(result)

            # Save after every partition so completed results survive
            # if the run is interrupted later.
            os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
            with open(
                RESULTS_PATH,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=list(results[0].keys()),
                )
                writer.writeheader()
                writer.writerows(results)

            print(
                f"\nSaved progress to {RESULTS_PATH}",
                flush=True,
            )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)
    for result in results:
        print(
            f"P{result['partition']} | "
            f"Noise {result['noise_percent']}% | "
            f"Accuracy {result['accuracy_percent']}%"
        )

    print(f"\nFull results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
