
import csv
import sys
from pathlib import Path

# Make the project modules importable when run from the project root.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "icl"))
sys.path.insert(0, str(ROOT / "src" / "bandit"))

from evaluate_partition import load_model, predict, normalize_prediction
from sampler import load_partition, sample_demonstrations
from prompt import build_icl_prompt
from bandit_selection import RandomSelector, UCBSelector


RESULTS_DIR = ROOT / "results"
DETAILS_PATH = RESULTS_DIR / "bandit_query_results.csv"
SUMMARY_PATH = RESULTS_DIR / "bandit_summary.csv"
EVAL_PATH = ROOT / "data" / "processed" / "eval_set"

NUM_QUERIES = 100
NUM_SHOTS = 4
SEED = 42

LABEL_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def get_label(example):
    """Read the true label from an evaluation example."""
    label = example["label"]
    return int(label)


def run_policy(policy_name, selector, eval_data, tokenizer, model, partition_cache):
    """Evaluate one partition-selection policy."""
    rows = []
    correct_count = 0
    invalid_count = 0

    query_count = min(NUM_QUERIES, len(eval_data))

    for query_number in range(query_count):
        query = eval_data[query_number]
        true_label = get_label(query)

        if policy_name == "Fixed-P1":
            partition_index = 0
        else:
            partition_index = selector.select()

        partition_id = partition_index + 1

        if partition_id not in partition_cache:
            partition_cache[partition_id] = load_partition(
                partition_id, noisy=True
            )

        partition = partition_cache[partition_id]
        demonstrations = sample_demonstrations(
            partition,
            num_shots=NUM_SHOTS,
            seed=SEED + query_number,
        )

        prompt = build_icl_prompt(demonstrations, query["text"])
        response = predict(model, tokenizer, prompt)
        predicted_label = normalize_prediction(response)

        is_correct = (
            predicted_label is not None
            and int(predicted_label) == true_label
        )

        reward = int(is_correct)
        correct_count += reward

        if predicted_label is None:
            invalid_count += 1

        if policy_name == "UCB1":
            selector.update(partition_index, reward)

        rows.append({
            "policy": policy_name,
            "query_number": query_number + 1,
            "partition": partition_id,
            "noise_percent": {
                1: 0, 2: 10, 3: 20, 4: 30, 5: 40,
                6: 50, 7: 40, 8: 30, 9: 20, 10: 10,
            }[partition_id],
            "true_label": LABEL_NAMES.get(true_label, str(true_label)),
            "predicted_label": (
                LABEL_NAMES.get(int(predicted_label), str(predicted_label))
                if predicted_label is not None else "INVALID"
            ),
            "correct": reward,
            "reward": reward,
        })

        if policy_name == "Random":
            selector.update(partition_index, reward)

        if (query_number + 1) % 10 == 0:
            print(
                f"{policy_name}: completed {query_number + 1}/{query_count} queries"
            )

    accuracy = 100.0 * correct_count / query_count if query_count else 0.0

    summary = {
        "policy": policy_name,
        "queries": query_count,
        "correct": correct_count,
        "invalid_predictions": invalid_count,
        "accuracy_percent": accuracy,
        "mean_reward": correct_count / query_count if query_count else 0.0,
    }

    return rows, summary


def main():
    """Run the three policies and save their results."""
    from datasets import load_from_disk

    if not EVAL_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation set not found at {EVAL_PATH}. "
            "Check the configured evaluation-set path before running."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading evaluation data...")
    eval_data = load_from_disk(str(EVAL_PATH))

    print("Loading language model...")
    tokenizer, model = load_model()

    all_rows = []
    summaries = []
    partition_cache = {}

    policies = [
        ("Fixed-P1", None),
        ("Random", RandomSelector(num_partitions=10, seed=SEED)),
        ("UCB1", UCBSelector(num_partitions=10)),
    ]

    for policy_name, selector in policies:
        print(f"\nStarting policy: {policy_name}")
        rows, summary = run_policy(
            policy_name,
            selector,
            eval_data,
            tokenizer,
            model,
            partition_cache,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    detail_columns = [
        "policy", "query_number", "partition", "noise_percent",
        "true_label", "predicted_label", "correct", "reward",
    ]
    summary_columns = [
        "policy", "queries", "correct", "invalid_predictions",
        "accuracy_percent", "mean_reward",
    ]

    # Write to new filenames so existing results are not overwritten.
    details_path = RESULTS_DIR / "bandit_query_results_repaired.csv"
    summary_path = RESULTS_DIR / "bandit_summary_repaired.csv"

    with details_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=detail_columns)
        writer.writeheader()
        writer.writerows(all_rows)

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(summaries)

    print("\nNew result files saved:")
    print(details_path)
    print(summary_path)
    print("The original CSV files were not overwritten.")


if __name__ == "__main__":
    main()
