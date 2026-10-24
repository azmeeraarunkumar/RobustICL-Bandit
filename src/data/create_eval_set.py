from datasets import load_dataset
from pathlib import Path

SEED = 123
NUM_EVAL_SAMPLES = 100


def main():
    print("=" * 70)
    print("ROBUST-ICL-BANDIT")
    print("CREATING FIXED EVALUATION SET")
    print("=" * 70)

    dataset = load_dataset("fancyzhx/ag_news")

    test_data = dataset["test"]

    # Fixed shuffle so every experiment uses exactly the same queries
    test_data = test_data.shuffle(seed=SEED)

    eval_data = test_data.select(
        range(NUM_EVAL_SAMPLES)
    )

    output_path = Path("data/processed/eval_set")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    eval_data.save_to_disk(
        str(output_path)
    )

    print(f"\nEvaluation samples: {len(eval_data)}")
    print(f"Saved to: {output_path}")

    print("\nClass distribution:")

    counts = {}

    for label in eval_data["label"]:
        counts[label] = counts.get(label, 0) + 1

    for label in sorted(counts):
        print(
            f"Label {label}: "
            f"{counts[label]} samples"
        )

    print("\n" + "=" * 70)
    print("EVALUATION SET CREATED")
    print("=" * 70)


if __name__ == "__main__":
    main()