from datasets import load_dataset
from collections import Counter


def main():
    print("Loading AG-News dataset...")

    dataset = load_dataset("fancyzhx/ag_news")

    train_data = dataset["train"]
    test_data = dataset["test"]

    print("\n==============================")
    print("DATASET INFORMATION")
    print("==============================")

    print(f"Training samples: {len(train_data)}")
    print(f"Test samples:     {len(test_data)}")

    print("\nFeatures:")
    print(train_data.features)

    # -----------------------------
    # Class distribution
    # -----------------------------

    train_labels = train_data["label"]
    test_labels = test_data["label"]

    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)

    print("\n==============================")
    print("TRAIN CLASS DISTRIBUTION")
    print("==============================")

    for label, count in sorted(train_counts.items()):
        percentage = (count / len(train_labels)) * 100
        print(
            f"Class {label}: "
            f"{count} samples "
            f"({percentage:.2f}%)"
        )

    print("\n==============================")
    print("TEST CLASS DISTRIBUTION")
    print("==============================")

    for label, count in sorted(test_counts.items()):
        percentage = (count / len(test_labels)) * 100
        print(
            f"Class {label}: "
            f"{count} samples "
            f"({percentage:.2f}%)"
        )

    # -----------------------------
    # Example from each class
    # -----------------------------

    print("\n==============================")
    print("EXAMPLES")
    print("==============================")

    for label in sorted(train_counts.keys()):
        example = next(
            item for item in train_data
            if item["label"] == label
        )

        print(f"\nClass {label}:")
        print(example["text"][:500])


if __name__ == "__main__":
    main()