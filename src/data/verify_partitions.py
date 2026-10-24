from datasets import load_from_disk
from pathlib import Path
from collections import Counter


NUM_PARTITIONS = 10


def main():
    partition_root = Path("data/partitions")

    print("=" * 60)
    print("VERIFYING IID PARTITIONS")
    print("=" * 60)

    for i in range(1, NUM_PARTITIONS + 1):

        partition_path = partition_root / f"partition_{i}"

        partition = load_from_disk(str(partition_path))

        labels = partition["label"]
        counts = Counter(labels)

        total = len(partition)

        print(f"\nPartition {i}")
        print("-" * 40)
        print(f"Total samples: {total}")

        for label in range(4):
            count = counts[label]
            percentage = (count / total) * 100

            print(
                f"Class {label}: "
                f"{count:5d} samples "
                f"({percentage:6.2f}%)"
            )


if __name__ == "__main__":
    main()