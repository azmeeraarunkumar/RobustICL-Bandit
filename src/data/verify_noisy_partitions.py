from datasets import load_from_disk
from pathlib import Path

NUM_PARTITIONS = 10
EXPECTED_NOISE = {
    1: 0.00,
    2: 0.10,
    3: 0.20,
    4: 0.30,
    5: 0.40,
    6: 0.50,
    7: 0.40,
    8: 0.30,
    9: 0.20,
    10: 0.10,
}


def main():
    noisy_root = Path("data/noisy_partitions")

    print("=" * 70)
    print("ROBUST-ICL-BANDIT")
    print("VERIFYING NOISY PARTITIONS")
    print("=" * 70)

    for partition_id in range(1, NUM_PARTITIONS + 1):

        partition_path = noisy_root / f"partition_{partition_id}"

        partition = load_from_disk(str(partition_path))

        labels = partition["label"]

        # Load the corresponding clean partition
        clean_path = Path("data/partitions") / f"partition_{partition_id}"
        clean_partition = load_from_disk(str(clean_path))
        clean_labels = clean_partition["label"]

        changed = sum(
            noisy != clean
            for noisy, clean in zip(labels, clean_labels)
        )

        actual_noise = changed / len(labels)
        expected_noise = EXPECTED_NOISE[partition_id]

        print(
            f"\nPartition {partition_id}: "
            f"Expected = {expected_noise * 100:.1f}% | "
            f"Actual = {actual_noise * 100:.2f}% | "
            f"Samples = {len(labels)}"
        )

    print("\n" + "=" * 70)
    print("NOISY PARTITION VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()