from datasets import load_from_disk
from pathlib import Path
import random


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

NUM_PARTITIONS = 10
NUM_CLASSES = 4

BASE_SEED = 42

NOISE_RATES = {
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


# ============================================================
# SYMMETRIC LABEL NOISE
# ============================================================

def add_symmetric_noise(partition, noise_rate, seed):

    rng = random.Random(seed)

    original_labels = partition["label"]
    noisy_labels = []

    for label in original_labels:

        if rng.random() < noise_rate:

            alternative_labels = [
                c
                for c in range(NUM_CLASSES)
                if c != label
            ]

            corrupted_label = rng.choice(
                alternative_labels
            )

            noisy_labels.append(corrupted_label)

        else:

            noisy_labels.append(label)

    noisy_partition = partition.remove_columns("label")

    noisy_partition = noisy_partition.add_column(
        "label",
        noisy_labels
    )

    return noisy_partition


# ============================================================
# MAIN
# ============================================================

def main():

    clean_root = Path("data/partitions")
    noisy_root = Path("data/noisy_partitions")

    noisy_root.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("ROBUST-ICL-BANDIT : NOISE GENERATION")
    print("=" * 70)

    print(f"Number of partitions : {NUM_PARTITIONS}")
    print(f"Number of classes    : {NUM_CLASSES}")
    print(f"Base random seed     : {BASE_SEED}")

    print("\nNoise configuration:")
    print("-" * 40)

    for partition_id, rate in NOISE_RATES.items():

        print(
            f"Partition {partition_id:2d} "
            f"-> {rate * 100:5.1f}% noise"
        )

    for partition_id in range(
        1,
        NUM_PARTITIONS + 1
    ):

        clean_path = (
            clean_root /
            f"partition_{partition_id}"
        )

        noisy_path = (
            noisy_root /
            f"partition_{partition_id}"
        )

        noise_rate = NOISE_RATES[
            partition_id
        ]

        print("\n" + "-" * 70)

        print(
            f"Processing Partition {partition_id}"
        )

        print(
            f"Target noise rate: "
            f"{noise_rate * 100:.1f}%"
        )

        partition = load_from_disk(
            str(clean_path)
        )

        noisy_partition = add_symmetric_noise(
            partition,
            noise_rate,
            BASE_SEED + partition_id
        )

        noisy_partition.save_to_disk(
            str(noisy_path)
        )

        print(
            f"Saved -> {noisy_path}"
        )

    print("\n" + "=" * 70)
    print("NOISY PARTITION GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()