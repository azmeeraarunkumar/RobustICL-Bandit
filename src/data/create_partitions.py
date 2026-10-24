from datasets import load_dataset
from pathlib import Path


NUM_PARTITIONS = 10
SEED = 42


def main():
    print("Loading AG-News...")

    dataset = load_dataset("fancyzhx/ag_news")
    train_data = dataset["train"]

    print(f"Total training samples: {len(train_data)}")

    # Shuffle with a fixed seed for reproducibility
    train_data = train_data.shuffle(seed=SEED)

    # Output directory
    output_dir = Path("data/partitions")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Number of samples per partition
    partition_size = len(train_data) // NUM_PARTITIONS

    print(f"Number of partitions: {NUM_PARTITIONS}")
    print(f"Samples per partition: {partition_size}")

    # Create partitions
    for i in range(NUM_PARTITIONS):

        start = i * partition_size

        if i == NUM_PARTITIONS - 1:
            end = len(train_data)
        else:
            end = (i + 1) * partition_size

        partition = train_data.select(range(start, end))

        partition_path = output_dir / f"partition_{i + 1}"

        partition.save_to_disk(str(partition_path))

        print(
            f"Partition {i + 1}: "
            f"{len(partition)} samples"
        )

    print("\nAll partitions created successfully!")


if __name__ == "__main__":
    main()