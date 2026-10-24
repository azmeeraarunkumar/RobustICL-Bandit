import random
from datasets import load_from_disk
from pathlib import Path


def load_partition(partition_id, noisy=True):
    if noisy:
        root = Path("data/noisy_partitions")
    else:
        root = Path("data/partitions")

    partition_path = root / f"partition_{partition_id}"

    return load_from_disk(str(partition_path))


def sample_demonstrations(partition, num_shots=4, seed=42):
    rng = random.Random(seed)

    indices = rng.sample(
        range(len(partition)),
        num_shots
    )

    demonstrations = []

    for index in indices:
        demonstrations.append(
            (
                partition[index]["text"],
                partition[index]["label"]
            )
        )

    return demonstrations