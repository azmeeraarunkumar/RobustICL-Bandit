
"""
RobustICL-Bandit: Partition Selection Algorithms

Implements:
1. Random selection baseline
2. Upper Confidence Bound (UCB1) selection

This module tests the selection logic independently.
It does not yet run the language model or evaluate real predictions.
"""

import math
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class BanditStats:
    """Track the observed rewards for one partition."""

    pulls: int = 0
    reward_sum: float = 0.0

    @property
    def average_reward(self) -> float:
        if self.pulls == 0:
            return 0.0
        return self.reward_sum / self.pulls


class RandomSelector:
    """Select a partition uniformly at random."""

    def __init__(
        self,
        num_partitions: int,
        seed: Optional[int] = None,
    ):
        if num_partitions < 1:
            raise ValueError("num_partitions must be at least 1.")

        self.num_partitions = num_partitions
        self.rng = random.Random(seed)

    def select(self) -> int:
        """Return a zero-based partition index."""
        return self.rng.randrange(self.num_partitions)


class UCBSelector:
    """Upper Confidence Bound (UCB1) partition selector."""

    def __init__(self, num_partitions: int):
        if num_partitions < 1:
            raise ValueError("num_partitions must be at least 1.")

        self.num_partitions = num_partitions
        self.stats = [
            BanditStats() for _ in range(num_partitions)
        ]
        self.total_pulls = 0

    def select(self) -> int:
        """
        Return a zero-based partition index.

        Untried partitions are selected first.
        After every partition has been tried, select the
        partition with the highest UCB1 score.
        """
        for index, stats in enumerate(self.stats):
            if stats.pulls == 0:
                return index

        log_total = math.log(self.total_pulls)

        scores = [
            stats.average_reward
            + math.sqrt(2.0 * log_total / stats.pulls)
            for stats in self.stats
        ]

        best_score = max(scores)
        best_indices = [
            index
            for index, score in enumerate(scores)
            if math.isclose(score, best_score)
        ]

        return random.choice(best_indices)

    def update(self, partition_index: int, reward: float) -> None:
        """
        Update the chosen partition using an observed reward.

        For classification, use 1.0 for a correct prediction
        and 0.0 for an incorrect prediction.
        """
        if not 0 <= partition_index < self.num_partitions:
            raise IndexError("Invalid partition index.")

        if not 0.0 <= reward <= 1.0:
            raise ValueError("Reward must be between 0.0 and 1.0.")

        self.stats[partition_index].pulls += 1
        self.stats[partition_index].reward_sum += reward
        self.total_pulls += 1

    def summary(self) -> list[dict]:
        """Return statistics for each partition."""
        return [
            {
                "partition": index + 1,
                "pulls": stats.pulls,
                "average_reward": round(
                    stats.average_reward, 4
                ),
            }
            for index, stats in enumerate(self.stats)
        ]


def run_smoke_test() -> None:
    """Verify that both selectors and UCB updates work."""
    print("=" * 60)
    print("ROBUSTICL-BANDIT: BANDIT SELECTOR TEST")
    print("=" * 60)

    num_partitions = 10

    random_selector = RandomSelector(
        num_partitions=num_partitions,
        seed=42,
    )

    random_choice = random_selector.select()
    assert 0 <= random_choice < num_partitions
    print(
        "\nRandom selector test passed."
        f"\nExample selected partition: {random_choice + 1}"
    )

    ucb_selector = UCBSelector(num_partitions)

    # Verify that every partition is selected once initially.
    initial_choices = [
        ucb_selector.select()
        for _ in range(num_partitions)
    ]

    assert initial_choices == list(range(num_partitions))

    # Supply synthetic rewards only to test the implementation.
    # These are NOT actual model evaluation results.
    for index in initial_choices:
        synthetic_reward = 1.0 if index == 2 else 0.0
        ucb_selector.update(index, synthetic_reward)

    # Verify selection and reward updates continue to work.
    next_choice = ucb_selector.select()
    ucb_selector.update(next_choice, 1.0)

    assert ucb_selector.total_pulls == num_partitions + 1

    print("\nUCB selector test passed.")
    print(f"Next selected partition: {next_choice + 1}")
    print("\nTEST STATISTICS")
    print("-" * 60)

    for item in ucb_selector.summary():
        print(
            f"Partition {item['partition']:2d} | "
            f"Pulls: {item['pulls']:2d} | "
            f"Average reward: {item['average_reward']:.4f}"
        )

    print("\nAll selector smoke tests passed.")
    print(
        "\nNote: Synthetic rewards were used for testing. "
        "Real evaluation integration is still required."
    )


if __name__ == "__main__":
    run_smoke_test()
