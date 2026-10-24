
import math
import random


class RandomSelector:
    """Select a partition uniformly at random."""

    def __init__(self, num_partitions=10, seed=42):
        self.num_partitions = num_partitions
        self.rng = random.Random(seed)

    def select(self):
        return self.rng.randrange(self.num_partitions)

    def update(self, partition_index, reward):
        # Random selection does not learn from rewards.
        pass


class UCBSelector:
    """UCB1 selector using observed binary rewards."""

    def __init__(self, num_partitions=10):
        self.num_partitions = num_partitions
        self.counts = [0] * num_partitions
        self.rewards = [0.0] * num_partitions
        self.total_selections = 0
        self.pending = set()

    def select(self):
        # Select every partition once before using UCB scores.
        for i in range(self.num_partitions):
            if self.counts[i] == 0 and i not in self.pending:
                self.pending.add(i)
                return i

        available = [
            i for i in range(self.num_partitions)
            if i not in self.pending
        ]

        if not available:
            raise RuntimeError(
                "Call update() before selecting another partition."
            )

        def ucb_score(i):
            mean_reward = self.rewards[i] / self.counts[i]
            exploration = math.sqrt(
                2 * math.log(max(1, self.total_selections))
                / self.counts[i]
            )
            return mean_reward + exploration

        chosen = max(available, key=ucb_score)
        self.pending.add(chosen)
        return chosen

    def update(self, partition_index, reward):
        if partition_index not in self.pending:
            raise ValueError(
                "Update requires a partition selected by select()."
            )
        if reward not in (0, 1, 0.0, 1.0):
            raise ValueError("Reward must be 0 or 1.")

        self.pending.remove(partition_index)
        self.counts[partition_index] += 1
        self.rewards[partition_index] += float(reward)
        self.total_selections += 1
