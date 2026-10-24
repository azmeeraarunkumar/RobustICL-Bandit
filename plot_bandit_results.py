from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

results_dir = Path("results")
details = pd.read_csv(results_dir / "bandit_query_results.csv")
summary = pd.read_csv(results_dir / "bandit_summary.csv")

plt.figure(figsize=(8, 5))
plt.bar(summary["policy"], summary["accuracy_percent"])
plt.ylabel("Accuracy (%%)")
plt.title("RobustICL-Bandit: Policy Comparison")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(results_dir / "bandit_policy_comparison.png", dpi=200)
plt.close()

noise_results = details.groupby(["policy", "noise_percent"])["correct"].mean().mul(100).reset_index(name="accuracy_percent")
plt.figure(figsize=(9, 5))
for policy in ["Fixed-P1", "Random", "UCB1"]:
    subset = noise_results[noise_results["policy"] == policy]
    plt.plot(subset["noise_percent"], subset["accuracy_percent"], marker="o", label=policy)
plt.xlabel("Noise Level (%%)")
plt.ylabel("Observed Accuracy (%%)")
plt.title("Policy Accuracy by Noise Level")
plt.xticks(sorted(details["noise_percent"].unique()))
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig(results_dir / "bandit_noise_analysis.png", dpi=200)
plt.close()
print("Plots saved successfully.")
