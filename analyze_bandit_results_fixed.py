from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
DETAILS_PATH = RESULTS_DIR / "bandit_query_results.csv"
SUMMARY_PATH = RESULTS_DIR / "bandit_summary.csv"


def main() -> None:
    for path in (DETAILS_PATH, SUMMARY_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Results file not found: {path}")

    details = pd.read_csv(DETAILS_PATH)
    summary = pd.read_csv(SUMMARY_PATH)

    required_columns = {
        "policy",
        "partition",
        "noise_percent",
        "correct",
        "query_number",
        "reward",
    }
    missing_columns = required_columns.difference(details.columns)
    if missing_columns:
        raise ValueError(
            f"{DETAILS_PATH} is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    print("\n" + "=" * 60)
    print("ROBUSTICL-BANDIT: RESULT ANALYSIS")
    print("=" * 60)

    print("\n1. POLICY PERFORMANCE")
    print(summary.to_string(index=False))

    print("\n2. PARTITION SELECTION COUNTS")
    print(pd.crosstab(details["policy"], details["partition"]).to_string())

    print("\n3. ACCURACY BY NOISE LEVEL")
    noise_results = (
        details.groupby(["policy", "noise_percent"])["correct"]
        .agg(["count", "mean"])
        .reset_index()
    )
    noise_results["accuracy_percent"] = noise_results["mean"] * 100
    print(
        noise_results[
            ["policy", "noise_percent", "count", "accuracy_percent"]
        ]
        .round(2)
        .to_string(index=False)
    )

    print("\n4. FIRST 30 UCB1 SELECTIONS")
    ucb_results = details.loc[details["policy"] == "UCB1"]
    print(
        ucb_results[
            ["query_number", "partition", "noise_percent", "reward"]
        ]
        .head(30)
        .to_string(index=False)
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
