# RobustICL-Bandit

## 1. Project Overview

RobustICL-Bandit investigates partition selection for in-context learning (ICL) when training data contains noisy labels.

It compares three strategies:

- **Fixed-P1:** Always selects partition 1, with 0% intended label noise.
- **Random:** Selects a partition randomly.
- **UCB1:** Uses Upper Confidence Bound (UCB1) to balance exploration and exploitation based on observed rewards.

## 2. Dataset

The project uses AG News for four-class text classification:

- World
- Sports
- Business
- Sci/Tech

Dataset details:

- Training examples: 120,000
- Test examples: 7,600
- Evaluation queries: 100
- Number of noisy partitions: 10
- Samples per partition: 12,000

## 3. Experimental Setup

- **Language model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Task:** News text classification
- **Demonstrations per query:** 4
- **Reward:** 1 for a correct prediction, 0 otherwise

Intended partition noise rates:

| Partition | Noise rate |
|---|---:|
| P1 | 0% |
| P2 | 10% |
| P3 | 20% |
| P4 | 30% |
| P5 | 40% |
| P6 | 50% |
| P7 | 40% |
| P8 | 30% |
| P9 | 20% |
| P10 | 10% |

## 4. Preliminary Results

### Earlier saved run

| Policy | Correct predictions | Accuracy |
|---|---:|---:|
| Fixed-P1 | 59/100 | 59% |
| Random | 65/100 | 65% |
| UCB1 | 54/100 | 54% |

### Repaired runner run

| Policy | Correct predictions | Accuracy |
|---|---:|---:|
| Fixed-P1 | 62/100 | 62% |
| Random | 64/100 | 64% |
| UCB1 | 66/100 | 66% |

The repaired runner completed 100 queries per policy and saved 300 detailed query records. No invalid predictions were reported in the summary. Its output was saved separately, preserving the earlier CSV files.

These are preliminary results from separate runs. They do not establish statistical significance or prove that one policy is generally superior.

## 5. Result Visualizations and Data

The `results` directory contains visualizations and CSV files, including:

- `partition_accuracy.png` — partition-level accuracy.
- `bandit_policy_comparison.png` — policy comparison.
- `bandit_noise_analysis.png` — analysis across noise levels.
- `partition_evaluation.csv` — partition-level evaluation.
- `bandit_query_results.csv` and `bandit_summary.csv` — earlier policy results.
- `bandit_query_results_repaired.csv` and `bandit_summary_repaired.csv` — repaired runner results.

## 6. Validation

The noisy-partition verification previously completed successfully, with observed noise rates close to the intended rates.

The bandit selection module passed basic tests. The repaired runner passed syntax and import checks and subsequently completed an end-to-end run, generating separate result CSV files. The repaired results were checked for 300 rows, 100 queries per policy, missing values, and valid class labels.

## 7. Limitations

- Only 100 queries were evaluated per policy in each reported run.
- Results come from separate runs and may vary with random seeds and demonstration selection.
- The observed differences have not been established as statistically significant.
- Further repeated trials are needed before drawing strong conclusions about policy performance.

## 8. Future Work

- Repeat experiments with multiple random seeds.
- Increase the number of evaluation queries.
- Report variability and confidence intervals.
- Compare additional bandit algorithms.
- Study the relationship between label noise and prediction accuracy.

## 9. Reproducibility

Run commands from the project root with the virtual environment activated.

Activate the environment on Windows:

```cmd
.venv\Scripts\activate
```

Install dependencies in a fresh environment:

```cmd
python -m pip install -r requirements.txt
```

Verify noisy partitions using the verification script available in the repository. Check its filename under `src\data` before running it.

Regenerate plots from the available saved results:

```cmd
python plot_bandit_results.py
```

The experiment runner is:

```cmd
python src\bandit\evaluate_bandit.py
```

**Important:** Model inference may take considerable time. Back up existing results before running experiments. The repaired runner creates separate CSV files named `bandit_query_results_repaired.csv` and `bandit_summary_repaired.csv`.

## 10. Project Artifacts

The repository includes source code, configuration and data-processing scripts, saved experiment results, visualizations, this README, and `CSE399_Final_Report.md`.
