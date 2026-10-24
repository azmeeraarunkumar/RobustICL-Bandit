\# RobustICL-Bandit



\## 1. Project Overview



RobustICL-Bandit investigates the selection of training partitions for In-Context Learning (ICL) when the available data contains noisy labels.



The project compares three strategies:



\- \*\*Fixed-P1:\*\* Always selects partition 1, which has 0% injected label noise.

\- \*\*Random:\*\* Randomly selects a partition.

\- \*\*UCB1:\*\* Uses the Upper Confidence Bound algorithm to balance exploration and exploitation using observed rewards.



\## 2. Dataset



The project uses the AG News text classification dataset with four classes:



1\. World

2\. Sports

3\. Business

4\. Sci/Tech



Dataset information:

\- Training examples: 120,000

\- Test examples: 7,600

\- Evaluation queries: 100

\- Number of noisy partitions: 10

\- Samples per partition: 12,000



\## 3. Experimental Setup



\- \*\*Language model:\*\* Qwen/Qwen2.5-0.5B-Instruct

\- \*\*Task:\*\* News text classification

\- \*\*Demonstrations per query:\*\* 4

\- \*\*Reward:\*\* 1 for a correct prediction and 0 for an incorrect prediction



The intended noise rates are:



| Partition | Noise Rate |

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



\## 4. Preliminary Results



| Policy | Correct Predictions | Accuracy |

|---|---:|---:|

| Fixed-P1 | 59/100 | 59% |

| Random | 65/100 | 65% |

| UCB1 | 54/100 | 54% |



These results represent one reported run with 100 queries per policy. They are preliminary and do not establish that one policy is generally better than another.



\## 5. Result Visualizations



The `results` directory contains:



\- `partition\_accuracy.png` — partition-level accuracy visualization.

\- `bandit\_policy\_comparison.png` — comparison of policy performance.

\- `bandit\_noise\_analysis.png` — analysis of performance across noise levels.



CSV files in the same directory store detailed query results and policy summaries.



\## 6. Validation



The noisy-partition verification script was executed successfully. The observed noise rates were close to the intended rates, with small deviations expected from random label corruption.



The bandit selection module passed basic tests. The reconstructed experiment runner passed Python compilation and import checks. A complete end-to-end run of the reconstructed runner has not yet been verified.



\## 7. Limitations



\- Only 100 evaluation queries were used per policy.

\- The reported results represent one run per policy.

\- Random seeds and demonstration selection may affect results.

\- The current results do not establish statistical significance.

\- Additional experiments are needed before drawing strong conclusions about policy performance.



\## 8. Future Work



\- Repeat experiments using multiple random seeds.

\- Increase the number of evaluation queries.

\- Compare additional bandit algorithms.

\- Study the effect of different noise levels on prediction accuracy.

\- Report variability and statistical significance across repeated runs.



\## 9. Reproducibility



Run commands from the project root with the virtual environment activated.



Verify noisy partitions:



```cmd

python src\\data\\verify\_noisy\_partitions.py

```



Regenerate the bandit plots using existing results:



```cmd

python plot\_bandit\_results.py

```



\*\*Important:\*\* The full experiment runner may take considerable time and can overwrite existing result CSV files. Back up the current results before starting a new run.

