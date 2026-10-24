\# RobustICL-Bandit: Robust In-Context Learning with Bandit-Based Partition Selection



\*\*Course:\*\* CSE-399 Project  

\*\*Project:\*\* RobustICL-Bandit  

\*\*Date:\*\* October 2026



\## Abstract



This project investigates partition selection for in-context learning (ICL) under noisy-label conditions. Using the AG News text classification dataset and the Qwen2.5-0.5B-Instruct language model, we evaluate classification performance across ten training partitions with different intended label-noise rates. Three partition-selection policies are compared: Fixed-P1, Random, and Upper Confidence Bound (UCB1). Each policy is evaluated on 100 queries using four demonstrations per query. In the saved experiment results, Fixed-P1 achieves 59% accuracy, Random achieves 65%, and UCB1 achieves 54%. These preliminary findings show differences in observed performance across the tested policies, but further repeated experiments are required to establish the reliability and generality of the comparison.



\## 1. Introduction



In-context learning enables a language model to perform a task using examples included directly in its prompt. The quality and labels of these examples can influence prediction performance.



In practical settings, training data may contain incorrect labels. Selecting demonstrations from noisy data may therefore affect classification accuracy.



This project studies whether bandit-based selection can help choose among multiple data partitions with different noise levels. It compares a fixed-partition baseline, random selection, and a UCB1-based policy using a common text classification task.



\## 2. Objectives



The main objectives are:



1\. Prepare multiple partitions of the AG News dataset with different levels of label noise.

2\. Evaluate ICL classification performance across the partitions.

3\. Compare Fixed-P1, Random, and UCB1 selection policies.

4\. Record prediction accuracy, invalid predictions, and execution time.

5\. Analyze the observed results and identify limitations for future work.



\## 3. Dataset and Experimental Setup



\### 3.1 Dataset



The project uses the AG News text classification dataset. It contains four classes:



\- World

\- Sports

\- Business

\- Sci/Tech



The original training dataset contains 120,000 examples, and the test dataset contains 7,600 examples. The experiment uses a separate evaluation set containing 100 queries.



\### 3.2 Noisy partitions



Ten partitions were prepared, each containing 12,000 examples. The intended noise rates were:



| Partition | Intended label noise |

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



Symmetric label noise was introduced by replacing a label with one of the other three class labels. The partition verification results showed observed noise rates close to their intended values, with small differences due to random sampling.



\### 3.3 Model and configuration



The experiment uses the following configuration:



\- \*\*Language model:\*\* Qwen/Qwen2.5-0.5B-Instruct

\- \*\*Classification classes:\*\* Four

\- \*\*Demonstrations per query:\*\* Four

\- \*\*Evaluation queries per policy:\*\* 100

\- \*\*Reward:\*\* 1 for a correct prediction and 0 for an incorrect prediction



The saved evaluation results contain no invalid predictions for the reported partition and policy experiments.



\## 4. Methodology



\### 4.1 Fixed-P1 policy



Fixed-P1 always selects partition 1, which has 0% intentionally injected label noise. This provides a baseline for evaluating a fixed selection strategy.



\### 4.2 Random policy



The Random policy selects a partition randomly from the available partitions. It provides a baseline for comparison against an adaptive selection strategy.



\### 4.3 UCB1 policy



The UCB1 policy uses observed rewards to balance exploration and exploitation. It considers both the average reward of a partition and an exploration bonus related to how often that partition has been selected.



The goal is to use past prediction outcomes to guide subsequent partition choices. However, the performance of this strategy depends on the reward feedback and experimental design.



\### 4.4 Evaluation procedure



For each query, the selected partition supplies four demonstration examples. These demonstrations are incorporated into an ICL prompt, and the language model generates a category prediction.



The prediction is compared with the query's true label. A correct prediction receives reward 1; an incorrect prediction receives reward 0.



Accuracy is calculated as:



\*\*Accuracy (%) = (Number of correct predictions / Total evaluation queries) × 100\*\*



Each policy is evaluated on 100 queries, and the resulting metrics are saved in CSV files.



\## 5. Results



\### 5.1 Policy comparison



The saved summary CSV reports the following results:



| Policy | Queries | Correct predictions | Invalid predictions | Accuracy |

|---|---:|---:|---:|---:|

| Fixed-P1 | 100 | 59 | 0 | 59% |

| Random | 100 | 65 | 0 | 65% |

| UCB1 | 100 | 54 | 0 | 54% |



Random achieves the highest observed accuracy at 65%, followed by Fixed-P1 at 59% and UCB1 at 54%.



In this run, Random exceeds Fixed-P1 by 6 percentage points and UCB1 by 11 percentage points. These differences describe only the observed experiment and should not be interpreted as statistically significant without repeated trials and suitable analysis.



\### 5.2 Partition-level evaluation



The saved partition evaluation results are:



| Partition | Intended noise | Correct predictions | Accuracy |

|---|---:|---:|---:|

| P1 | 0% | 59/100 | 59% |

| P2 | 10% | 60/100 | 60% |

| P3 | 20% | 63/100 | 63% |

| P4 | 30% | 60/100 | 60% |

| P5 | 40% | 58/100 | 58% |

| P6 | 50% | 63/100 | 63% |

| P7 | 40% | 62/100 | 62% |

| P8 | 30% | 58/100 | 58% |

| P9 | 20% | 62/100 | 62% |

| P10 | 10% | 63/100 | 63% |



The observed partition accuracy ranges from 58% to 63%. The results do not show a monotonic decrease in accuracy as intended label noise increases. This indicates that the observed outcomes cannot be explained by the intended noise percentage alone. Sampling variability, the selected demonstrations, and model behavior may also contribute.



\### 5.3 Visualizations



Three result visualizations are available in the project's `results` directory:



\- `partition\_accuracy.png` shows the partition-level accuracy results.

\- `bandit\_policy\_comparison.png` compares the saved policy results.

\- `bandit\_noise\_analysis.png` presents an analysis of policy performance in relation to noise.



The CSV files provide the underlying numerical results for further analysis.



\## 6. Discussion



The saved experiment shows that Random performed best among the three evaluated policies, while UCB1 achieved the lowest accuracy. This does not necessarily mean that random selection is generally superior to UCB1. The experiment uses a limited number of queries and only one reported run per policy.



The partition-level results also show that higher intended noise does not always correspond to lower observed accuracy. Label noise is one factor affecting demonstration quality, but the specific examples selected and the language model's response behavior can influence individual outcomes.



A stronger evaluation would repeat the experiments using multiple independent seeds, increase the evaluation set size, and report variability across runs. The reward feedback and selection procedure should also be examined to ensure that the policy comparison is methodologically sound.



\## 7. Limitations



The current study has several limitations:



1\. \*\*Limited evaluation size:\*\* Each policy is evaluated on 100 queries.

2\. \*\*Single reported run:\*\* The results do not measure variability across independent runs.

3\. \*\*Model and prompt dependence:\*\* Outcomes may depend on the language model and prompt construction.

4\. \*\*Randomness:\*\* Demonstration sampling and partition selection may influence performance.

5\. \*\*Statistical uncertainty:\*\* The observed differences have not been established as statistically significant.

6\. \*\*Implementation validation:\*\* The reconstructed experiment runner passed syntax compilation and import checks, but a complete end-to-end run of that reconstructed version has not been verified.



Consequently, the reported numbers should be treated as preliminary experimental observations.



\## 8. Future Work



Future work could include:



\- Running each policy with multiple independent random seeds.

\- Increasing the number of evaluation queries.

\- Reporting confidence intervals and statistical comparisons.

\- Comparing additional bandit algorithms and selection strategies.

\- Evaluating the impact of different noise schedules.

\- Investigating how demonstration selection affects model predictions.

\- Verifying the complete experiment pipeline from data preparation through result generation.



\## 9. Conclusion



This project developed an experimental framework for comparing partition-selection policies for in-context learning under noisy-label conditions. It prepared ten noisy partitions, evaluated partition-level accuracy, and compared Fixed-P1, Random, and UCB1 using the saved results.



The reported accuracies were 59% for Fixed-P1, 65% for Random, and 54% for UCB1. Random achieved the best observed result in this run. However, the limited evaluation size and absence of repeated trials prevent strong conclusions about general performance.



The project provides a starting point for further research into robust demonstration selection and bandit-based strategies for in-context learning.



\## 10. Project Artifacts



The project contains:



\- `README.md` — project overview and instructions.

\- `results/partition\_evaluation.csv` — partition-level evaluation results.

\- `results/bandit\_query\_results.csv` — per-query policy results.

\- `results/bandit\_summary.csv` — policy summary metrics.

\- `results/partition\_accuracy.png` — partition-level accuracy plot.

\- `results/bandit\_policy\_comparison.png` — policy comparison plot.

\- `results/bandit\_noise\_analysis.png` — noise analysis plot.



These artifacts document the saved experiments and provide a basis for further validation and analysis.



