Here's a polished version of the README file for your GitHub repository:

---

# NLGCorrEval

This repository contains code for **[Analyzing and Evaluating Correlation Measures in NLG Meta-Evaluation](https://arxiv.org/abs/2410.16834)**.

### Requirements

The `nlpstats` library is essential for this project, and it needs to be installed locally. Some parts of the code have been modified (e.g., the addition of item-level correlation). Other necessary libraries include `pandas`, `numpy`, and others.

To set up the environment and install dependencies:

```bash
cd nlpstats
pip install --editable .
pip install numpy
pip install pandas
```

### Example Usage

#### Data Setup
- Download the dataset from [this link](https://drive.google.com/file/d/1W_m8hWsGL61Lxqd0IaILi-byuu1t_1aD/view?usp=share_link), and place the `data.csv` file under the `DP_RC` directory.
- For sensitivity to score granularity, download from [this link](https://drive.google.com/file/d/1PooFDpgvNhXzqOgDV2u5bzMMdZNp0KjV/view?usp=share_link), and place the `data_all_rescaled.json` file under the `score_granularity` directory.

#### 1. Calculate Ranking Consistency
To calculate ranking consistency, you can use the following command. This will save the individual group results:

```bash
cd DP_RC
python ranking_consistency.py --input-file data.csv --output-file results.json --world-size 32 --number-trials 1000 --save-group-results
```

#### 2. Calculate Discriminative Power Using Permutation Tests
To calculate the discriminative power using permutation tests, use the following command. It will also save the individual group results:

```bash
cd DP_RC
python discriminative_power_permutaion_test.py --input-file data.csv --output-file results.json --world-size 32 --number-trials 1000 --save-group-results
```

#### 3. Re-sample Scores Generated by GPT-3.5/4/4o to Measure Sensitivity to Score Granularity
To measure the sensitivity to score granularity using the scores of GPT-3.5/4/4o, use this command. You can adjust the number of workers with the `--num_workers` argument (default is 4):

```bash
cd score_granularity
python re_sampling.py --data_type summarization --model GPT-3.5 --input_file data_all_rescaled.json --output_file gpt3.5_summ_result.csv --num_workers 4
```