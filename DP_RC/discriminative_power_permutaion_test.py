
import warnings
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from nlpstats.correlations.permutation import permutation_test
from multi import Processer, pcount

# Ignore warnings
warnings.filterwarnings("ignore")

# Class for processing results and performing statistical analysis
class P(Processer):
    def __init__(self):
        self.res_dict = defaultdict(list)

    def merge(self, other):
        for k, v in other.res_dict.items():
            self.res_dict[k].extend(v)

    def out(self):
        return self.res_dict
        
    def count(self, inputs, id):
        for metric_i, metric_j, human_output in inputs:
            for level in levels:
                for coefficient in coefficients:
                    res = permutation_test(metric_i, metric_j, human_output, level, coefficient, 'both', n_resamples=999)
                    self.res_dict['{}_{}'.format(level, coefficient)].append(res.pvalue)

# Define the command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Calcualte DP values using permutation testsw.")
    parser.add_argument(
        '--input-file', 
        type=str, 
        required=True, 
        help="Path to the input CSV file containing the dataset."
    )
    parser.add_argument(
        '--output-file', 
        type=str, 
        default='discriminative_power_permutation_test.json',
        help="Path to the output JSON file where results will be stored."
    )
    parser.add_argument(
        '--world-size', 
        type=int, 
        default=32, 
        help="Number of processes to run in parallel."
    )
    parser.add_argument(
        '--save-group-results', 
        action='store_true', 
        help="Flag to indicate whether to save results for each group/dataset individually."
    )
    return parser.parse_args()

# Configuration and dataset information
metric_list = [
    'bertscore_p', 'bertscore_r', 'bertscore_f', 'bartscore_s_h', 'bartscore_h_r',
    'bartscore_r_h', 'bleu', 'comet', 'moverscore', 'rouge1', 'rouge2', 'rougeL',
    'chrf', 'bleurt', 'GPT3.5_T=0_1_5', 'GPT3.5_T=0_1_10', 'GPT3.5_T=0_0_100',
    'GPT3.5_T=1_1_5', 'GPT3.5_T=1_1_10', 'GPT3.5_T=1_0_100', 'GPT4_T=0_1_5',
    'GPT4_T=0_1_10', 'GPT4_T=0_0_100', 'GPT4_T=1_1_5', 'GPT4_T=1_1_10',
    'GPT4_T=1_0_100', 'GPT4o_T=0_1_5', 'GPT4o_T=0_1_10', 'GPT4o_T=0_0_100',
    'GPT4o_T=1_1_5', 'GPT4o_T=1_1_10', 'GPT4o_T=1_0_100'
]

metric_number = len(metric_list)
levels = ['system', 'input', 'global', 'item']
coefficients = ['pearson', 'spearman', 'kendall']

def main():
    # Parse command-line arguments
    args = parse_args()

    # Read the input dataset
    df = pd.read_csv(args.input_file)

    all_results = []

    # Iterate over groups in the dataset based on dataset, original dataset, and aspect
    for n, g in tqdm(df.groupby(by=['dataset', 'ori_dataset', 'aspect'], dropna=False)):
        print(f"Processing group: {n}")

        dataset_result = []
        sys_ids_total = sorted(set(g['sys_id']))
        input_ids_total = sorted(set(g['new_seg_id']))

        # Number of systems and inputs
        sys_num_total = len(sys_ids_total)
        input_num_total = len(input_ids_total)

        # Create mapping of system and input ids to indices
        sys2id = {sys: i for i, sys in enumerate(sys_ids_total)}
        input2ids = {input_: i for i, input_ in enumerate(input_ids_total)}

        # Convert group DataFrame to a dictionary of records
        g_dict = g.to_dict('records')

        # Initialize human_output matrix
        human_output = np.zeros((sys_num_total, input_num_total))
        for d in g_dict:
            human_output[sys2id[d['sys_id']]][input2ids[d['new_seg_id']]] = d['human_score']

        # Initialize metric outputs
        metric2output = {}
        for metric in metric_list:
            metric_output = np.zeros((sys_num_total, input_num_total))
            for d in g_dict:
                metric_output[sys2id[d['sys_id']]][input2ids[d['new_seg_id']]] = d[metric]
            metric2output[metric] = metric_output

        # Create pairs of metrics to compare
        samples = []
        for i in range(0, metric_number - 1):
            for j in range(i + 1, metric_number):
                metric_i = metric2output[metric_list[i]]
                metric_j = metric2output[metric_list[j]]
                samples.append((metric_i, metric_j, human_output))

        # Perform permutation tests and collect p-values
        p_value_list_dict = pcount(samples, P, args.world_size)

        # Store results for the current dataset
        for k, v in p_value_list_dict.items():
            level, coefficient = k.split('_')
            item = {
                'dataset': n[0],
                'ori_dataset': n[1],
                'aspect': n[2],
                'level': level,
                'coefficient': coefficient,
                'p_value_list': v
            }
            all_results.append(item)
            dataset_result.append(item)

        # Save the results for the current group into a JSON file if save-group-results is True
        if args.save_group_results:
            group_output_file = f'discriminative_power_permutation_test_{n[0]}_{n[1].replace("/", "")}_{n[2]}.json'
            with open(group_output_file, 'w') as f:
                json.dump(dataset_result, f, ensure_ascii=False, indent=2)

    # Save all results in a global JSON file
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
