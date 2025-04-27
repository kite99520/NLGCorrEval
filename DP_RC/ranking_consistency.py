import argparse
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import random
from scipy.stats import kendalltau

from nlpstats.correlations.correlations import correlate
from multi import Processer, pcount

# Ignore warnings
warnings.filterwarnings("ignore")

# Command-line argument parsing function
def parse_args():
    parser = argparse.ArgumentParser(description="Calculate ranking consistency")
    parser.add_argument(
        '--input-file', 
        type=str, 
        required=True, 
        help="Path to the input CSV file containing the dataset."
    )
    parser.add_argument(
        '--output-file', 
        type=str, 
        default='rank_consistency.json',
        help="Path to the output JSON file where results will be stored."
    )
    parser.add_argument(
        '--world-size', 
        type=int, 
        default=32, 
        help="Number of processes to run in parallel."
    )
    parser.add_argument(
        '--number-trials', 
        type=int, 
        default=1000, 
        help="Number of trials for sampling."
    )
    parser.add_argument(
        '--save-group-results', 
        action='store_true', 
        help="Flag to indicate whether to save results for each group/dataset individually."
    )
    return parser.parse_args()

# Define the global variables and metric list
metric_list = ['bertscore_p', 'bertscore_r', 'bertscore_f',
       'bartscore_s_h', 'bartscore_h_r', 'bartscore_r_h', 'bleu', 'comet',
       'moverscore', 'rouge1', 'rouge2', 'rougeL', 'chrf', 'bleurt',
       'GPT3.5_T=0_1_5', 'GPT3.5_T=0_1_10', 'GPT3.5_T=0_0_100',
       'GPT3.5_T=1_1_5', 'GPT3.5_T=1_1_10', 'GPT3.5_T=1_0_100', 'GPT4_T=0_1_5',
       'GPT4_T=0_1_10', 'GPT4_T=0_0_100', 'GPT4_T=1_1_5', 'GPT4_T=1_1_10',
       'GPT4_T=1_0_100', 'GPT4o_T=0_1_5', 'GPT4o_T=0_1_10', 'GPT4o_T=0_0_100',
       'GPT4o_T=1_1_5', 'GPT4o_T=1_1_10', 'GPT4o_T=1_0_100']

levels = ['system', 'input', 'global', 'item']
coefficients = ['pearson', 'spearman', 'kendall']

# Function to calculate correlation scores between metrics
def cal_metric_output(data, level='system', coefficient='kendall'):
    sys_ids = sorted(set(d['sys_id'] for d in data))
    input_ids = sorted(set(d['new_seg_id'] for d in data))

    sys_num = len(sys_ids)
    input_num = len(input_ids)

    sys2id = {sys:i for i, sys in enumerate(sys_ids)}
    input2ids = {input_:i for i, input_ in enumerate(input_ids)}

    human_output = np.zeros((sys_num, input_num))
    for d in data:
        human_output[sys2id[d['sys_id']]][input2ids[d['new_seg_id']]] = d['human_score']

    metric2output = dict()
    for metric in metric_list:
        metric_output = np.zeros((sys_num, input_num))
        for d in data:
            metric_output[sys2id[d['sys_id']]][input2ids[d['new_seg_id']]] = d[metric]
        metric2output[metric] = metric_output

    scores_list = []
    for metric in metric_list:
        corr_score = correlate(metric2output[metric], human_output, level, coefficient)
        scores_list.append(corr_score)

    return scores_list

# Define a custom class for processing data
class P(Processer):
    def __init__(self):
        self.res_dict = defaultdict(list)

    def merge(self, other):
        for k, v in other.res_dict.items():
            self.res_dict[k].extend(v)

    def out(self):
        return self.res_dict
        
    def count(self, inputs, id):
        for data_1, data_2 in inputs:
            for level in levels:
                for coefficient in coefficients:
                    scores_list_1 = cal_metric_output(data_1, level=level, coefficient=coefficient)
                    scores_list_2 = cal_metric_output(data_2, level=level, coefficient=coefficient)
                    corr_between, p_value = kendalltau(scores_list_1, scores_list_2)
                    self.res_dict['{}_{}'.format(level, coefficient)].append(corr_between)

# Main function
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

        sys_num_total = len(sys_ids_total)
        input_num_total = len(input_ids_total)

        g_dict = g.to_dict('records')

        # Create sample data for permutation testing
        samples = []
        for i in range(args.number_trials):
            ids = list(range(input_num_total))
            random.shuffle(ids)

            ids_1 = ids[:(input_num_total // 2)]
            ids_2 = ids[(input_num_total // 2):]

            input_names_1 = [input_ids_total[i] for i in ids_1]
            input_names_2 = [input_ids_total[i] for i in ids_2]

            data_1 = [d for d in g_dict if d['new_seg_id'] in input_names_1]
            data_2 = [d for d in g_dict if d['new_seg_id'] in input_names_2]
            samples.append((data_1, data_2))

        # Perform multi-threading to collect consistency results for different correlations
        rank_consisteny_dict = pcount(samples, P, args.world_size)

        for k, v in rank_consisteny_dict.items():
            level, coefficient = k.split('_')
            item = {
                'dataset': n[0],
                'ori_dataset': n[1],
                'aspect': n[2],
                'level': level,
                'coefficient': coefficient,
                'rank_consistency_values': v
            }
            all_results.append(item)
            dataset_result.append(item)

        # Save the results for the current group into a JSON file if save-group-results is True
        if args.save_group_results:
            group_output_file = f'ranking_consistency_{n[0]}_{n[1].replace("/", "")}_{n[2]}.json'
            with open(group_output_file, 'w') as f:
                json.dump(dataset_result, f, ensure_ascii=False, indent=2)

    # Save all results in a global JSON file
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
