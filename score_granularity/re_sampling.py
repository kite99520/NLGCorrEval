import argparse
import numpy as np
import json
import pandas as pd
from utils import calc_metric_output
from multi_process import multi_process

# Declare summ as a global variable
summ = None

# Function to process input data
def process(inputs, _):
    global summ  # Access the global variable 'summ'
    gpt4_result = []
    max_times = 1000
    for aspect, metric, n in inputs:
        data_aspect = summ[summ["aspect"] == aspect]
        data = data_aspect[["sys_id", "new_seg_id", "human_score"]].copy()
        data_metirc = data_aspect[metric]
        for i in range(max_times):
            out = data_metirc.apply(lambda x: np.random.choice(x, size=n).mean())
            data[metric+"_"+str(i)] = out
        for level in ['system', 'input', 'global', 'item']:
            for corr in ['pearson', 'spearman', 'kendall']:
                scores_map = calc_metric_output(data, level, corr)
                scores = np.array(list(scores_map.values()))
                median = np.median(scores)
                down = np.percentile(scores, q=2.5)
                up = np.percentile(scores, q=97.5)
                gpt4_result.append({
                    "aspect": aspect,
                    "metric": metric,
                    "n": n,
                    "level": level,
                    "corr": corr,
                    "median": median,
                    "per2.5": down,
                    "per97.5": up,
                })
    return gpt4_result

# Argument parser to handle user inputs
def parse_args():
    parser = argparse.ArgumentParser(description="Process Summarization or Translation Data with different evaluation models.")
    parser.add_argument('--data_type', choices=['summarization', 'translation'], required=True,
                        help="Choose 'summarization' for SummEval or 'translation' for google_mqm.")
    parser.add_argument('--model', choices=['GPT-3.5', 'GPT-4-Turbo', 'GPT-4o'], required=True,
                        help="Choose the evaluation model: 'GPT-3.5', 'GPT-4-Turbo', or 'GPT-4o'.")
    parser.add_argument('--input_file', required=True, help="Path to the input JSON file.")
    parser.add_argument('--output_file', required=True, help="Path to save the output CSV file.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for parallel processing.")
    return parser.parse_args()

# Main execution logic
def main():
    global summ  # Declare 'summ' as a global variable
    
    # Parse command-line arguments
    args = parse_args()

    # Load data from the specified input file
    data = json.load(open(args.input_file))
    all_metric_names = list(data[0]['metric_scores'].keys())
    all_metric_names = [x for x in all_metric_names if not ('GPT' in x and 'all' not in x)]

    # Create DataFrame
    df = pd.DataFrame(data)
    for metric_name in all_metric_names:
        df[metric_name] = df.apply(lambda row: row['metric_scores'][metric_name], axis=1)

    # Filter based on dataset type
    if args.data_type == 'summarization':
        summ = df.query('dataset == "SummEval"')
    else:
        summ = df.query('dataset == "google_mqm"')

    # Set the model-specific metrics based on the chosen model
    if args.model == 'GPT-3.5':
        metrics = ['GPT3.5_T=1_1_5_all', 'GPT3.5_T=1_1_10_all', 'GPT3.5_T=1_0_100_all']
    elif args.model == 'GPT-4-Turbo':
        metrics = ['GPT4_T=1_1_5_all', 'GPT4_T=1_1_10_all', 'GPT4_T=1_0_100_all']
    else:  # GPT-4o
        metrics = ['GPT4o_T=1_1_5_all', 'GPT4o_T=1_1_10_all', 'GPT4o_T=1_0_100_all']

    # Prepare the input combinations for multi-processing
    inputs = []
    for aspect in ['coherence', 'consistency', 'fluency', 'relevance']:
        for metric in metrics:
            for n in range(1, 11):
                inputs.append((aspect, metric, n))

    # Run the multi-processing logic with the specified number of workers
    gpt4_result = multi_process(inputs, process, args.num_workers)
    
    # Save the result to the specified output file
    pd.DataFrame(gpt4_result).to_csv(args.output_file)
    print(f"Results saved to {args.output_file}")

# Run the main function if the script is executed directly
if __name__ == '__main__':
    main()
