import copy
import numpy as np
import pandas as pd
from nlpstats.correlations.correlations import correlate




def calc_metric_output(data: pd.DataFrame, level='system', coefficient='kendall'):
    '''
        return a map, each element indicates the correlation score of an evaluation metric
    '''

    sys_num = len(set(data['sys_id']))
    input_num = len(set(data['new_seg_id']))
    assert sys_num * input_num == len(data)

    sorted_data = data.sort_values(by=["sys_id", "new_seg_id"])
    metircs = list(set(data.columns) - set(["sys_id", "new_seg_id", "human_score"]))

    human_output = sorted_data["human_score"].to_numpy().reshape((sys_num, input_num))

    scores_map = {}
    for metric in metircs:
        corr_score = correlate(sorted_data[metric].to_numpy().reshape((sys_num, input_num)), human_output, level, coefficient)
        scores_map[metric] = corr_score

    return scores_map