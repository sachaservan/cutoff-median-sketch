import os
import sys
import time
import csv
import argparse
import numpy as np

from itertools import repeat
from multiprocessing import Pool, set_start_method, get_context
from sketches import count_min, count_sketch, median_sketch
from experiment_constants import *

# utils for parsing the dataset and results 
from utils import get_data_aol_query_list
from utils import get_data, get_stat, git_log, feat_to_string, get_data_str_with_ports_list

# setup logging 
import logging
logger = logging.getLogger('learned_estimators_log')
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger.propagate = False
logger.addHandler(logging.FileHandler('experiments/eval.log', 'a'))

def order_y_wkey(y, results, key, n_examples=0):
    logger.info('loading results from %s' % results)
    results = np.load(results)
    pred_prob = results[key].squeeze()
    if n_examples:
        pred_prob = pred_prob[:n_examples]
    idx = np.argsort(pred_prob)[::-1]
    assert len(idx) == len(y)
    return y[idx], pred_prob[idx]

def order_y_wkey_list(y, results_list, key):
    pred_prob = np.array([])
    for results in results_list:
        results = np.load(results)
        pred_prob = np.concatenate((pred_prob, results[key].squeeze()))
    idx = np.argsort(pred_prob)[::-1]
    assert len(idx) == len(y)
    return y[idx], pred_prob[idx]

def load_dataset(dataset, model, key, perfect_oracle=False, is_aol=False, is_synth_zipfian=False, is_synth_pareto=False):

    if is_synth_zipfian:
        N = 200000
        a = 2.5 # zipf param > 1
        data = np.random.zipf(a, N).flatten() + 1
        sort = np.argsort(data)[::-1]
        data = data[sort]
        space_idx = int(len(SYNTHETIC_DATA_ORACLE_SPACE_FACTOR_TO_TEST) / 2)
        scores = count_min(data, int(len(data)*SYNTHETIC_DATA_ORACLE_SPACE_FACTOR_TO_TEST[space_idx]), 1)
        sort = np.argsort(scores)[::-1]
        data = data[sort]
        scores = scores[sort]

        return data, scores

    elif is_synth_pareto:
        N = 1000000
        a = 1 # pareto param > 0
        data = np.random.pareto(a, N).flatten() + 1
        data = data.astype(int) 
        sort = np.argsort(data)[::-1]
        data = data[sort]
        space_idx = int(len(SYNTHETIC_DATA_ORACLE_SPACE_FACTOR_TO_TEST) / 2)
        scores = count_min(data, int(len(data)*SYNTHETIC_DATA_ORACLE_SPACE_FACTOR_TO_TEST[space_idx]), 1)
        sort = np.argsort(scores)[::-1]
        data = data[sort]
        scores = scores[sort]

        return data, scores


    start_t = time.time()
    logger.info("Loading dataset...")
    if is_aol:
        x, y = get_data_aol_query_list(dataset)
    else:
        _, y = get_data_str_with_ports_list(dataset)
    
    logger.info('Done loading datasets (took %.1f sec)' % (time.time() - start_t))

    start_t = time.time()
    logger.info("Loading model...")
    data, oracle_scores = order_y_wkey_list(y, model, key)

    # IP data is stored in log form
    if not is_aol:
        oracle_scores = np.exp(oracle_scores)
    logger.info('Done loading model (took %.1f sec)' % (time.time() - start_t))

    logger.info("///////////////////////////////////////////////////////////")
    logger.info("Dataset propertiess")
    logger.info("Size:        " + str(len(data)))
    logger.info("Data:        " + str(data))
    logger.info("Predictions: " + str(oracle_scores))
    logger.info("///////////////////////////////////////////////////////////")

    if perfect_oracle:
        oracle_scores = data / np.sum(data) # perfect predictions 
        sort = np.argsort(oracle_scores)[::-1]
        oracle_scores = oracle_scores[sort]
        data = data[sort]

    return data, oracle_scores
