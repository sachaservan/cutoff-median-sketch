#!/usr/bin/env python
# encoding: utf-8
"""
Constants.py
"""

ALGO_TYPE_CUTOFF_MEDIAN = "LEARNED_CUTOFF_AND_MEDIAN"

CUTOFF_FRAC_TO_TEST = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# standard deviation in the oracle error (error sampled from normal distribution)
# when evaluating the synthetic data 
SYNTHETIC_DATA_ORACLE_SPACE_FACTOR_TO_TEST = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.6, 0.75, 0.8, 1, 1.25, 1.5, 1.75, 2, 3, 4, 5, 10]
SYNTHETIC_DATA_CUTOFF_FRAC_TO_TEST = CUTOFF_FRAC_TO_TEST

MODEL_SIZE_AOL = 0.015 # amortized model sizes (in MB)
MODEL_SIZE_IP =  0.5

COUNT_SKETCH_OPTIMAL_N_HASH = 3
COUNT_MIN_OPTIMAL_N_HASH = 2
FRAC_SAMPLES_FOR_MEDIAN_ESTIMATE = 0.0005 # fraction of samples to take 
CUTOFF_SPACE_COST_FACTOR = 2 # need 2x buckets to store in cutoff 

N_REGISTERS_FOR_HLL=128