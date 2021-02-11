import os
import sys
import time
import argparse
import random
import math
import numpy as np
from scipy.optimize import minimize

def log(x, b):
    '''
    convinience function for computing log in arbitrary base b
    '''
    return  math.log2(x) / math.log2(b)

def random_hash(y, n_buckets):
    '''
    randomly assign items in y into n_buckets
    '''
    counts = np.zeros(n_buckets)
    y_buckets = np.random.choice(np.arange(n_buckets), size=len(y))
    for i in range(len(y)):
        counts[y_buckets[i]] += y[i]
    return counts, y_buckets

def random_hash_with_sign(y, n_buckets):
    '''
    randomly assign items in y into n_buckets, randomly pick a sign for each item
    '''
    counts = np.zeros(n_buckets)
    y_buckets = np.random.choice(np.arange(n_buckets), size=len(y))
    y_signs = np.random.choice([-1, 1], size=len(y))
    for i in range(len(y)):
        counts[y_buckets[i]] += (y[i] * y_signs[i])
    return counts, y_buckets, y_signs
    
def space_needed_for_median_estimate(values, num_samples):
    # Space needed: 
    # U = universe of items = 2^32 (int)
    # 64 registers (1 byte per register) for global HLL used to keep track of total number of distinct elements 
    # 2 registers (1 byte per register) for each DE_j for j = 1 ... log U ==> 2 * 32 bytes
    # 8 bytes for UE_j for j = 1 ... log U ==> 8 * 32 bytes 
    # DE_j and UE_j are required *per sample* that we need ==> everything multiplied by num_samples 
    # Assumption: 1/2 of the samples FAIL (in expectation) so need to multiply num_samples by 2
    # NOTE: assumption can be verified by running median_estimate 

    total_space = 0
    total_space += 2 * 32 # space needed for DE_j 
    total_space += 8 * 32  # space needed for UE_j 
    total_space *= 2       # accounting for failure rate 
    total_space *= num_samples
    
    total_space += 64  # global HLL 

    return total_space


def simulate_median_estimate(values, num_samples=10):
    '''
    Like median estimate but much faster; ok for experiments 
    '''
    indices = np.random.choice(len(values), num_samples, replace=False)
    sample = np.take(values, indices)
    return np.median(sample), indices

def median_estimate(values, num_samples=10):

    # estimate for total number of distinct elements
    distinct_element_estimate = simulate_hyperloglog(len(values), 128) 

    space_usage = 0
    unique = []
    for i in range(num_samples):
        u = get_unique_element(values, distinct_element_estimate)
        if u != -1:
            unique.append(u)
        
    return np.median(unique)
        

def get_unique_element(values, distinct_element_estimate):

    lv = math.ceil(math.log2(len(values)))
    counters = np.zeros(lv) # unique element counters 
    numvalues = np.zeros(lv) # hack: keep track of exact collision count then run HLL to estimate 
    
    probs = np.zeros(lv)
    for j in range(lv):
        probs[j] = 2**j
    
    for v in values:
        for j in range(lv):
            rand = random.randint(0, probs[j]) # sample 0 with probability 2^-j
            if rand == 0:
                counters[j] += v
                numvalues[j] += 1 # keep track of how many items hit 
    
    for j in range(lv):
        # undo the hack above by estimating the number of distinct elements rather
        # than keeping track of them 
        if numvalues[j] != 0:
            numvalues[j] = math.ceil(simulate_hyperloglog(int(numvalues[j]), 2)) - 1

    unique = -1
    j = math.ceil(math.log2(distinct_element_estimate))
    j = min(j, lv-1)
    if numvalues[j] == 1:
        unique = counters[j]
    
    return unique

def simulate_hyperloglog(n, m): 
    '''
    computes the estimated number of unique values for a set of n values using m 
    byte registers. see https://en.wikipedia.org/wiki/HyperLogLog
    '''

    assert math.log2(m).is_integer() # m must be a power of two

    lm = int(math.log2(m))

    assert lm < 64 # the "hash" assumes 128 bits of which the first lm are used

    # compute optimal constant based on number of registers
    if m == 16:
        a = 0.673
    elif m == 32:
        a = 0.697
    elif m == 64:
        a = 0.709
    else:
        a = 0.7213 / (1 + 1.079/m) # correction value

    registers = np.zeros(m, dtype=int)

    # don't take about the item b/c y is assumed to be sorted
    # and only contain unique values
    for _ in range(n): 
        # y contains only unique values; just pick a random hash output for it
        bval = np.random.choice(np.arange(2), size=m) 
        regidx = 0
        for i in range(0, lm): # compute register index
            regidx += (1 << i) * bval[i]

        msbidx = 0 # index (+1) of most significant bit in the hash
        for i in range(len(bval[lm:])):
            if bval[lm:][i] == 1:
                msbidx = i + 1
                break

        # set to the max beteween current register and new value
        if msbidx > registers[regidx]:
            registers[regidx] = msbidx
    
    # hyperloglog estimate
    z = 0.0
    for i in range(len(registers)): 
        s = (1 << registers[i])
        z += 1.0/s

    E = a * (m**2) / z 


    # see http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf 
    # for range correction (p. 140)

    # small range correction 
    if E <= (5/2) * m: 
        v = 0
        for i in range(m):
            if registers[i] == 0:
                v += 1
        if v != 0:
            E = m * math.log2(m/v) 

    # no intermediate range correction (just E)

    # large range correction 
    if E > 2**32 / 30:
        E = -2**32 * math.log2(1-E/2**32)

    return E 