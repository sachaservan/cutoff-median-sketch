import os
import sys
import time
import csv
import argparse
import math 
import numpy as np
import copy 

from halo import Halo
from itertools import repeat
from multiprocessing import Pool, set_start_method, get_context
from sketches import count_min, count_sketch
from sketch_common import space_needed_for_median_estimate, simulate_median_estimate, estimate_zipf_param, simulate_hyperloglog

from dataset import load_dataset

# constants used in experiments 
from experiment_constants import *

# setup logging 
import logging
logger = logging.getLogger('learned_estimators_log')

#################################################
# setup logging 
#################################################
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger.propagate = False
logger.addHandler(logging.FileHandler('experiments/eval.log', 'a'))

is_sorted = lambda a: np.all(a[:-1] <= a[1:])


##################################################$
# each algorithm variant used in the experiments #
###################################################

def run_count_min(y, space, sanity_check_space_bound):
    n_buckets = int(space / COUNT_MIN_OPTIMAL_N_HASH)
    n_hashes = COUNT_MIN_OPTIMAL_N_HASH
    estimates = count_min(y, n_buckets, n_hashes)

    if n_buckets * n_hashes > sanity_check_space_bound:
        print("[sanity check failed] TOTAL SPACE (vanilla count min) = " + str(n_buckets * n_hashes) + " > " + str(sanity_check_space_bound))
        exit(0)

    logger.info("Count min space_used / space_allocated = " + str(n_buckets * n_hashes/ sanity_check_space_bound))


    return estimates

def run_count_sketch(y, space, sanity_check_space_bound):
    n_buckets = int(space / COUNT_SKETCH_OPTIMAL_N_HASH)
    n_hashes = COUNT_SKETCH_OPTIMAL_N_HASH
    estimates = count_sketch(y, n_buckets, n_hashes)

    if n_buckets * n_hashes > sanity_check_space_bound:
        print("[sanity check failed] TOTAL SPACE (vanilla count sketch) = " + str(n_buckets * n_hashes) + " > " + str(sanity_check_space_bound))
        exit(0)

    logger.info("Count Sketch space_used / space_allocated = " + str(n_buckets * n_hashes/ sanity_check_space_bound))


    return estimates

# 1) store cutoff_threshold items in a table and report their *exact* counts
# 2) store all items beyond the threshold in a count sketch and report the sketch counts
def run_cutoff_count_sketch(y, y_scores, y_test_scores, space_for_sketch, cutoff_threshold, sanity_check_space_bound): 
    if not is_sorted(y_scores[::-1]):
        print("scores not sorted; is everything ok?")
        exit(0)

    if not is_sorted(y_test_scores[::-1]):
        print("scores not sorted; is everything ok?")
        exit(0)

    # finds first index of value less than y_test_scores[cutoff_threshold]
    # https://stackoverflow.com/questions/16243955/numpy-first-occurrence-of-value-greater-than-existing-value 
    cut = np.argmax(y_scores < y_test_scores[cutoff_threshold]) 
    cut = min(cut, cutoff_threshold) # in case there are more values than we can store in the table

    y_cutoff = y[:cut]
    y_noncutoff = y[cut:] # all items that have a predicted score < cutoff_thresh
    table_estimates = np.array(y_cutoff) # store exact counts for all 

    sketch_estimates = run_count_sketch(y_noncutoff, space_for_sketch, space_for_sketch)

    # prepend the table estimates to the count sketch estimates 
    all_estimates = table_estimates.tolist() + sketch_estimates.tolist()

    space_used = space_for_sketch + cut * CUTOFF_SPACE_COST_FACTOR
    if space_used > sanity_check_space_bound:
        print("[sanity check failed] TOTAL SPACE (cutoff count sketch) = " + str(space_used) + " > " + str(sanity_check_space_bound))
        exit(0)

    logger.info("Cutoff Count Sketch space_used / space_allocated = " + str(space_used / sanity_check_space_bound))

    
    return all_estimates
  

# 1) store cutoff_threshold items in a table and report their *exact* counts
# 2) for all other items, report their (normalized) predicted frequency 
# use_exact_sum = True will not estimate the sum on the fly and instead compute the *exact* sum when normalizing predictions
# use_median_as_prediction = True will output the median of all non-cutoff values as the prediction for non-cutoff items 
def run_cutoff_median_sketch(y, y_scores, space, sanity_check_space_bound, is_perfect_oracle=False): 

    if not is_sorted(y_scores[::-1]) and not is_perfect_oracle:
        print("scores not sorted; is everything ok?")
        exit(0)

    num_samples = int(len(y) * FRAC_SAMPLES_FOR_MEDIAN_ESTIMATE)

    # determine a bound on the space needed to compute a median estimate 
    space_needed = space_needed_for_median_estimate(len(y), num_samples) / 4  # space is in units of 4 bytes 

    # not enough space to properly compute the median; just output 1 
    if space < space_needed:
        table_cutoff = int(space / CUTOFF_SPACE_COST_FACTOR)
        cutoff = y[:table_cutoff]
        noncutoff = y[table_cutoff:]
        all_estimates = cutoff.tolist() + np.ones(len(noncutoff)).tolist()
        return all_estimates


    # allocate space for the cutoff table 
    # item | count (can just keep 4 byte "min score" so no need to store scores)
    table_cutoff = int((space - space_needed) / CUTOFF_SPACE_COST_FACTOR)
   
    # split into cutoff and noncutoff items 
    cutoff = y[:table_cutoff]
    noncutoff = y[table_cutoff:]

    # exact estimates for these items   
    cutoff_estimates = cutoff 

    space_used = table_cutoff*CUTOFF_SPACE_COST_FACTOR + space_needed
    if space_used > sanity_check_space_bound:
        print("[sanity check failed] TOTAL SPACE (learned count sketch) = " + str(space_used) + " > " + str(sanity_check_space_bound))
        exit(0)

    logger.info("Learned Sketch space_used / space_allocated = " + str(space_used / sanity_check_space_bound))

    # estimate the median of the non-cutoff items 
    median_est, indices = simulate_median_estimate(noncutoff, num_samples)

    # output the median as the prediction for all non-cutoff values 
    pred_estimates = np.ones(len(noncutoff)) * median_est

    # median estimate keeps counters for all items it selects so we get to 
    # estimate those frequencies "for free"
    pred_estimates[indices] = np.take(noncutoff, indices)

    # concat cutoff and non-cutoff predictions 
    all_estimates = cutoff_estimates.tolist() + pred_estimates.tolist()
    
    return all_estimates


def experiment_comapre_loss(
    algo_type,
    space_list,
    data,
    oracle_scores,
    test_oracle_scores,
    space_allocations,
    best_cutoff_thresh_count_sketch_weighted,
    best_cutoff_thresh_count_sketch_absolute,
    best_cutoff_thresh_count_sketch_relative,
    num_trials,
    n_workers, 
    save_folder, 
    save_file):

    true_counts = data.copy()

    # learned algo with cutoff 
    logger.info("Running learned count sketch")

    # results across all trials 
    algo_predictions_all = []
    cutoff_count_sketch_predictions_weighted_all = []
    cutoff_count_sketch_predictions_absolute_all = []
    cutoff_count_sketch_predictions_relative_all = []
    count_sketch_prediction_all = []
    count_min_prediction_all = []

    for trial in range(num_trials):

        # vanilla count sketch
        logger.info("Running vanilla count sketch on all parameters...")
        spinner = Halo(text='Evaluating vanilla count sketch algorithm', spinner='dots')
        spinner.start()
        with get_context("spawn").Pool() as pool:
            count_sketch_prediction = pool.starmap(
                run_count_sketch, zip(repeat(data), space_allocations, space_allocations))
            pool.close()
            pool.join()

        spinner.stop()

        count_sketch_prediction_all.append(count_sketch_prediction)

        # vanilla count min
        logger.info("Running vanilla count min on all parameters...")
        spinner.stop()
        spinner = Halo(text='Evaluating vanilla count min algorithm', spinner='dots')
        spinner.start()
        with get_context("spawn").Pool() as pool:
            count_min_prediction = pool.starmap(
                run_count_min, zip(repeat(data), space_allocations, space_allocations))
            pool.close()
            pool.join()

        spinner.stop()

        count_min_prediction_all.append(count_min_prediction)

        ########################################################

        algo_predictions = []
        cutoff_count_sketch_predictions_weighted = []
        cutoff_count_sketch_predictions_absolute = []
        cutoff_count_sketch_predictions_relative = []

        spinner = Halo(text='Evaluating learned predictions algorithm', spinner='dots')
        spinner.start()

        if algo_type == ALGO_TYPE_CUTOFF_MEDIAN:
            # learned algorithm with cutoff 
            with get_context("spawn").Pool() as pool:
                algo_predictions = pool.starmap(
                    run_cutoff_median_sketch, 
                    zip(repeat(data), 
                    repeat(oracle_scores), 
                    copy.deepcopy(space_allocations),
                    copy.deepcopy(space_allocations),
                    repeat(False)))
                pool.close()
                pool.join()
        
        algo_predictions_all.append(algo_predictions)

        ########################################################

        # vanilla sketch + cutoff 
        # NOTE: need to evaluate this with each cutoff threshold 
        space_allocations_cutoff_weighted = []
        space_allocations_cutoff_absolute = []
        space_allocations_cutoff_relative = []
        for i, space in enumerate(space_allocations):
            space_cutoff_weighted = space - best_cutoff_thresh_count_sketch_weighted[i] * CUTOFF_SPACE_COST_FACTOR # ID | count (4 bytes each)
            space_cutoff_absolute = space - best_cutoff_thresh_count_sketch_absolute[i] * CUTOFF_SPACE_COST_FACTOR # ID | count (4 bytes each)
            space_cutoff_relative = space - best_cutoff_thresh_count_sketch_relative[i] * CUTOFF_SPACE_COST_FACTOR # ID | count (4 bytes each)

            space_allocations_cutoff_weighted.append(space_cutoff_weighted)
            space_allocations_cutoff_absolute.append(space_cutoff_absolute)
            space_allocations_cutoff_relative.append(space_cutoff_relative)

        # 1.  evaluate cutoff count sketch on weighted error 
        logger.info("Running cutoff count sketch on all parameters for weighted error...")
        spinner.stop()
        spinner = Halo(text='Evaluating cutoff count sketch algorithm for weighted error', spinner='dots')
        spinner.start()
        with get_context("spawn").Pool() as pool:
            cutoff_count_sketch_predictions_weighted = pool.starmap(
                run_cutoff_count_sketch, 
                zip(repeat(data), repeat(oracle_scores), repeat(test_oracle_scores), space_allocations_cutoff_weighted, best_cutoff_thresh_count_sketch_weighted, space_allocations))
            pool.close()
            pool.join()


        # 2.  evaluate cutoff count sketch on absolute error 
        logger.info("Running cutoff count sketch on all parameters for absolute error...")
        spinner.stop()
        spinner = Halo(text='Evaluating cutoff count sketch algorithm for absolute error', spinner='dots')
        spinner.start()
        with get_context("spawn").Pool() as pool:
            cutoff_count_sketch_predictions_absolute = pool.starmap(
                run_cutoff_count_sketch, 
                zip(repeat(data), repeat(oracle_scores), repeat(test_oracle_scores), space_allocations_cutoff_absolute, best_cutoff_thresh_count_sketch_absolute, space_allocations))
            pool.close()
            pool.join()

        # 3.  evaluate cutoff count sketch on relative error 
        logger.info("Running cutoff count sketch on all parameters for relative error...")
        spinner.stop()
        spinner = Halo(text='Evaluating cutoff count sketch algorithm for relative error', spinner='dots')
        spinner.start()
        with get_context("spawn").Pool() as pool:
            cutoff_count_sketch_predictions_relative = pool.starmap(
                run_cutoff_count_sketch, 
                zip(repeat(data), repeat(oracle_scores), repeat(test_oracle_scores), space_allocations_cutoff_relative, best_cutoff_thresh_count_sketch_relative, space_allocations))
            pool.close()
            pool.join()
        spinner.stop()

        cutoff_count_sketch_predictions_weighted_all.append(cutoff_count_sketch_predictions_weighted)
        cutoff_count_sketch_predictions_absolute_all.append(cutoff_count_sketch_predictions_absolute)
        cutoff_count_sketch_predictions_relative_all.append(cutoff_count_sketch_predictions_relative)


    #################################################################
    # save all results to the folder
    #################################################################
    np.savez(os.path.join(save_folder, save_file),
        space_list=space_list,
        true_values=true_counts,
        oracle_predictions=oracle_scores,
        algo_predictions=algo_predictions_all,
        cutoff_count_sketch_predictions_weighted=cutoff_count_sketch_predictions_weighted_all,
        cutoff_count_sketch_predictions_absolute=cutoff_count_sketch_predictions_absolute_all,
        cutoff_count_sketch_predictions_relative=cutoff_count_sketch_predictions_relative_all,
        count_sketch_predictions=count_sketch_prediction_all,
        count_min_predictions=count_min_prediction_all)

def experiment_comapre_loss_vs_oracle_error_on_synthetic_data(
    algo_type,
    space,
    data,
    n_trials,
    n_workers, 
    save_folder, 
    save_file):

    # learned algo with cutoff 
    logger.info("Running learned count sketch on synthetic data with different prediction errors")

    algo_mean_absolute_error_per_oracle_error = []
    algo_std_absolute_error_per_oracle_error = []
    cutoff_absolute_error_per_oracle_error = []
    
    algo_mean_relative_error_per_oracle_error = []
    algo_std_relative_error_per_oracle_error = []
    cutoff_relative_error_per_oracle_error = []
    
    algo_mean_weighted_error_per_oracle_error = []
    algo_std_weighted_error_per_oracle_error = []
    cutoff_weighted_error_per_oracle_error = []

    # all the data / scores we test 
    eval_data = []
    eval_scores = []

    # random oracle 
    np.random.shuffle(data)
    scores = np.asarray(range(0, len(data)))[::-1]
    eval_data.append(data.copy())
    eval_scores.append(scores.copy())

    spinner = Halo(text='Computing oracle scores...', spinner='dots')
    spinner.start()

    for space_factor in SYNTHETIC_DATA_ORACLE_SPACE_FACTOR_TO_TEST:
        # oracle error = countmin error on allocated space 
        oracle_scores = count_min(data.copy(), int(len(data)*space_factor), 1)
        
        sort = np.argsort(oracle_scores)[::-1]
        eval_data.append(data.copy()[sort])
        eval_scores.append(scores.copy())

    # perfect oracle 
    sort = np.argsort(data)[::-1]
    eval_data.append(data.copy()[sort])
    eval_scores.append(scores.copy())

    spinner.stop()
    spinner = Halo(text='Evaluating algo on parameters...', spinner='dots')
    spinner.start()

    algo_predictions_per_trial_per_error = []

    for trial in range(n_trials):
        algo_predictions_trial = []
        # learned algorithm with cutoff 
        with get_context("spawn").Pool() as pool:
            algo_predictions_trial = pool.starmap(
                run_cutoff_median_sketch, 
                zip(eval_data.copy(), 
                eval_scores.copy(), 
                repeat(space),
                repeat(space),
                repeat(False)))
            pool.close()
            pool.join()
        
        algo_predictions_per_trial_per_error.append(np.array(algo_predictions_trial))


    spinner.stop()
    spinner = Halo(text='Computing errors...', spinner='dots')
    spinner.start()

    i = 0
    for d in eval_data:

        algo_absolute_error_per_trial = []
        algo_relative_error_per_trial = []
        algo_weighted_error_per_trial = []

        for trial in range(n_trials):
            abs_error = np.abs(algo_predictions_per_trial_per_error[trial][i] - np.array(d))
            rel_error = abs_error / np.array(d) 
            weighted_error = abs_error * np.array(d) 
          
            algo_absolute_error_per_trial.append(np.sum(abs_error))
            algo_relative_error_per_trial.append(np.sum(rel_error))
            algo_weighted_error_per_trial.append(np.sum(weighted_error))
        

        algo_mean_absolute_error_per_oracle_error.append(np.mean(algo_absolute_error_per_trial))
        algo_mean_relative_error_per_oracle_error.append(np.mean(algo_relative_error_per_trial))
        algo_mean_weighted_error_per_oracle_error.append(np.mean(algo_weighted_error_per_trial))
      
        algo_std_absolute_error_per_oracle_error.append(np.std(algo_absolute_error_per_trial))
        algo_std_relative_error_per_oracle_error.append(np.std(algo_relative_error_per_trial))
        algo_std_weighted_error_per_oracle_error.append(np.std(algo_weighted_error_per_trial))

        i += 1

    
    spinner.stop()
    spinner = Halo(text='Evaluating cutoff count sketch on parameters...', spinner='dots')
    spinner.start()

    best_cutoff_absolute_eror_per_oracle_error = []
    best_cutoff_relative_eror_per_oracle_error = []
    best_cutoff_weighted_eror_per_oracle_error = []

    i = 0
    for d in eval_data:

        cutoff_fracs = np.array(SYNTHETIC_DATA_CUTOFF_FRAC_TO_TEST) * (space/CUTOFF_SPACE_COST_FACTOR)
        cutoff_fracs = cutoff_fracs.astype(int)
        space_for_sketch = np.ones(len(cutoff_fracs)) * space - cutoff_fracs*CUTOFF_SPACE_COST_FACTOR

        cutoff_absolute_error_per_cutoff = []
        cutoff_relative_error_per_cutoff = []
        cutoff_weighted_error_per_cutoff = []

        # learned algorithm with cutoff 
        with get_context("spawn").Pool() as pool:
            cutoff_predictions_per_cutoff_frac = pool.starmap(
                run_cutoff_count_sketch, 
                zip(repeat(d), 
                repeat(eval_scores[i]), 
                repeat(eval_scores[i]), 
                space_for_sketch,
                cutoff_fracs,
                repeat(space)))
            pool.close()
            pool.join()

        for predictions in cutoff_predictions_per_cutoff_frac:
            abs_error = np.abs(np.array(predictions) - np.array(d))
            rel_error = abs_error / np.array(d) 
            weighted_error = abs_error * np.array(d) 
        
            cutoff_absolute_error_per_cutoff.append(np.sum(abs_error))
            cutoff_relative_error_per_cutoff.append(np.sum(rel_error))
            cutoff_weighted_error_per_cutoff.append(np.sum(weighted_error))
        
        best_cutoff_absolute_eror_per_oracle_error.append(np.min(cutoff_absolute_error_per_cutoff))
        best_cutoff_relative_eror_per_oracle_error.append(np.min(cutoff_relative_error_per_cutoff))
        best_cutoff_weighted_eror_per_oracle_error.append(np.min(cutoff_weighted_error_per_cutoff))

        i += 1

    # vanilla count sketch
    spinner = Halo(text='Running vanilla count sketch...', spinner='dots')
    spinner.start()
    count_sketch_prediction = run_count_sketch(data, space, space)
    spinner.stop()

    count_sketch_abs_error = np.abs(np.array(count_sketch_prediction) - np.array(data)) 
    count_sketch_rel_error = count_sketch_abs_error / np.array(data) 
    count_sketch_weighted_error = count_sketch_abs_error * np.array(data) 
    spinner.stop()

    # vanilla count min
    spinner = Halo(text='Running vanilla count min...', spinner='dots')
    spinner.start()
    count_min_prediction = run_count_min(data, space, space)
    spinner.stop()

    count_min_abs_error = np.abs(np.array(count_min_prediction) - np.array(data)) 
    count_min_rel_error = count_min_abs_error / np.array(data) 
    count_min_weighted_error = count_min_abs_error * np.array(data) 
    spinner.stop()

    #################################################################
    # save all results to the folder
    #################################################################
    np.savez(os.path.join(save_folder, save_file),
        space=space,
        num_items=len(data),
        count_sketch_abs_error=np.sum(count_sketch_abs_error),
        count_sketch_rel_error=np.sum(count_sketch_rel_error),
        count_sketch_weighted_error=np.sum(count_sketch_weighted_error),
        count_min_abs_error=np.sum(count_min_abs_error),
        count_min_rel_error=np.sum(count_min_rel_error),
        count_min_weighted_error=np.sum(count_min_weighted_error),
        cutoff_abs_error=best_cutoff_absolute_eror_per_oracle_error,
        cutoff_rel_error=best_cutoff_relative_eror_per_oracle_error,
        cutoff_weighted_error=best_cutoff_weighted_eror_per_oracle_error,
        algo_abs_error=algo_mean_absolute_error_per_oracle_error,
        algo_rel_error=algo_mean_relative_error_per_oracle_error,
        algo_weighted_error=algo_mean_weighted_error_per_oracle_error,
        algo_abs_error_std=algo_std_absolute_error_per_oracle_error,
        algo_rel_error_std=algo_std_relative_error_per_oracle_error,
        algo_weighted_error_std=algo_std_weighted_error_per_oracle_error,
        num_trials=n_trials,
    )




#################################################################
# optimal parameter finding for each algorithm
#################################################################

def find_best_parameters_for_cutoff(
    space_list,
    data, 
    oracle_scores, 
    space_allocations, 
    n_workers, 
    save_folder, 
    save_file):

    spinner = Halo(text='Finding optimal parameters for cutoff count sketch', spinner='dots')
    spinner.start()

    # figure out best cutoff threshold for 
    # each error metric 
    best_cutoff_thresh_for_space_weighted = []
    best_cutoff_thresh_for_space_absolute = []
    best_cutoff_thresh_for_space_relative = []

    for i, test_space in enumerate(space_allocations):
        test_space_cs = []
        test_params_cutoff_thresh = []

        # test all combinations 
        for test_cutoff_frac in CUTOFF_FRAC_TO_TEST:
            # combination of parameters to test
            cutoff_thresh = int((test_cutoff_frac * test_space) / CUTOFF_SPACE_COST_FACTOR)
            test_params_cutoff_thresh.append(cutoff_thresh)
            test_space_post_cutoff = int(test_space - cutoff_thresh*CUTOFF_SPACE_COST_FACTOR)
            test_space_cs.append(int(test_space_post_cutoff))

        logger.info("Learning best parameters for space setting...")
        start_t = time.time()

        test_cutoff_predictions = []
        with get_context("spawn").Pool() as pool:
            test_cutoff_predictions = pool.starmap(
                run_cutoff_count_sketch, 
                zip(repeat(data), 
                repeat(oracle_scores), 
                repeat(oracle_scores), 
                test_space_cs, 
                test_params_cutoff_thresh, 
                repeat(test_space))
            )
            pool.close()
            pool.join()

      
        losses_weighted = [np.sum(np.abs(data - predictions)*data) for predictions in test_cutoff_predictions]
        losses_absolute = [np.sum(np.abs(data - predictions)) for predictions in test_cutoff_predictions]
        losses_relative = [np.sum(np.abs(data - predictions) / data) for predictions in test_cutoff_predictions]

        best_loss_idx_weighted = np.argmin(losses_weighted)
        best_loss_idx_absolute = np.argmin(losses_absolute)
        best_loss_idx_relative = np.argmin(losses_relative)

        cutoff_thresh_weighted = test_params_cutoff_thresh[best_loss_idx_weighted]
        cutoff_thresh_absolute = test_params_cutoff_thresh[best_loss_idx_absolute]
        cutoff_thresh_relative = test_params_cutoff_thresh[best_loss_idx_relative]

        logger.info('Found optimal params for %.1f MB (took %.1f sec)' % (4*test_space/1e6, time.time() - start_t))

        best_cutoff_thresh_for_space_weighted.append(cutoff_thresh_weighted)
        best_cutoff_thresh_for_space_absolute.append(cutoff_thresh_absolute)
        best_cutoff_thresh_for_space_relative.append(cutoff_thresh_relative)

    spinner.stop()
    np.savez(os.path.join(save_folder, save_file),
        space_list=space_list,
        best_cutoff_thresh_for_space_weighted=best_cutoff_thresh_for_space_weighted,
        best_cutoff_thresh_for_space_absolute=best_cutoff_thresh_for_space_absolute,
        best_cutoff_thresh_for_space_relative=best_cutoff_thresh_for_space_relative)



if __name__ == '__main__':
    set_start_method("spawn") # bug fix for deadlock in Pool: https://pythonspeed.com/articles/python-multiprocessing/

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--test_dataset", type=str, nargs='*', help="list of input .npy data for testing")
    argparser.add_argument("--valid_dataset", type=str, nargs='*', help="list of input .npy data for validation")
    argparser.add_argument("--find_optimal_params", action='store_true', default=False)
    argparser.add_argument("--optimal_params", type=str, nargs='*', help="optimal parameters to run the validation with")
    argparser.add_argument("--optimal_params_cutoff_cs", type=str, help="optimal parameters to run the validation with for cutoff count sketch")
    argparser.add_argument("--model_test", type=str, nargs='*', help="ml model to use as predictor (.npz file)")
    argparser.add_argument("--model_valid", type=str, nargs='*', help="ml model to use as predictor (.npz file)")
    argparser.add_argument("--count_sketch_results", type=str,  help="results for count sketch .npz file")
    argparser.add_argument("--save_folder", type=str, required=True, help="folder to save the results in")
    argparser.add_argument("--save_file", type=str, required=True, help="prefix to save the results")
    argparser.add_argument("--seed", type=int, default=42, help="random state for sklearn")
    argparser.add_argument("--space_list", type=float, nargs='*', default=[], help="space in MB")
    argparser.add_argument("--n_workers", type=int, default=10, help="number of worker threads",)
    argparser.add_argument("--n_trials", type=int, default=5, help="number of trial runs in the evaluation",)
    argparser.add_argument("--aol_data", action='store_true', default=False)
    argparser.add_argument("--learned_algo_type", type=str, required=True, help="learned algorithm variant THRESHOLD | PARITION")
    argparser.add_argument("--synth_zipfian", action='store_true', default=False)
    argparser.add_argument("--synth_pareto", action='store_true', default=False)
    argparser.add_argument("--synth_space_frac", type=float, default=0.5, help="space fraction to use for synth experiment")
    argparser.add_argument("--run_cutoff_count_sketch", action='store_true', default=False)
    argparser.add_argument("--run_perfect_oracle_version", action='store_true', default=False)
    argparser.add_argument("--model_size", type=float, default=0.0, help="model size in MB")
    args = argparser.parse_args()

    assert (args.learned_algo_type == ALGO_TYPE_CUTOFF_MEDIAN)

    # set the random seed for numpy values
    np.random.seed(args.seed)

    spinner = Halo(text='Loading datasets...', spinner='dots')
    spinner.start()

    # load the test dataset (needed for both validation and testing)
    # specifically, we use the test_oracle_scores to find the score cutoff 
    # value in the validation data (just the way the experiment code is setup)
    test_data, test_oracle_scores = load_dataset(
        args.test_dataset, 
        args.model_test, 
        'valid_output', # old models were trained with valid/test swapped 
        args.run_perfect_oracle_version,
        args.aol_data,
        args.synth_zipfian,
        args.synth_pareto)

    if args.find_optimal_params:
        space_alloc = np.zeros(len(args.space_list))
        for i, space in enumerate(args.space_list):
            space_alloc[i] = int((space - args.model_size) * 1e6 / 4.0) # 4 bytes per bucket

        spinner.stop()

        if args.run_cutoff_count_sketch:
            find_best_parameters_for_cutoff(
                args.space_list, 
                test_data, 
                test_oracle_scores, 
                space_alloc, 
                args.n_workers, 
                args.save_folder,  
                args.save_file + '_count_sketch')

        if args.learned_algo_type == ALGO_TYPE_CUTOFF_MEDIAN:
            print("This variant of the algorithm does not require test data")
            
            # save the space list used so that it can be loaded in the validation stage 
            np.savez(os.path.join(args.save_folder, args.save_file + '_learned'),
            best_cutoff_thresh_for_space=[],
            space_list=args.space_list)

    elif args.valid_dataset is not None:

        # load the test dataset
        valid_data, valid_oracle_scores = load_dataset(
            args.valid_dataset, 
            args.model_valid, 
            'test_output', # models in [Hsu et al.] were trained with valid/test swapped 
            args.run_perfect_oracle_version,
            args.aol_data,
            args.synth_zipfian,
            args.synth_pareto)

        spinner.stop()

        if args.synth_zipfian or args.synth_pareto:
            space_factor = args.synth_space_frac
            synth_space = (len(valid_data) * 8) * space_factor / 4.0  

            # evaluate experiemnt with different errors for the oracle 
            if args.synth_zipfian:
                save_name = '_synth_zipfian'
            elif args.synth_pareto:
                save_name = '_synth_pareto'

            experiment_comapre_loss_vs_oracle_error_on_synthetic_data(
                args.learned_algo_type,
                synth_space,
                valid_data, 
                args.n_trials,
                args.n_workers, 
                args.save_folder, 
                args.save_file + save_name)

        else:
            best_cutoff_thresh_count_sketch_weighted = []
            best_cutoff_thresh_count_sketch_absolute = []
            best_cutoff_thresh_count_sketch_relative = []
            learned_optimal_params = args.optimal_params[0]
            if len(args.optimal_params) > 1:
                count_sketch_optimal_params = args.optimal_params[1]
                data = np.load(count_sketch_optimal_params)
                best_cutoff_thresh_count_sketch_weighted = np.array(data['best_cutoff_thresh_for_space_weighted'])
                best_cutoff_thresh_count_sketch_absolute = np.array(data['best_cutoff_thresh_for_space_absolute'])
                best_cutoff_thresh_count_sketch_relative = np.array(data['best_cutoff_thresh_for_space_relative'])

            data = np.load(learned_optimal_params)
            space_list = np.array(data['space_list'])

            space_alloc = np.zeros(len(space_list))
            for i, space in enumerate(space_list):
                space_alloc[i] = int((space - args.model_size) * 1e6 / 4.0) # 4 bytes per bucket


            # run the experiment with the specified parameters
            experiment_comapre_loss(
                args.learned_algo_type,
                space_list,
                valid_data, 
                valid_oracle_scores,
                test_oracle_scores,
                space_alloc,
                best_cutoff_thresh_count_sketch_weighted,
                best_cutoff_thresh_count_sketch_absolute,
                best_cutoff_thresh_count_sketch_relative,
                args.n_trials,
                args.n_workers, 
                args.save_folder, 
                args.save_file)
    else:
        logger.info("Error: need either testing or validation dataset")
        
