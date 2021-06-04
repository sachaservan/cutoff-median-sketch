# delete old log file
rm ./experiments/eval.log 

# === INTERNET TRAFFIC ===

# # find optimal parameters for regular learned experiment 
# python3 experiments.py \
#    --learned_algo_type LEARNED_CUTOFF_AND_MEDIAN \
#    --space_list 0.75 1 1.5 2 2.5 3 3.5 4 4.5 5 5.26 \
#    --test_dataset ./data/equinix-chicago.dirA.20160121-130700.ports.npy\
#    --model_test ./paper_model/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1359_res.npz \
#    --save_folder experiments \
#    --save_file ip_optimal_params \
#    --n_workers 100 \
#    --model_size 0.5 \
#    --run_cutoff_count_sketch \
#    --find_optimal_params \

# # run experiment over the validation dataset with optimal params 
#  python3 experiments.py \
#    --learned_algo_type LEARNED_CUTOFF_AND_MEDIAN \
#    --test_dataset ./data/equinix-chicago.dirA.20160121-130700.ports.npy\
#    --model_test ./paper_model/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1359_res.npz \
#    --valid_dataset ./data/equinix-chicago.dirA.20160121-135900.ports.npy\
#    --model_valid ./paper_model/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1359_res.npz \
#    --optimal_params ./experiments/ip_optimal_params_learned.npz  ./experiments/ip_optimal_params_count_sketch.npz\
#    --save_folder experiments \
#    --save_file ip_learned_sketch_experiment_results \
#    --model_size 0.5 \
#    --n_workers 100 \
#    --n_trials 10 \

# # === AOL ===
# python3 experiments.py \
#    --learned_algo_type LEARNED_CUTOFF_AND_MEDIAN \
#    --aol_data \
#    --space_list 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.6 0.676 \
#    --test_dataset ./data/aol_0005_len60.npz \
#    --model_test ./paper_model/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz  \
#    --save_file aol_optimal_params \
#    --save_folder experiments \
#    --model_size 0.015 \
#    --n_workers 100 \
#    --run_cutoff_count_sketch \
#    --find_optimal_params \

#  python3 experiments.py \
#    --learned_algo_type LEARNED_CUTOFF_AND_MEDIAN \
#    --aol_data \
#    --test_dataset ./data/aol_0005_len60.npz \
#    --model_test ./paper_model/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz  \
#    --valid_dataset ./data/aol_0050_len60.npz \
#    --model_valid ./paper_model/aol_inf_all_v05_t50_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz  \
#    --optimal_params ./experiments/aol_optimal_params_learned.npz ./experiments/aol_optimal_params_count_sketch.npz \
#    --save_folder experiments \
#    --save_file aol_learned_sketch_experiment_results \
#    --model_size 0.015 \
#    --n_workers 100 \
#    --n_trials 10 \


# === SYNTHETIC===


# evaluate synthetic data (zipfian) and 25% space 
# python3 experiments.py \
#    --learned_algo_type LEARNED_CUTOFF_AND_MEDIAN \
#    --synth_zipfian \
#    --synth_space_frac 0.25 \
#    --valid_dataset NONE \
#    --save_folder experiments \
#    --save_file sythn_learned_sketch_experiment_results_25_space \
#    --n_workers 100 \
#    --n_trials 10 \


# evaluate synthetic data (pareto) and 25% space 
# python3 experiments.py \
#    --learned_algo_type LEARNED_CUTOFF_AND_MEDIAN \
#    --synth_pareto \
#    --synth_space_frac 0.25 \
#    --valid_dataset NONE \
#    --save_folder experiments \
#    --save_file sythn_learned_sketch_experiment_results_25_space \
#    --n_workers 100 \
#    --n_trials 10 \


# evaluate synthetic data (zipfian) and 50% space 
# python3 experiments.py \
#    --learned_algo_type LEARNED_CUTOFF_AND_MEDIAN \
#    --synth_zipfian \
#    --synth_space_frac 0.50 \
#    --valid_dataset NONE \
#    --save_folder experiments \
#    --save_file sythn_learned_sketch_experiment_results_50_space \
#    --n_workers 100 \
#    --n_trials 10 \


# evaluate synthetic data (pareto) and 50% space 
# python3 experiments.py \
#    --learned_algo_type LEARNED_CUTOFF_AND_MEDIAN \
#    --synth_pareto \
#    --synth_space_frac 0.50 \
#    --valid_dataset NONE \
#    --save_folder experiments \
#    --save_file sythn_learned_sketch_experiment_results_50_space \
#    --n_workers 100 \
#    --n_trials 10 \


