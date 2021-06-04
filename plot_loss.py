import sys
import time
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math 


###########################
# PLOT PARAMETERS
###########################
font = {'family': 'sans-serif',
        'sans-serif': ['Tahoma', 'DejaVu Sans', ' Lucida Grande', 'Verdana'],
        'size': 25}
matplotlib.rc('font', **font)
colors = ['#0099CC',  '#955196', '#ffa600', '#32cd32', '#ff6e54',  '#003f5c', '#333333']

figure_width = 6.5
figure_height = 5

# more colors:
# https://www.color-hex.com/color-palettes/
edgecolor = '#34495e'
gridcolor = '#2f3640'
linestyle = 'dotted'
opacity = 1
error_opacity=0.2
line_width=3
legend_font_size=23
###########################


###########################
# loss functions to compute 
###########################
def loss_weighted(y_true, y_est):
    return np.sum(np.abs(y_true - y_est) * y_true)

def loss_relative(y_true, y_est):
    return np.sum(np.abs(y_true - y_est) / y_true)

def loss_l1(y_true, y_est):
    return np.sum(np.abs(y_true - y_est))

def loss_l2(y_true, y_est):
    return np.sum(np.abs(y_true - y_est) ** 2)

# order of functions and labels must match
loss_functions = [loss_weighted, loss_l1, loss_relative]
loss_labels = ["Mean Weighted Error", "Mean Absolute Error", "Mean Relative Error"]
###########################

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--results", type=str, default='')
    argparser.add_argument("--x_lim", type=float, nargs='*', default=[])
    argparser.add_argument("--y_lim", type=float, nargs='*', default=[])
    argparser.add_argument("--title", type=str, default='')
    argparser.add_argument("--algo", type=str, default='Alg')
    argparser.add_argument("--ip", action='store_true', default=False)
    argparser.add_argument("--aol", action='store_true', default=False)
    argparser.add_argument("--synth_zipfian", action='store_true', default=False)
    argparser.add_argument("--synth_pareto", action='store_true', default=False)
    argparser.add_argument("--max_space", type=float, default=4.0)
    args = argparser.parse_args()
    
    save_file_names = ""
    if args.ip:
        save_file_names = ["experiments/loss_weighted_ip.pdf", "experiments/loss_l1_ip.pdf", "experiments/loss_relative_ip.pdf"]
    elif args.aol:
        save_file_names = ["experiments/loss_weighted_aol.pdf","experiments/loss_l1_aol.pdf",  "experiments/loss_relative_aol.pdf"]
    elif args.synth_zipfian:
        save_file_names = ["experiments/loss_weighted_synth_zipfian.pdf", "experiments/loss_l1_synth_zipfian.pdf", "experiments/loss_relative_synth_zipfian.pdf"]
    elif args.synth_pareto:
        save_file_names = ["experiments/loss_weighted_synth_pareto.pdf", "experiments/loss_l1_synth_pareto.pdf", "experiments/loss_relative_synth_pareto.pdf"]



    results = np.load(args.results,  allow_pickle=True)
    space = np.array(results['space_list'])
    true_counts = np.array(results['true_values'])
    pred_counts = np.array(results['oracle_predictions'])
    algo_predictions = np.array(results['algo_predictions'])  
  
    count_sketch_predictions = np.array(results['count_sketch_predictions'])
    count_min_predictions = np.array(results['count_min_predictions'])
    cutoff_count_sketch_predictions_weighted = np.array(results['cutoff_count_sketch_predictions_weighted'])
    cutoff_count_sketch_predictions_absolute = np.array(results['cutoff_count_sketch_predictions_absolute'])
    cutoff_count_sketch_predictions_relative = np.array(results['cutoff_count_sketch_predictions_relative'])

    cutoff_count_sketch_predictions = [
        cutoff_count_sketch_predictions_weighted,
        cutoff_count_sketch_predictions_absolute,
        cutoff_count_sketch_predictions_relative,
    ]

    #################################################
    # 1. plot the data distribution 
    #################################################
    sort = np.argsort(true_counts)[::-1]
    true_counts_grouped = np.array([np.mean(x) for x in np.array_split(true_counts[sort], 1000)])
    true_counts_sorted = np.sort(true_counts)[::-1]

    ax = plt.figure().gca()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(30, 30)
    ax.set_axisbelow(True)
    ax.xaxis.grid(color=gridcolor, linestyle=linestyle, linewidth=3, zorder=-1)
    ax.yaxis.grid(color=gridcolor, linestyle=linestyle, linewidth=3, zorder=-1)
    ax.plot(np.linspace(0, 1, len(true_counts_grouped)), true_counts_grouped, linewidth=15, color=colors[0])
    ax.set_yscale('log', base=10)
    ax.set_xscale('log', base=10)

    textsize=180
    titlesize=210

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    ax.set_ylabel('Item Frequency', fontsize=textsize)
    ax.set_xlabel('Index', fontsize=textsize)

    save_file_name = "experiments/ip_dataset.pdf"
    if args.aol:
        save_file_name = "experiments/aol_dataset.pdf"
    elif args.synth_zipfian:
        save_file_name = "experiments/synth_zipfian_dataset.pdf"
    elif args.synth_pareto:
        save_file_name = "experiments/synth_pareto_dataset.pdf"

    if args.aol:
            ax.set_title('AOL', fontsize=titlesize)
    elif args.synth_zipfian:
        ax.set_title('Zipfian', fontsize=titlesize)
    elif args.synth_pareto:
        ax.set_title('Pareto', fontsize=titlesize)
    else:
        ax.set_title('CAIDA', fontsize=titlesize)

    if args.y_lim:
        ax.set_ylim(args.y_lim)
    if args.x_lim:
        ax.set_xlim(args.x_lim)
        
    plt.tight_layout()
    plt.savefig(save_file_name)
    #################################################


    #################################################
    # 2. plot the results for each error metric
    #################################################
    normalizer = 1.0 / len(true_counts) 

    # assumes 4*2 bytes needed to store ID and count of each item
    space_percent = [math.ceil(x) for x in (space*1e8) / ((len(true_counts)-1) * 8)]  
         
    for i in range(len(loss_functions)):
        loss_sketch_mean = [] # count sketch 
        loss_min_mean = [] # count min 
        loss_algo_mean =  []
        loss_cutoff_mean =  []
        loss_min_std = []
        loss_sketch_std = []
        loss_algo_std =  []
        loss_cutoff_std =  []

        for j in range(len(space)):
            losses_for_space_algo = []
            losses_for_space_min = []
            losses_for_space_sketch = []
            losses_for_space_cutoff = []
            
            for trial in range(len(algo_predictions)):
                loss_for_space_algo = loss_functions[i](true_counts, algo_predictions[trial][j]) 
                loss_for_space_sketch = loss_functions[i](true_counts, count_sketch_predictions[trial][j]) 
                loss_for_space_min = loss_functions[i](true_counts, count_min_predictions[trial][j]) 
                loss_for_space_cutoff = loss_functions[i](true_counts, cutoff_count_sketch_predictions[i][trial][j])  
    
                losses_for_space_algo.append(loss_for_space_algo * normalizer)
                losses_for_space_sketch.append(loss_for_space_sketch * normalizer)
                losses_for_space_min.append(loss_for_space_min * normalizer)
                losses_for_space_cutoff.append(loss_for_space_cutoff * normalizer)

            loss_algo_mean.append(np.mean(losses_for_space_algo))
            loss_sketch_mean.append(np.mean(losses_for_space_sketch))
            loss_cutoff_mean.append(np.mean(losses_for_space_cutoff))
            loss_min_mean.append(np.mean(losses_for_space_min))
            loss_algo_std.append(np.std(losses_for_space_algo))
            loss_sketch_std.append(np.std(losses_for_space_sketch))
            loss_min_std.append(np.std(losses_for_space_min))
            loss_cutoff_std.append(np.std(losses_for_space_cutoff))

        loss_sketch_mean = np.array(loss_sketch_mean)
        loss_min_mean = np.array(loss_min_mean)
        loss_algo_mean =  np.array(loss_algo_mean)
        loss_cutoff_mean = np.array(loss_cutoff_mean)
        loss_sketch_std = np.array(loss_sketch_std)
        loss_min_std = np.array(loss_min_std)
        loss_algo_std = np.array(loss_algo_std)
        loss_cutoff_std = np.array(loss_cutoff_std)

        loss_improv_count_sketch = loss_sketch_mean / loss_algo_mean
        loss_improv_cutoff = loss_cutoff_mean / loss_algo_mean

        # print("Performance breakdown:")
        # print("Count-Sketch vs. Cutoff Median (" + loss_labels[i] + ")         min = %.2f" % np.min(loss_improv_count_sketch) + " max = %.2f" % np.max(loss_improv_count_sketch))
        # print("Cutoff Count-Sketch vs. Cutoff Median (" + loss_labels[i] + ")  min = %.2f" % np.min(loss_improv_cutoff) + " max = %.2f" % np.max(loss_improv_cutoff))

        ax = plt.figure().gca()
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(figure_width, figure_height)

        ax.plot(space_percent, loss_min_mean, label="CM", linewidth=line_width, linestyle='dotted', color=colors[3])        
        ax.plot(space_percent, loss_sketch_mean, label="CS", linewidth=line_width, linestyle='dashdot', color=colors[4])        
        ax.plot(space_percent, loss_cutoff_mean, label="CCS", linewidth=line_width, linestyle='dashed', color=colors[1])
        ax.plot(space_percent, loss_algo_mean, label="CMS", linewidth=line_width, color=colors[0])
    
        # 95% confidence interval 
        plt.fill_between(
            space_percent, 
            loss_min_mean - loss_min_std, 
            loss_min_mean + loss_min_std, 
            color=colors[3],
            alpha=error_opacity,
        )
        plt.fill_between(
            space_percent, 
            loss_sketch_mean - loss_sketch_std, 
            loss_sketch_mean + loss_sketch_std, 
            color=colors[4],
            alpha=error_opacity,
        )
        plt.fill_between(
            space_percent, 
            loss_cutoff_mean - loss_cutoff_std, 
            loss_cutoff_mean + loss_cutoff_std, 
            color=colors[1],
            alpha=error_opacity,
        )
        plt.fill_between(
            space_percent, 
            loss_algo_mean - loss_algo_std, 
            loss_algo_mean + loss_algo_std, 
            color=colors[0],
            alpha=error_opacity,
        )


        # ax.yaxis.grid(color=gridcolor, linestyle=linestyle)
        ax.xaxis.grid(color=gridcolor, linestyle=linestyle, zorder=-1)
        ax.set_axisbelow(True)
        ax.set_yscale('log', base=10)
        ax.set_ylabel(loss_labels[i])
        ax.set_xlabel('Percent Space')
        ax.set_ylim(top=max(np.max(loss_min_mean), np.max(loss_algo_mean)) * 10)

        if args.aol:
            ax.set_title('AOL Dataset')
        elif args.synth_zipfian:
            ax.set_title('Zipfian Dataset')
        elif args.synth_pareto:
            ax.set_title('Pareto Dataset')
        else:
            ax.set_title('CAIDA Dataset')

        if args.y_lim:
            ax.set_ylim(args.y_lim)
        if args.x_lim:
            ax.set_xlim(args.x_lim)
            
        plt.xticks(np.arange(10, 61, 10.0))
        plt.legend(loc='upper right', ncol=2, fontsize=legend_font_size)
        plt.tight_layout()
        plt.savefig(save_file_names[i])

