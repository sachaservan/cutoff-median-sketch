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



loss_labels = ["Mean Absolute Error", "Mean Relative Error", "Mean Weighted Error"]
save_file_names = ["experiments/loss_l1_synth_zipfian.pdf", "experiments/loss_relative_synth_zipfian.pdf", "experiments/loss_weighted_synth_zipfian.pdf"]

xticks = [0, 0.25, 0.5, 0.75, 1]
xlabel = "Classification Accuracy"

# 95% confidence interval 
def error_margins(std, ntrials):
    return 2*std / math.sqrt(ntrials)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--results", type=str, default='')
    argparser.add_argument("--x_lim", type=float, nargs='*', default=[])
    argparser.add_argument("--y_lim", type=float, nargs='*', default=[])
    argparser.add_argument("--synth_zipfian", action='store_true', default=False)
    argparser.add_argument("--synth_pareto", action='store_true', default=False)
    args = argparser.parse_args()

    title = "Zipfian Dataset"
    if args.synth_pareto:
        title = "Pareto Dataset"
        save_file_names = ["experiments/loss_l1_synth_pareto.pdf", "experiments/loss_relative_synth_pareto.pdf", "experiments/loss_weighted_synth_pareto.pdf"]


    results = np.load(args.results,  allow_pickle=True)

    num_items = results['num_items']
    normalizer = 1.0 / num_items
    num_trials = 10

    count_sketch_abs_error = results['count_sketch_abs_error'] * normalizer
    count_sketch_rel_error = results['count_sketch_rel_error'] * normalizer
    count_sketch_weighted_error = results['count_sketch_weighted_error'] * normalizer

    count_min_abs_error = results['count_min_abs_error'] * normalizer
    count_min_rel_error = results['count_min_rel_error'] * normalizer
    count_min_weighted_error = results['count_min_weighted_error'] * normalizer
   
    cutoff_abs_error = np.array(results['cutoff_abs_error']) * normalizer
    cutoff_rel_error = np.array(results['cutoff_rel_error']) * normalizer
    cutoff_weighted_error = np.array(results['cutoff_weighted_error']) * normalizer

    algo_abs_error = np.array(results['algo_abs_error']) * normalizer
    algo_rel_error = np.array(results['algo_rel_error']) * normalizer
    algo_weighted_error = np.array(results['algo_weighted_error']) * normalizer

    algo_abs_error_std = np.array(results['algo_abs_error_std']) * normalizer
    algo_rel_error_std = np.array(results['algo_rel_error_std']) * normalizer
    algo_weighted_error_std = np.array(results['algo_weighted_error_std']) * normalizer

    x = np.array(range(0, len(algo_abs_error)))
    x = x / (len(x) - 1)

    # hack to extend the count sketch (a straight line)
    count_sketch_abs_error  = count_sketch_abs_error * np.ones(len(x))
    count_sketch_rel_error  = count_sketch_rel_error * np.ones(len(x))
    count_sketch_weighted_error  = count_sketch_weighted_error * np.ones(len(x))
    count_min_abs_error  = count_min_abs_error * np.ones(len(x))
    count_min_rel_error  = count_min_rel_error * np.ones(len(x))
    count_min_weighted_error  = count_min_weighted_error * np.ones(len(x))

    ####################################################################
    # plot absolute error       
    ####################################################################
    ax = plt.figure().gca()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(figure_width, figure_height)

    ax.plot(x, count_min_abs_error, label="CM", linewidth=line_width, linestyle='dotted', color=colors[3])      
    ax.plot(x, count_sketch_abs_error, label="CS", linewidth=line_width, linestyle='dashdot', color=colors[4])    
    ax.plot(x, cutoff_abs_error, label="CCS", linewidth=line_width, linestyle='dashed', color=colors[1])
    ax.plot(x, algo_abs_error, label="CMS", linewidth=line_width, color=colors[0])

    # 95% confidence interval 
    plt.fill_between(
        x, 
        algo_abs_error - error_margins(algo_abs_error_std, num_trials), 
        algo_abs_error + error_margins(algo_abs_error_std, num_trials), 
        color=colors[0],
        alpha=error_opacity,
    )

    ax.yaxis.grid(color=gridcolor, linestyle=linestyle)
    ax.xaxis.grid(color=gridcolor, linestyle=linestyle)
    ax.set_axisbelow(True)
    ax.set_ylabel(loss_labels[0])
    ax.set_xlabel('Classification Accuracy')
    ax.set_xticks(xticks)
    ax.set_yscale('log', base=10)
    ax.set_ylim(top=max(np.max(count_sketch_abs_error), np.max(algo_abs_error)) * 101)
    ax.set_title(title)
    if args.y_lim:
        ax.set_ylim(args.y_lim)
    if args.x_lim:
        ax.set_xlim(args.x_lim)
    plt.legend(loc='upper right', ncol=2, fontsize=legend_font_size)
    plt.tight_layout()
    plt.savefig(save_file_names[0])


    ####################################################################
    # plot relative error       
    ####################################################################
    ax = plt.figure().gca()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(figure_width, figure_height)

    ax.plot(x, count_min_rel_error, label="CM", linewidth=3, linestyle='dotted', color=colors[3])      
    ax.plot(x, count_sketch_rel_error, label="CS", linewidth=3, linestyle='dashdot', color=colors[4])  
    ax.plot(x, cutoff_rel_error, label="CCS", linewidth=3, linestyle='dashed', color=colors[1])
    ax.plot(x, algo_rel_error, label="CMS", linewidth=3, color=colors[0])
    
    # 95% confidence interval 
    plt.fill_between(
        x, 
        algo_rel_error - error_margins(algo_rel_error_std, num_trials), 
        algo_rel_error + error_margins(algo_rel_error_std, num_trials), 
        color=colors[0],
        alpha=error_opacity,
    )

    ax.yaxis.grid(color=gridcolor, linestyle=linestyle)
    ax.xaxis.grid(color=gridcolor, linestyle=linestyle)
    ax.set_axisbelow(True)
    ax.set_ylabel(loss_labels[1])
    ax.set_xlabel(xlabel)
    ax.set_xticks(xticks)
    ax.set_yscale('log', base=10)
    ax.set_ylim(top=max(np.max(count_sketch_rel_error), np.max(algo_rel_error)) * 101)
    ax.set_title(title)
    if args.y_lim:
        ax.set_ylim(args.y_lim)
    if args.x_lim:
        ax.set_xlim(args.x_lim)
    plt.legend(loc='upper right', ncol=2, fontsize=legend_font_size)
    plt.tight_layout()
    plt.savefig(save_file_names[1])


    ####################################################################
    # plot weighted error       
    ####################################################################
    ax = plt.figure().gca()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(figure_width, figure_height)

    ax.plot(x, count_min_weighted_error, label="CM", linewidth=3, linestyle='dotted', color=colors[3])      
    ax.plot(x, count_sketch_weighted_error, label="CS", linewidth=3,linestyle='dashdot', color=colors[4])    
    ax.plot(x, cutoff_weighted_error, label="CCS", linewidth=3, linestyle='dashed', color=colors[1])
    ax.plot(x, algo_weighted_error, label="CMS", linewidth=3, color=colors[0])
    
    # 95% confidence interval 
    plt.fill_between(
        x, 
        algo_weighted_error - error_margins(algo_weighted_error_std, num_trials), 
        algo_weighted_error + error_margins(algo_weighted_error_std, num_trials), 
        color=colors[0],
        alpha=0.2,
    )

    ax.yaxis.grid(color=gridcolor, linestyle=linestyle)
    ax.xaxis.grid(color=gridcolor, linestyle=linestyle)
    ax.set_axisbelow(True)
    ax.set_ylabel(loss_labels[2])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(xticks)
    ax.set_yscale('log', base=10)
    ax.set_ylim(top=max(np.max(count_sketch_weighted_error), np.max(algo_weighted_error)) * 10)
    if args.y_lim:
        ax.set_ylim(args.y_lim)
    if args.x_lim:
        ax.set_xlim(args.x_lim)
    plt.legend(loc='upper right', ncol=2, fontsize=legend_font_size)
    plt.tight_layout()
    plt.savefig(save_file_names[2])

