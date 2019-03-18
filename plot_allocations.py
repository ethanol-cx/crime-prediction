import pickle
import numpy as np
import pysal as ps
from scipy import sparse
from datetime import date
from datetime import timedelta
import pandas as pd
from pandas import TimeGrouper
import matplotlib.pyplot as plt
import config
import os


def plot_resource_allocation(ax, gridshapes, periodsAhead, thresholds, methods, ignoreFirst):
    for i in range(len(methods)):
        method = methods[i]
        ax[i].set_ylabel('fractions of crimes avoided')
        for threshold in thresholds:
            for gridshape in gridshapes:
                file = os.path.abspath("results/resource_allocation/{}_{}_({}x{})({})_{}_ahead.pkl".format(
                    'LA' if ignoreFirst == 104 else 'USC', method, gridshape[0], gridshape[1], threshold, periodsAhead))
                with open(file, "rb") as ifile:
                    result = pickle.load(ifile)
                if threshold != 0:
                    ax[i].plot(result, marker='o', alpha=0.85)
                    ax[i].legend()
                    continue
                ax[i].plot(result, alpha=0.85)
                ax[i].legend()


def main():
    os.makedirs(os.path.abspath("results/"), exist_ok=True)
    os.makedirs(os.path.abspath("results/plots"), exist_ok=True)
    for periodsAhead in config.periodsAhead_list:
        plt.close('all')
        fig, ax = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        if config.grid_prediction == 1:
            plot_resource_allocation(
                ax, config.ug_gridshapes, periodsAhead, config.ug_threshold, config.ug_methods, config.ignoreFirst)
        if config.cluster_prediction == 1:
            plot_resource_allocation(
                ax, config.c_gridshapes, periodsAhead, config.c_thresholds, config.c_methods, config.ignoreFirst)
        plt.savefig(
            'results/plots/{}-week-ahead.png'.format(periodsAhead), dpi=300)


if __name__ == "__main__":
    main()
