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

def plot_resource_allocation(resource_indexes, cell_coverage_units, gridshapes, periodsAhead_list, thresholds, methods):
    os.makedirs(os.path.abspath("results/"), exist_ok=True)
    os.makedirs(os.path.abspath("results/plots"), exist_ok=True)
    for periodsAhead in periodsAhead_list:
        plt.close('all')
        fig, ax = plt.subplots(1, 3, figsize=(18,5), sharey=True)
        for i in range(len(methods)):
            method = methods[i]
            ax[i].set_ylabel('fractions of crimes avoided')
            for threshold in thresholds:
                for gridshape in gridshapes:
                    file = os.path.abspath("results/resource_allocation/{}_({}x{})({})_{}_ahead.pkl".format(method, gridshape[0], gridshape[1], threshold, periodsAhead))
                    with open(file, "rb") as ifile:
                        result = pickle.load(ifile)
                    ax[i].plot(result, alpha = 0.85)
                    ax[i].legend()
        plt.savefig('results/plots/{}-week-ahead.png'.format(periodsAhead), dpi=300)

def main():
    # Clusters
    if config.cluster_prediction == 1:
        thresholds = config.c_thresholds
        thresholds.append(0)
        print("Scoring cluster predictions...")
        plot_resource_allocation(config.resource_indexes, config.cell_coverage_units, config.c_gridshapes, config.periodsAhead_list, thresholds, config.c_methods)

if __name__ == "__main__":
    main()