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

def printGrids(periodsAhead):
    %matplotlib inline
    plt.close('all')
    fig, ax = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for result in results_mm_grid + results_ar_grid:
        ax[0].set_ylabel('% crime avoided')
        ax[0].plot(result)
        ax[0].legend()
    for result in results_mm_grid + results_h_grid:
        ax[1].set_xlabel('Available resources')
        ax[1].set_title('{} week(s) ahead grid results'.format(periodsAhead))
        ax[1].plot(result)
        ax[1].legend()
    for result in results_ar_grid + results_h_grid:
        ax[2].plot(result)
        ax[2].legend()

    plt.savefig('results/grid/{}week_grids.svg'.format(periodsAhead))
    plt.savefig('results/grid/{}week_grids.png'.format(periodsAhead), dpi=300)
    plt.show()


def printClusterGrids(periodsAhead):
    %matplotlib inline
    plt.close('all')
    fig, ax = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for result in results_mm_grid + results_ar_grid:
        ax[0].set_ylabel('% crime avoided')
        ax[0].plot(result)
        ax[0].legend()
    for result in results_mm_grid + results_h_grid:
        ax[1].set_xlabel('Available resources')
        ax[1].set_title(
            '{} week(s) ahead cluster results'.format(periodsAhead))
        ax[1].plot(result)
        ax[1].legend()
    for result in results_ar_grid + results_h_grid:
        ax[2].plot(result)
        ax[2].legend()

    plt.savefig('results/cluster/{}week_clusters.svg'.format(periodsAhead))
    plt.savefig(
        'results/cluster/{}week_clusters.png'.format(periodsAhead), dpi=300)
    plt.show()

def main:
    for periodsAhead in 