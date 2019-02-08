import matplotlib.pyplot as plt
import config
import pandas as pd
from fwdfiles.general_functions import plotTimeSeries
import os
import numpy as np


def plotCrimePredictions(clusters, realCrimes, forecasts, file_path):
    for c in clusters.Cluster.values:
        df = realCrimes['C{}_Crimes'.format(c)]
        testPrediction = forecasts['C{}_Forecast'.format(c)]
        plotTimeSeries(
            df, testPrediction, '{}/{}_cluster({}).png'.format('observations', file_path, c))


if __name__ == '__main__':
    data = pd.read_pickle('DPSUSC.pkl')

   # Uniform grid predictions
    if config.grid_prediction == 1:
        print("Plotting grid predictions...")
        for gridshape in config.ug_gridshapes:
            for periodsAhead in config.periodsAhead_list:
                for method in config.ug_methods:
                    file_path = "{}/{}_predictions_grid({},{})_ignore({})_ahead({})_threshold({})_dist({})".format(
                        method, method, *gridshape, config.ignoreFirst, periodsAhead, 0, 0
                    )
                    data = pd.read_pickle('results/{}.pkl'.format(file_path))
                    [clusters, realCrimes, forecasts] = data
                    # save it to disk
                    os.makedirs(os.path.abspath(
                        "observations/"), exist_ok=True)
                    os.makedirs(os.path.abspath(
                        "observations/{}".format(method)), exist_ok=True)
                    plotCrimePredictions(
                        clusters, realCrimes, forecasts, file_path)

    # Cluster predictions
    if config.cluster_prediction == 1:
        print("Ploting cluster predictions...")
        for threshold in config.c_thresholds:
            print("Cluster plotting with threshold {}".format(threshold))
            for gridshape in config.c_gridshapes:
                for periodsAhead in config.periodsAhead_list:
                    for method in config.c_methods:
                        file_path = "{}/{}_predictions_grid({},{})_ignore({})_ahead({})_threshold({})_dist({})".format(
                            method, method, *gridshape, config.ignoreFirst, periodsAhead, threshold, 1
                        )
                        os.makedirs(os.path.abspath(
                            "observations/"), exist_ok=True)
                        os.makedirs(os.path.abspath(
                            "observations/{}".format(method)), exist_ok=True)
                        data = pd.read_pickle(
                            'results/{}.pkl'.format(file_path))
                        [clusters, realCrimes, forecasts] = data
                        plotCrimePredictions(
                            clusters, realCrimes, forecasts, file_path)
        print("Cluster predictions done!")
