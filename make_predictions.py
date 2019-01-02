import pickle
import pandas as pd
import sys
import os
from fwdfiles.forecast_ARIMA import forecast_ARIMA
from fwdfiles.cluster_functions import computeClustersAndOrganizeData
import config


def compute_predictions(data, gridshapes, ignoreFirst, periodsAhead_list,
                        threshold, maxDist, methods):
    for gridshape in gridshapes:

        # Compute the cluster/grid distribution based on the threshold.
        print('Computing clusters ...')

        # In grid_prediction, which predict the crimes without clustering, the threshold is set to 0
        # `clusters` is the cluster distributions
        clusters, realCrimes = computeClustersAndOrganizeData(
            data, gridshape, ignoreFirst, threshold, maxDist)

        print('Number of clusters: {}'.format(len(clusters)))
        print('Computing predictions ...')

        for method in methods:
            forecast_ARIMA(method=method, clusters=clusters, realCrimes=realCrimes,
                           periodsAhead_list=periodsAhead_list, gridshape=gridshape, ignoreFirst=ignoreFirst, threshold=threshold, maxDist=maxDist, orders=[], seasonal_orders=[])


def main(ifilename):
    data = pd.read_pickle(ifilename)

    # Uniform grid predictions
    if config.grid_prediction == 1:
        print("Making grid predictions...")
        for threshold in config.c_thresholds:
            print("Grid prediction with threshold {}".format(threshold))
            compute_predictions(data=data, gridshapes=config.ug_gridshapes, ignoreFirst=config.ignoreFirst,
                                periodsAhead_list=config.periodsAhead_list, threshold=threshold,
                                maxDist=config.ug_maxDist, methods=config.ug_methods)
        print("Grid predictions done!")

    # Cluster predictions
    if config.cluster_prediction == 1:
        print("Making cluster predictions...")
        for threshold in config.c_thresholds:
            print("Cluster prediction with threshold {}".format(threshold))
            compute_predictions(data=data, gridshapes=config.c_gridshapes, ignoreFirst=config.ignoreFirst,
                                periodsAhead_list=config.periodsAhead_list, threshold=threshold,
                                maxDist=config.c_maxDist, methods=config.c_methods)
        print("Cluster predictions done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception(
            "Usage: python make_predictions.py [input file name].pkl")
    main(sys.argv[1])
