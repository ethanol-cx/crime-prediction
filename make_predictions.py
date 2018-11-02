import pickle
import pandas as pd
import sys
import os
import fwdfiles.forecast_harmonic as forecast_harmonic
import fwdfiles.forecast_mean as forecast_mean
import fwdfiles.forecast_ar as forecast_ar
from fwdfiles.general_functions import *
from fwdfiles.cluster_functions import *
from fwdfiles.ScriptForecast import computeClustersAndOrganizeData
import config


def savePredictions(clusters, realCrimes, forecasts, algo,
                    gridshape=(60, 85), ignoreFirst=149,
                    periodsAhead=52, threshold=4000, maxDist=5):
    # save it to disk
    os.makedirs(os.path.abspath("results/"), exist_ok=True)
    os.makedirs(os.path.abspath("results/{}".format(algo)), exist_ok=True)
    fileName = os.path.abspath(
        "results/{}/{}_predictions_grid({},{})_ignore({})_ahead({})_threshold({})_dist({}).pkl".format(
            algo, algo, *gridshape, ignoreFirst, periodsAhead, threshold, maxDist
        )
    )
    output = open(fileName, 'wb')
    pickle.dump((clusters, realCrimes, forecasts), output)
    output.close()
    return


def saveParameters(orders, seasonal_orders, algo,
                   gridshape=(60, 85), ignoreFirst=149,
                   periodsAhead=52, threshold=4000, maxDist=5):
    # save it to disk
    os.makedirs(os.path.abspath("parameters/"), exist_ok=True)
    os.makedirs(os.path.abspath("parameters/{}".format(algo)), exist_ok=True)
    fileName = os.path.abspath(
        "parameters/{}/{}_parameters_grid({},{})_ignore({})_ahead({})_threshold({})_dist({}).pkl".format(
            algo, algo, *gridshape, ignoreFirst, periodsAhead, threshold, maxDist
        )
    )
    output = open(fileName, 'wb')
    pickle.dump((orders, seasonal_orders), output)
    output.close()
    return


def predict(data, gridshape, ignoreFirst, periodsAhead, threshold, maxDist, methods, clusters, realCrimes,
            mm_orders, ar_orders, h_orders, mm_seasonal_orders, ar_seasonal_orders, h_seasonal_orders):
    if "mm" in methods:
        forecast_mean.init_predictions_Mean()
        fileName = os.path.abspath(
            "parameters/{}/{}_parameters_grid({},{})_ignore({})_ahead({})_threshold({})_dist({}).pkl".format(
                'mm', 'mm', *gridshape, ignoreFirst, 1, threshold, maxDist
            )
        )
        # orders = pd.read_pickle(fileName)
        # mm_orders = orders[0]
        # mm_seasonal_orders = orders[1]

        mm_forecasts, mm_orders, mm_seasonal_orders = forecast_mean.predictions_Mean(
            clusters, realCrimes, periodsAhead, None, None)
        savePredictions(clusters, realCrimes, mm_forecasts, "mm",
                        gridshape, ignoreFirst, periodsAhead, threshold, maxDist)
        # saveParameters(mm_orders, mm_seasonal_orders, "mm",
        #                gridshape, ignoreFirst, periodsAhead, threshold, maxDist)

    if "ar" in methods:
        forecast_ar.init_predictions_AR()
        fileName = os.path.abspath(
            "parameters/{}/{}_parameters_grid({},{})_ignore({})_ahead({})_threshold({})_dist({}).pkl".format(
                'ar', 'ar', *gridshape, ignoreFirst, 1, threshold, maxDist
            )
        )
        # orders = pd.read_pickle(fileName)
        # ar_orders = orders[0]
        # ar_seasonal_orders = orders[1]

        ar_forecasts, ar_orders, ar_seasonal_orders = forecast_ar.predictions_AR(
            clusters, realCrimes, periodsAhead, None, None)
        savePredictions(clusters, realCrimes, ar_forecasts, "ar",
                        gridshape, ignoreFirst, periodsAhead, threshold, maxDist)
        # saveParameters(ar_orders, ar_seasonal_orders, "ar",
        #                gridshape, ignoreFirst, periodsAhead, threshold, maxDist)
    if "harmonic" in methods:
        forecast_harmonic.init_predictions_Harmonic()
        fileName = os.path.abspath(
            "parameters/{}/{}_parameters_grid({},{})_ignore({})_ahead({})_threshold({})_dist({}).pkl".format(
                'harmonic', 'harmonic', *gridshape, ignoreFirst, 1, threshold, maxDist
            )
        )
        # orders = pd.read_pickle(fileName)
        # h_orders = orders[0]
        # h_seasonal_orders = orders[1]

        h_forecasts, h_orders, h_seasonal_orders = forecast_harmonic.predictions_Harmonic(
            clusters, realCrimes, periodsAhead, None, None)
        savePredictions(clusters, realCrimes, h_forecasts, "harmonic",
                        gridshape, ignoreFirst, periodsAhead, threshold, maxDist)
        # saveParameters(h_orders, h_seasonal_orders, "harmonic",
        #                gridshape, ignoreFirst, periodsAhead, threshold, maxDist)
    return mm_orders, mm_seasonal_orders, ar_orders, ar_seasonal_orders, h_orders, h_seasonal_orders


def compute_grid_predictions(data, gridshapes, ignoreFirst, periodsAhead_list,
                             threshold, maxDist, methods):
    for gridshape in gridshapes:
        mm_orders = [(0, 0, 0)]
        mm_seasonal_orders = [(0, 0, 0, 0)]
        ar_orders = [(0, 0, 0)]
        ar_seasonal_orders = [(0, 0, 0, 0)]
        h_orders = [(0, 0, 0)]
        h_seasonal_orders = [(0, 0, 0, 0)]
        print('Computing clusters ...')
        clusters, realCrimes = computeClustersAndOrganizeData(
            data, gridshape, ignoreFirst, 0, threshold, maxDist)
        print('Number of clusters: {}'.format(len(clusters)))
        print('Computing predictions ...')
        for periodsAhead in periodsAhead_list:
            mm_orders, mm_seasonal_orders, ar_orders, ar_seasonal_orders, h_orders, h_seasonal_orders = predict(
                data, gridshape, ignoreFirst, periodsAhead, threshold, maxDist, methods, clusters, realCrimes,
                mm_orders, ar_orders, h_orders, mm_seasonal_orders, ar_seasonal_orders, h_seasonal_orders)


def compute_cluster_predictions(data, gridshape, ignoreFirst,
                                periodsAhead_list,
                                thresholds, maxDist, methods):
    for threshold in thresholds:
        mm_orders = [(0, 0, 0)]
        mm_seasonal_orders = [(0, 0, 0, 0)]
        ar_orders = [(0, 0, 0)]
        ar_seasonal_orders = [(0, 0, 0, 0)]
        h_orders = [(0, 0, 0)]
        h_seasonal_orders = [(0, 0, 0, 0)]
        print('Computing clusters ...')
        clusters, realCrimes = computeClustersAndOrganizeData(
            data, gridshape, ignoreFirst, 0, threshold, maxDist)
        print(clusters)
        print('Number of clusters: {}'.format(len(clusters)))
        print('Computing predictions ...')
        for periodsAhead in periodsAhead_list:
            mm_orders, mm_seasonal_orders, ar_orders, ar_seasonal_orders, h_orders, h_seasonal_orders = predict(
                data, gridshape, ignoreFirst, periodsAhead, threshold, maxDist, methods, clusters, realCrimes,
                mm_orders, ar_orders, h_orders, mm_seasonal_orders, ar_seasonal_orders, h_seasonal_orders)


def main(ifilename):
    data = pd.read_pickle(ifilename)
    # base = importr('base')
    # print(base._libPaths())

    # Uniform grid predictions
    # print("Making grid predictions...")
    # compute_grid_predictions(data, config.ug_gridshapes, config.ignoreFirst,
    #                          config.periodsAhead_list, config.ug_threshold,
    #                          config.ug_maxDist, config.ug_methods)
    # print("Grid predictions done!")

    # Cluster predictions
    print("Making cluster predictions...")
    compute_cluster_predictions(data, config.c_gridshape, config.ignoreFirst,
                                config.periodsAhead_list, config.c_thresholds,
                                config.c_maxDist, config.c_methods)
    print("Cluster predictions done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception(
            "Usage: python make_predictions.py [input file name].pkl")

    main(sys.argv[1])
