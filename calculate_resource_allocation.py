import pickle
import sys
import os
# sys.path.append(os.path.abspath("fwdfiles/"))
from fwdfiles.resourceAllocation_functions import fixResourceAvailable
import config
from fwdfiles.general_functions import getAreaFromLatLon


def compute_ra_grid(resource_indexes, cell_coverage_units, gridshapes,
                    periodsAhead_list, ignoreFirst, threshold, dist, methods):
    for periodsAhead in periodsAhead_list:
        os.makedirs(os.path.abspath("results/"), exist_ok=True)
        os.makedirs(os.path.abspath("results/grid"), exist_ok=True)
        output_filename = os.path.abspath(
            "results/grid/grid_{}ahead.pkl".format(periodsAhead))

        MA_results = []
        AR_results = []
        ARIMA_results = []

        for gridshape in gridshapes:
            for method in methods:
                file = os.path.abspath("results/{}/{}_predictions_grid({},{})_ignore({})_ahead({})_threshold({})_dist({}).pkl".format(
                    method, method, gridshape[0], gridshape[1], ignoreFirst, periodsAhead, threshold, dist))
                with open(file, "rb") as ifile:
                    clusters, realCrimes, forecasts = pickle.load(ifile)
                unit_area = getAreaFromLatLon(
                    config.lon_min, config.lon_max, config.lat_min, config.lat_max) / (gridshape[0] * gridshape[1])
                scores = fixResourceAvailable(resource_indexes, 0, forecasts, realCrimes, clusters, cell_coverage_units, unit_area).rename(
                    "{} ({}x{})".format(method, gridshape[0], gridshape[1]))
                if method == 'MA':
                    MA_results.append(scores)
                elif method == 'AR':
                    AR_results.append(scores)
                elif method == 'ARIMA':
                    ARIMA_results.append(scores)
                else:
                    print("ERROR: wrong method")

        with open(output_filename, "wb") as ofile:
            pickle.dump((MA_results, AR_results, ARIMA_results), ofile)


def compute_ra_clustering(resource_indexes, cell_coverage_units, gridshape, periodsAhead_list, ignoreFirst, thresholds, dist, methods):
    for periodsAhead in periodsAhead_list:
        os.makedirs(os.path.abspath("results/"), exist_ok=True)
        os.makedirs(os.path.abspath("results/cluster"), exist_ok=True)
        output_filename = os.path.abspath(
            "results/cluster/cluster_{}ahead.pkl".format(periodsAhead))
        output_filename = os.path.abspath(
            "results/cluster/cluster_{}ahead.pkl".format(periodsAhead))

        MA_results = []
        AR_results = []
        ARIMA_results = []

        for threshold in thresholds:
            for method in methods:
                file = os.path.abspath("results/{}/{}_predictions_grid({},{})_ignore({})_ahead({})_threshold({})_dist({}).pkl".format(
                    method, method, gridshape[0], gridshape[1], ignoreFirst, periodsAhead, threshold, dist))
                with open(file, "rb") as ifile:
                    clusters, realCrimes, forecasts = pickle.load(ifile)
                unit_area = getAreaFromLatLon(
                    config.lon_min, config.lon_max, config.lat_min, config.lat_max) / (gridshape[0] * gridshape[1])
                scores = fixResourceAvailable(resource_indexes, 0, forecasts, realCrimes, clusters, cell_coverage_units,
                                              unit_area).rename("{} ({})".format(method, threshold))
                if method == 'MA':
                    MA_results.append(scores)
                elif method == 'AR':
                    AR_results.append(scores)
                elif method == 'ARIMA':
                    ARIMA_results.append(scores)
                else:
                    print("ERROR: wrong method")

        with open(output_filename, "wb") as ofile:
            pickle.dump((MA_results, AR_results, ARIMA_results), ofile)


def main():
    # Grid
    if config.grid_prediction == 1:
        print("Scoring grid predictions...")
        compute_ra_grid(config.resource_indexes, config.cell_coverage_units,
                        config.ug_gridshapes, config.periodsAhead_list,
                        config.ignoreFirst, config.ug_threshold[0],
                        config.ug_maxDist, config.ug_methods)

    # Clusters
    if config.cluster_prediction == 1:
        print("Scoring cluster predictions...")
        compute_ra_clustering(config.resource_indexes, config.cell_coverage_units,
                              config.c_gridshapes[0], config.periodsAhead_list,
                              config.ignoreFirst, config.c_thresholds,
                              config.c_maxDist, config.c_methods)


if __name__ == "__main__":
    main()
