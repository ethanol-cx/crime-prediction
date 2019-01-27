import pickle
import sys
import os
from fwdfiles.resourceAllocation_functions import fixResourceAvailable
import config
from fwdfiles.general_functions import getAreaFromLatLon
import numpy as np


def compute_ra_grid(resource_indexes, cell_coverage_units, gridshapes,
                    periodsAhead_list, ignoreFirst, threshold, dist, methods):
    for periodsAhead in periodsAhead_list:
        os.makedirs(os.path.abspath("results/"), exist_ok=True)
        os.makedirs(os.path.abspath("results/grid"), exist_ok=True)
        output_filename = os.path.abspath(
            "results/grid/grid_{}ahead.pkl".format(periodsAhead))

        results = np.zeros(len(methods), len(gridshapes))

        for i in range(len(gridshapes)):
            gridshape = gridshapes[i]
            for j in range(len(methods)):
                method = methods[j]
                file = os.path.abspath("results/{}/{}_predictions_grid({},{})_ignore({})_ahead({})_threshold({})_dist({}).pkl".format(
                    method, method, gridshape[0], gridshape[1], ignoreFirst, periodsAhead, threshold, dist))
                with open(file, "rb") as ifile:
                    clusters, realCrimes, forecasts = pickle.load(ifile)
                unit_area = getAreaFromLatLon(
                    config.lon_min, config.lon_max, config.lat_min, config.lat_max) / (gridshape[0] * gridshape[1])
                scores = fixResourceAvailable(resource_indexes, 0, forecasts, realCrimes, clusters, cell_coverage_units, unit_area).rename(
                    "{} ({}x{})".format(method, gridshape[0], gridshape[1]))
                results[j][i] = scores

        with open(output_filename, "wb") as ofile:
            pickle.dump(results, ofile)


def compute_ra_clustering(resource_indexes, cell_coverage_units, gridshapes, periodsAhead_list, ignoreFirst, thresholds, dist, methods):
    for periodsAhead in periodsAhead_list:
        os.makedirs(os.path.abspath("results/"), exist_ok=True)
        os.makedirs(os.path.abspath("results/cluster"), exist_ok=True)
        output_filename = os.path.abspath(
            "results/cluster/cluster_{}ahead.pkl".format(periodsAhead))
        output_filename = os.path.abspath(
            "results/cluster/cluster_{}ahead.pkl".format(periodsAhead))
        i = -1
        j = -1
        results = np.zeros(len(methods), len(gridshapes) * len(thresholds))
        for gridshape in gridshapes:
            for threshold in thresholds:
                i += 1
                for method in methods:
                    j += 1
                    file = os.path.abspath("results/{}/{}_predictions_grid({},{})_ignore({})_ahead({})_threshold({})_dist({}).pkl".format(
                        method, method, gridshape[0], gridshape[1], ignoreFirst, periodsAhead, threshold, dist))
                    with open(file, "rb") as ifile:
                        clusters, realCrimes, forecasts = pickle.load(ifile)
                    unit_area = getAreaFromLatLon(
                        config.lon_min, config.lon_max, config.lat_min, config.lat_max) / (gridshape[0] * gridshape[1])
                    scores = fixResourceAvailable(resource_indexes, 0, forecasts, realCrimes, clusters, cell_coverage_units,
                                                  unit_area).rename("{} grid:({}, {}) threshold:({})".format(method, gridshape[0], gridshape[1], threshold))
                    results[j][i] = scores
        with open(output_filename, "wb") as ofile:
            pickle.dump(results, ofile)


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
                              config.c_gridshapes, config.periodsAhead_list,
                              config.ignoreFirst, config.c_thresholds,
                              config.c_maxDist, config.c_methods)


if __name__ == "__main__":
    main()
