import pickle
import sys
import os
from fwdfiles.resourceAllocation_functions import fixResourceAvailable
import config
from fwdfiles.general_functions import getAreaFromLatLon
import numpy as np


def compute_resource_allocation(resource_indexes, cell_coverage_units, gridshapes, periodsAhead_list, ignoreFirst, thresholds, dist, methods):
    for periodsAhead in periodsAhead_list:
        os.makedirs(os.path.abspath("results/"), exist_ok=True)
        os.makedirs(os.path.abspath(
            "results/resource_allocation"), exist_ok=True)
        for method in methods:
            for threshold in thresholds:
                for gridshape in gridshapes:
                    output_filename = os.path.abspath("results/resource_allocation/{}_{}_({}x{})({})_{}_ahead.pkl".format(
                        'LA' if ignoreFirst == 104 else 'USC', method, gridshape[0], gridshape[1], threshold, periodsAhead))
                    file = os.path.abspath("results/{}/{}_predictions_grid({},{})_ignore({})_ahead({})_threshold({})_dist({}).pkl".format(
                        method, method, gridshape[0], gridshape[1], ignoreFirst, periodsAhead, threshold, dist))
                    with open(file, "rb") as ifile:
                        clusters, realCrimes, forecasts = pickle.load(ifile)
                    unit_area = getAreaFromLatLon(
                        lon1=config.lon_min, lon2=config.lon_max, lat1=config.lat_min, lat2=config.lat_max) / (gridshape[0] * gridshape[1])

                    scores = fixResourceAvailable(resource_indexes, forecasts, realCrimes, clusters, cell_coverage_units, unit_area).rename(
                        "{} ({}x{})({})".format(method, gridshape[0], gridshape[1], threshold))
                    with open(output_filename, "wb") as ofile:
                        pickle.dump(scores, ofile)


def main():
    # Grid
    if config.grid_prediction == 1:
        print("Scoring grid predictions...")
        compute_resource_allocation(config.resource_indexes, config.cell_coverage_units,
                                    config.ug_gridshapes, config.periodsAhead_list,
                                    config.ignoreFirst, config.ug_threshold,
                                    config.ug_maxDist, config.ug_methods)

    # Clusters
    if config.cluster_prediction == 1:
        print("Scoring cluster predictions...")
        compute_resource_allocation(config.resource_indexes, config.cell_coverage_units,
                                    config.c_gridshapes, config.periodsAhead_list,
                                    config.ignoreFirst, config.c_thresholds,
                                    config.c_maxDist, config.c_methods)


if __name__ == "__main__":
    main()
