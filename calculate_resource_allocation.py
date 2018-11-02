import pickle
import sys
import os
# sys.path.append(os.path.abspath("fwdfiles/"))
from fwdfiles.resourceAllocation_functions import fixResourceAvailable
import config


def compute_ra_grid(resource_indexes, cell_coverage_units, gridshapes,
                    periodsAhead_list, ignoreFirst, threshold, dist):
    for periodsAhead in periodsAhead_list:
        os.makedirs(os.path.abspath("results/"), exist_ok=True)
        os.makedirs(os.path.abspath("results/grid"), exist_ok=True)
        output_filename = os.path.abspath("results/grid/grid_{}ahead.pkl".format(periodsAhead))
        
        mm_results = []
        ar_results = []
        harmonic_results = []
        
        for gridshape in gridshapes:
            mm_file = os.path.abspath(
                "results/mm/mm_predictions_grid({},{})_ignore({})_ahead({})_threshold({})_dist({}).pkl".format(
                    gridshape[0], gridshape[1], ignoreFirst, periodsAhead, threshold, dist
                )
            )
            ar_file = os.path.abspath(
                "results/ar/ar_predictions_grid({},{})_ignore({})_ahead({})_threshold({})_dist({}).pkl".format(
                    gridshape[0], gridshape[1], ignoreFirst, periodsAhead, threshold, dist
                )
            )
            harmonic_file = os.path.abspath(
                "results/harmonic/harmonic_predictions_grid({},{})_ignore({})_ahead({})_threshold({})_dist({}).pkl".format(
                    gridshape[0], gridshape[1], ignoreFirst, periodsAhead, threshold, dist
                )
            )
            
            with open(mm_file, "rb") as ifilemm:
                mm_clusters, mm_realCrimes, mm_forecasts = pickle.load(ifilemm)
            with open(ar_file, "rb") as ifilear:
                ar_clusters, ar_realCrimes, ar_forecasts = pickle.load(ifilear)
            with open(harmonic_file, "rb") as ifileh:
                harmonic_clusters, harmonic_realCrimes, harmonic_forecasts = pickle.load(ifileh)
                
            mm_scores = fixResourceAvailable(resource_indexes, 0, mm_forecasts, mm_realCrimes, mm_clusters, 
                                             cell_coverage_units, gridshape[0], gridshape[1]).rename("mm ({}x{})".format(gridshape[0], gridshape[1]))
            ar_scores = fixResourceAvailable(resource_indexes, ignoreFirst, ar_forecasts, ar_realCrimes, ar_clusters, 
                                             cell_coverage_units, gridshape[0], gridshape[1]).rename("ar ({}x{})".format(gridshape[0], gridshape[1]))
            harmonic_scores = fixResourceAvailable(resource_indexes, ignoreFirst, harmonic_forecasts, harmonic_realCrimes, 
                                                   harmonic_clusters, cell_coverage_units, gridshape[0], gridshape[1]).rename("h ({}x{})".format(
                                                       gridshape[0], gridshape[1]))
            
            mm_results.append(mm_scores)
            ar_results.append(ar_scores)
            harmonic_results.append(harmonic_scores)
            
        with open(output_filename, "wb") as ofile:
            pickle.dump((mm_results, ar_results, harmonic_results), ofile)


def compute_ra_clustering(resource_indexes, cell_coverage_units, gridshape,
                          periodsAhead_list, ignoreFirst, thresholds, dist):
    for periodsAhead in periodsAhead_list:
        os.makedirs(os.path.abspath("results/"), exist_ok=True)
        os.makedirs(os.path.abspath("results/cluster"), exist_ok=True)
        output_filename = os.path.abspath("results/cluster/cluster_{}ahead.pkl".format(periodsAhead))
        output_filename = os.path.abspath("results/cluster/cluster_{}ahead.pkl".format(periodsAhead))

        mm_results = []
        ar_results = []
        harmonic_results = []
        mm_clusters_count = []
        ar_clusters_count = []
        harmonic_clusters_count = []
        for threshold in thresholds:
            mm_file = os.path.abspath(
                "results/mm/mm_predictions_grid({},{})_ignore({})_ahead({})_threshold({})_dist({}).pkl".format(
                    gridshape[0], gridshape[1], ignoreFirst, periodsAhead, threshold, dist
                )
            )
            ar_file = os.path.abspath(
                "results/ar/ar_predictions_grid({},{})_ignore({})_ahead({})_threshold({})_dist({}).pkl".format(
                    gridshape[0], gridshape[1], ignoreFirst, periodsAhead, threshold, dist
                )
            )
            harmonic_file = os.path.abspath(
                "results/harmonic/harmonic_predictions_grid({},{})_ignore({})_ahead({})_threshold({})_dist({}).pkl".format(
                    gridshape[0], gridshape[1], ignoreFirst, periodsAhead, threshold, dist
                )
            )
            
            with open(mm_file, "rb") as ifilemm:
                mm_clusters, mm_realCrimes, mm_forecasts = pickle.load(ifilemm)
            with open(ar_file, "rb") as ifilear:
                ar_clusters, ar_realCrimes, ar_forecasts = pickle.load(ifilear)
            with open(harmonic_file, "rb") as ifileh:
                harmonic_clusters, harmonic_realCrimes, harmonic_forecasts = pickle.load(ifileh)
                
            mm_scores = fixResourceAvailable(resource_indexes, 0, mm_forecasts, mm_realCrimes, mm_clusters, 
                                             cell_coverage_units, gridshape[0], gridshape[1]).rename("mm ({})".format(threshold))
            ar_scores = fixResourceAvailable(resource_indexes, ignoreFirst, ar_forecasts, ar_realCrimes, ar_clusters, 
                                             cell_coverage_units, gridshape[0], gridshape[1]).rename("ar ({})".format(threshold))
            harmonic_scores = fixResourceAvailable(resource_indexes, ignoreFirst, harmonic_forecasts, harmonic_realCrimes, 
                                                   harmonic_clusters, cell_coverage_units, gridshape[0], gridshape[1]).rename("h ({})".format(threshold))
            
            mm_results.append(mm_scores)
            ar_results.append(ar_scores)
            harmonic_results.append(harmonic_scores)
            mm_clusters_count.append(len(mm_clusters))
            ar_clusters_count.append(len(ar_clusters))
            harmonic_clusters_count.append(len(harmonic_clusters))
        with open(output_filename, "wb") as ofile:
            pickle.dump((mm_results, ar_results, harmonic_results, mm_clusters_count, ar_clusters_count, harmonic_clusters_count), ofile)

def main():
    # Grid
    print("Scoring grid predictions...")
    compute_ra_grid(config.resource_indexes, config.cell_coverage_units,
                    config.ug_gridshapes, config.periodsAhead_list,
                    config.ignoreFirst, config.ug_threshold, 
                    config.ug_maxDist)

    # # Clusters
    print("Scoring cluster predictions...")
    compute_ra_clustering(config.resource_indexes, config.cell_coverage_units,
                          config.c_gridshape, config.periodsAhead_list,
                          config.ignoreFirst, config.c_thresholds,
                          config.c_maxDist)


if __name__ == "__main__":
    main()
