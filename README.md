Credit to: Data Science Lab at USC


Edit config.py for parameter settings:
    ignoreFirst - int: Minimum amount of training periods
    periodsAhead_list - List of ints: Periods ahead to forecast
    ug_gridshapes - List of tuples: # of cells along latitude & longitude (for uniform grid method)
    ug_maxDist - Leave at 0 (for uniform grid method)
    ug_threshold - Leave at 0 (for uniform grid method)
    ug_methods - List of str: Any of ["mm", "ar", "harmonic]. Forecasting algorithms to use (for uniform grid method)
    c_gridshape - Tuple: # of cells along latitude & longitude (for cluster method)
    c_thresholds - int: Threshold of clustering (for cluster method)
    c_maxDist - int: Neighborhood distance of clustering (for cluster method)
    c_methods - List of str: Any of ["mm", "ar", "harmonic]. Forecasting algorithms to use (for cluster method)
    resource_indexes - List of int: List of amount of resources to use for evaluation (RA calculation)
    cell_coverage_units - int: Number of resources needed to cover each cell (RA calculation)


Sample usage for forecasting & evaluation:
    python parse_data.py data14to16sorted.pkl LAdata.pkl
    python make_predictions.py LAdata.pkl
    python calculate_resource_allocation.py


To plot, open a Jupyter notebook (command: jupyter notebook) and run the Resource Allocation Plots.ipynb file


Pipeline (for each configuration): parse_data.py -> make_predictions.py -> calculate_resource_allocation.py -> Resource Allocation Plots.ipynb
