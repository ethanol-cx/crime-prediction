# General
ignoreFirst = 225
periodsAhead_list = [1, 16, 32]
lon_min = 34.015
lon_max = 34.038
lat_min = -118.297
lat_max = -118.27
# Uniform grids
grid_prediction = 1
ug_gridshapes = [(16, 16)]
ug_maxDist = 0
ug_threshold = [0]
ug_methods = ["AR"]

# Clusters
cluster_prediction = 1
c_gridshapes = [(16, 16)]
c_thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# c_thresholds = [20,40,60,100, 200, 300, 400]
c_maxDist = 1
c_methods = ["AR"]

# Evaluation
resource_indexes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                    100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
cell_coverage_units = 5
