# General
ignoreFirst = 104
periodsAhead_list = [1,8,64]
# periodsAhead_list = [1, 32]
# periodsAhead_list = [1]
# Uniform grids
ug_gridshapes = [(8,8),(16,16),(32,32)]
# ug_gridshapes = [(1,1), (4,4)]
ug_maxDist = 0
ug_threshold = 0
ug_methods = ["mm", "ar", "harmonic"]
# ug_methods = ["ar","harmonic"]
# Clusters
c_gridshape = (32,32)
c_thresholds = [0, 250, 500, 1500, 2000, 3000, 4000, 5000, 7000, 10000]
# c_thresholds = [0, 100, 200, 300]
# c_thresholds = [0]

c_maxDist = 1
c_methods = ["mm", "ar", "harmonic"]
# c_methods = ["harmonic"]

# Evaluation
resource_indexes = [100, 300, 600, 1100, 1600, 2100, 2600, 3100, 3600]
# resource_indexes = [0, 250, 500, 750, 1000, 1250, 1500]

cell_coverage_units = 1
