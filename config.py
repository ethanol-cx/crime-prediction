# General
ignoreFirst = 225
periodsAhead_list = [1,16]
# periodsAhead_list = [1, 32]
# periodsAhead_list = [1]
# Uniform grids
ug_gridshapes = [(32,32)]
# ug_gridshapes = [(1,1), (4,4)]
ug_maxDist = 0
ug_threshold = 0
ug_methods = ["mm", "ar", "harmonic"]
# ug_methods = ["ar","harmonic"]
# Clusters
c_gridshape = (32,32)
c_thresholds = [0, 50, 100, 200, 400, 600, 800,
                1250, 1500]

# final c_thresholds = [0, 50, 100, 200, 400, 600, 800, 1250, 1500, 1750, 2000, 2250, 2500,3000]
# c_thresholds = [0, 100, 200, 300]
# c_thresholds = [0]

c_maxDist = 1
# c_methods = ["mm", "ar", "harmonic"]
c_methods = ["mm", "ar", "harmonic"]

# Evaluation
# resource_indexes = [100, 600, 1100, 1600, 2100, 2600, 3100, 3600]
# resource_indexes = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
resource_indexes = [0,2,4,6,8,10,12,14,16,18,20]
cell_coverage_units = 1
