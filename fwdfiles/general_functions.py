import numpy as np
import pandas as pd
from datetime import date
from datetime import timedelta
import os
import pickle
import math
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
# count the amount of weeks elapsed since the begining of the database
from math import radians, cos, sin, asin, sqrt


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


def add_ElapsedWeeks(data, inplace=False):
    if not inplace:
        data = data.copy()
    data.Date = [pd.to_datetime(
        x, format='%m/%d/%Y', errors='ignore') for x in data.Date]
    minDate = data.Date.min()
    # start the week on Monday
    minDate = (minDate - timedelta(days=minDate.weekday()))

    data['ElapsedWeeks'] = (data.Date-minDate).apply(lambda f: f.days // 7)

    return data

# Group the DataFrame in a grid (Latitude, Longitude) of size gridshape


def gridLatLong(data, gridshape, inplace=False):
    if not inplace:
        data = data.copy()
    # classify range into bins: [0,bins]

    def defBin(data, vmin, vmax, bins): return int((data-vmin) //
                                                   ((vmax-vmin) / bins)) if data < vmax else int(bins-1)

    LatMin, LatMax = np.array(
        data.Latitude).min(), np.array(data.Latitude).max()
    LonMin, LonMax = np.array(
        data.Longitude).min(), np.array(data.Longitude).max()
    data['LatCell'] = [defBin(i, LatMin, LatMax, gridshape[0])
                       for i in data.Latitude]
    data['LonCell'] = [defBin(i, LonMin, LonMax, gridshape[1])
                       for i in data.Longitude]
    return data


def savePredictions(clusters, realCrimes, forecasts, method,
                    gridshape, ignoreFirst,
                    periodsAhead, threshold, maxDist):
    # save it to disk
    os.makedirs(os.path.abspath("results/"), exist_ok=True)
    os.makedirs(os.path.abspath("results/{}".format(method)), exist_ok=True)
    fileName = os.path.abspath(
        "results/{}/{}_predictions_grid({},{})_ignore({})_ahead({})_threshold({})_dist({}).pkl".format(
            method, method, *gridshape, ignoreFirst, periodsAhead, threshold, maxDist
        )
    )
    output = open(fileName, 'wb')
    pickle.dump((clusters, realCrimes, forecasts), output)
    output.close()
    return


def saveParameters(orders, seasonal_orders, method,
                   gridshape, cluster_id, ignoreFirst,
                   threshold, maxDist):
    # save it to disk
    os.makedirs(os.path.abspath("parameters/"), exist_ok=True)
    os.makedirs(os.path.abspath("parameters/{}".format(method)), exist_ok=True)
    fileName = os.path.abspath(
        "parameters/{}/{}_parameters_grid({},{})_cluster({})_ignore({})_threshold({})_dist({}).pkl".format(
            method, method, *gridshape, cluster_id, ignoreFirst, threshold, maxDist
        )
    )
    output = open(fileName, 'wb')
    pickle.dump((orders, seasonal_orders), output)
    output.close()
    return


def getAreaFromLatLon(lon1, lon2, lat1, lat2):
    return (math.pi / 180) * 10 ** 6 * math.fabs(math.sin(lat1) - math.sin(lat2)) * math.fabs(lon1-lon2)


def getIfParametersExists(method, gridshape, cluster_id, ignoreFirst, threshold, maxDist):
    if Path("parameters/{}/{}_parameters_grid({},{})_cluster({})_ignore({})_threshold({})_dist({}).pkl".format(method, method, *gridshape, cluster_id, ignoreFirst, threshold, maxDist)).is_file():
        return pd.read_pickle("parameters/{}/{}_parameters_grid({},{})_cluster({})_ignore({})_threshold({})_dist({}).pkl".format(method, method, *gridshape, cluster_id, ignoreFirst, threshold, maxDist))
    return None


def plotTimeSeries(df, testPredict, file_path):
    testPredict.index = df[-len(testPredict):].index
    # plot baseline and predictions
    plt.plot(df)
    plt.plot(testPredict)
    plt.savefig(file_path)
    plt.close()
