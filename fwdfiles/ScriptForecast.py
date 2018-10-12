import pickle
import numpy as np
import pysal as ps
from scipy import sparse
from datetime import date
from datetime import timedelta
import pandas as pd

# prediction functions
import fwdfiles.forecast_harmonic as forecast_harmonic
import fwdfiles.forecast_mean as forecast_mean
import fwdfiles.forecast_ar as forecast_ar

# General definitions for data manipulation
from fwdfiles.general_functions import *

# Clustering functions
from fwdfiles.cluster_functions import *


# compute the grid and organize the data into custers and weekly forecast
def computeClustersAndOrganizeData(ts, gridshape=(60, 85), ignoreFirst=149, periodsAhead=52, threshold=4000, maxDist=5):

    # create grid: assign crimes to cells
    np.set_printoptions(threshold=np.nan)

    dataGrid = gridLatLong(ts.copy(), gridshape, verbose=True)
    dataGrid = pd.DataFrame(dataGrid)
    newDataGrid = pd.DataFrame(
        {'Category': np.array(dataGrid['Category']).flatten(), 'Latitude': np.array(dataGrid['Latitude']).flatten(), 'Longitude': np.array(dataGrid['Longitude']).flatten(), 'Timestamp': np.array(dataGrid['Timestamp']).flatten(), 'Date': np.array(dataGrid['Date']).flatten(), 'Hour': np.array(dataGrid['Hour']).flatten(), 'LatCell': np.array(dataGrid['LatCell']).flatten(), 'LonCell': np.array(dataGrid['LonCell']).flatten()})
    newDataGrid = newDataGrid[(newDataGrid.Category == 'BURGLARY') | (
        newDataGrid.Category == 'BURGLARY FROM VEHICLE')
        | (newDataGrid.Category == 'THEFT FROM MOTOR VEHICLE - GRAND ($400 AND OVER)') | (newDataGrid.Category == 'THEFT FROM MOTOR VEHICLE - PETTY ($950 & UNDER)')
        | (newDataGrid.Category == 'THEFT PLAIN - PETTY ($950 & UNDER)') | (newDataGrid.Category == 'THEFT OF IDENTITY')
        | (newDataGrid.Category == 'VANDALISM - MISDEAMEANOR ($399 OR UNDER)')]
    ts = newDataGrid.groupby(['LatCell', 'LonCell', 'Date']).Hour.count().rename(
        'Crimes').reset_index()
    ts.index = pd.DatetimeIndex(ts.Date)

    # compute clusters using threshold and max distance from the border
    clusters, grid = computeClusters(
        ts, ignoreFirst, threshold=threshold, maxDist=maxDist, gridshape=gridshape)
    clusters = clusters.sort_values(by=['Crimes'], ascending=False)

    # Organize data to weekly crimes
    # create a dataframe with all weeks
    # range of prediction, filling missing values with 0
    dailyIdx = pd.date_range(
        start=ts.Date.min(), end=ts.Date.max(), freq='D', name='Date')
    weeklyIdx = pd.date_range(
        start=ts.Date.min(), end=ts.Date.max(), freq='W', name='Date')

    # dataframe with real crimes and clusters
    realCrimes = pd.DataFrame().reindex(weeklyIdx)
    # add crimes per cluster
    for selCluster in clusters.Cluster.values:
        # find geometry (cells) of the cluster
        selGeometry = clusters[clusters.Cluster ==
                               selCluster].Geometry.values[0].nonzero()
        # filter crimes inside the geometry
        weeks = ts.copy().set_index(['LatCell', 'LonCell']).loc[list(
            zip(*selGeometry))].groupby(by=['Date']).Crimes.sum().reset_index()
        weeks.index = pd.DatetimeIndex(weeks.Date)
        # resample to weekly data and normalize index
        weeks = weeks.Crimes.reindex(dailyIdx).fillna(0).resample('W').mean().reindex(
            weeklyIdx).fillna(0).rename('C{}_Crimes'.format(selCluster))

        for value in clusters[clusters.Cluster == selCluster].Geometry.values[1:]:
            selGeometry = value.nonzero()
            # filter crimes inside the geometry
            weeks_t = ts.copy().set_index(['LatCell', 'LonCell']).loc[list(
                zip(*selGeometry))].groupby(by=['Date']).Crimes.sum().reset_index()
            weeks_t.index = pd.DatetimeIndex(weeks_t.Date)
            # resample to weekly data and normalize index
            weeks_t = weeks_t.Crimes.reindex(dailyIdx).fillna(0).resample('W').mean(
            ).reindex(weeklyIdx).fillna(0).rename('C{}_Crimes'.format(selCluster))
            weeks += weeks_t
        # save crimes in dataframe
        realCrimes = realCrimes.join(weeks)

    return (clusters, realCrimes)


def savePredictions(clusters, realCrimes, forecasts, gridshape=(60, 85), ignoreFirst=149, periodsAhead=52, threshold=4000, maxDist=5):
    # save it to disk
    fileName = 'predictions_grid({},{})_ignore({})_ahead({})_threshold({})_dist({}).pkl'.format(
        *gridshape, ignoreFirst, periodsAhead, threshold, maxDist)
    output = open(fileName, 'wb')
    pickle.dump((clusters, realCrimes, forecasts), output)
    output.close()
    return
