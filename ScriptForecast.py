#!/usr/bin/env python


import pickle
import numpy as np
import pysal as ps
from scipy import sparse

from datetime import date
from datetime import timedelta

import pandas as pd
from pandas import TimeGrouper



# prediction functions
import forecast_harmonic
import forecast_mean
import forecast_ar

# script syntax
from syntax import *

# General definitions for data manipulation
from general_functions import *

# Clustering functions
from cluster_functions import *



#
# Main script code
#

# compute the grid and organize the data into custers and weekly forecast
def computeClustersAndOrganizeData(ts, gridshape=(60,85), ignoreFirst=149, periodsAhead=52, threshold=4000, maxDist=5):

    # create grid: assign crimes to cells
    dataGrid = gridLatLong(ts.copy(), gridshape, verbose=True)
    ts = dataGrid.groupby(['LatCell', 'LonCell', 'Date']).Hour.count().rename('Crimes').reset_index()
    ts.index = pd.DatetimeIndex(ts.Date)

    # compute clusters using threshold and max distance from the border
    clusters, grid = None, None
    #if (no_cluster):
    #    clusters = initializeGeometries(ts, ignoreFirst, gridshape)
    #else:
        #clusters, grid = computeClusters(ts, ignoreFirst, threshold=threshold, maxDist=maxDist, gridshape=gridshape)
    clusters, grid = computeClusters(ts, ignoreFirst, threshold=threshold, maxDist=maxDist, gridshape=gridshape)
    clusters = clusters.sort_values(by=['Crimes'], ascending=False)


    #
    # Organize data to weekly crimes
    #
    # create a dataframe with all weeks

    # range of prediction, filling missing values with 0
    dailyIdx = pd.date_range(start=ts.Date.min(), end=ts.Date.max(), freq='D', name='Date')
    weeklyIdx = pd.date_range(start=ts.Date.min(), end=ts.Date.max(), freq='W', name='Date')

    # dataframe with real crimes and clusters
    realCrimes = pd.DataFrame().reindex(weeklyIdx)
    # add crimes per cluster
    for selCluster in clusters.Cluster.values:
        # find geometry (cells) of the cluster
        selGeometry = clusters[clusters.Cluster==selCluster].Geometry.values[0].nonzero()
        # filter crimes inside the geometry
        weeks = ts.copy().set_index(['LatCell', 'LonCell']).loc[list(zip(*selGeometry))].groupby(by=['Date']).Crimes.sum().reset_index()
        weeks.index = pd.DatetimeIndex(weeks.Date)
        # resample to weekly data and normalize index
        weeks = weeks.Crimes.reindex(dailyIdx).fillna(0).resample('W').mean().reindex(weeklyIdx).fillna(0).rename('C{}_Crimes'.format(selCluster))

        # save crimes in dataframe
        realCrimes = realCrimes.join(weeks)

    # remove week 53
    realCrimes = realCrimes.loc[realCrimes.index.strftime('%U')!='53']

    return (clusters, realCrimes)



def savePredictions(clusters, realCrimes, forecasts, gridshape=(60,85), ignoreFirst=149, periodsAhead=52, threshold=4000, maxDist=5):
    # save it to disk
    fileName = 'predictions_grid({},{})_ignore({})_ahead({})_threshold({})_dist({}).pkl'.format(*gridshape, ignoreFirst, periodsAhead, threshold, maxDist)
    output = open(fileName, 'wb')
    pickle.dump((clusters, realCrimes, forecasts), output)
    output.close()
    return



#
# Main code
#
def main():
    conf = checkSyntax()

    # Loading data
    data = pd.read_csv(conf.filename) \
        .rename(columns={'CaseNbr':'#', 'latitude':'Latitude', 'longitude':'Longitude', 'time':'Timestamp'}) \
        .set_index('#');

    #
    # cleaning data
    #
    # removing data without location
    data.drop( data.query('Latitude==0 or Longitude==0').index, inplace=True)

    # removing data with crazy timestamp
    data.drop( data[data.Timestamp<693596].index, inplace=True) # 693596: 1/1/1900

    # drop unique observation event
    data.drop(data.query('Category=="OBSERVATION"').index, inplace=True)

    # correction on Timestamp: Python starts from year 1 and data is from year 0. Apply -366
    data['Timestamp'] = data['Timestamp'] - 366

    data['Date'],data['Hour'] = zip(*[[
        date.fromordinal(int(date_s)) if int(date_s) > 0 else None,
        float("0."+hour_s)] for date_s, hour_s in [str(s).split('.') for s in data['Timestamp']] ])
    data.dropna(inplace=True)

    # selecting a square window to analyze
    print ('Latitude:', data.Latitude.min(), data.Latitude.max())
    print ('Longitude:', data.Longitude.min(), data.Longitude.max())
    print ('Selecting area of study. Change the code to choose another window!')
    data = data[(34.015<=data.Latitude) & (data.Latitude<=34.038)]
    data = data[(-118.297<=data.Longitude) & (data.Longitude<=-118.27)]

    # compute the forecasts
    print('Computing clusters ...')
    clusters, realCrimes = computeClustersAndOrganizeData(data, conf.gridshape, conf.ignoreFirst, conf.periodsAhead, conf.threshold, conf.maxDist)

    print('Computing predictions ...')
    forecasts = None
    if (conf.args.f_harmonic):
        forecast_harmonic.init_predictions_Harmonic()
        forecasts = forecast_harmonic.predictions_Harmonic(clusters, realCrimes, periodsAhead=conf.periodsAhead)
    elif (conf.args.f_autoregressive):
        forecast_ar.init_predictions_AR()
        forecasts = forecast_ar.predictions_AR(clusters, realCrimes, periodsAhead=conf.periodsAhead)
    else:
        forecast_mean.init_predictions_Mean(window=conf.f_mm_window)
        forecasts = forecast_mean.predictions_Mean(clusters, realCrimes, periodsAhead=conf.periodsAhead)
    savePredictions(clusters, realCrimes, forecasts, conf.gridshape, conf.ignoreFirst, conf.periodsAhead, conf.threshold, conf.maxDist)

    print('Done!!!')

if __name__ == "__main__":
    main()
