import pandas as pd
import numpy as np

# assign resources optimally, for one specific date


def assignResources(forecast, clusters, cell_coverage_units=1, available_resources=40):
    cluster_order = clusters.Cluster.values
    # reset allocations
    allocation = np.zeros(len(cluster_order))

    # assign allocations in order
    geo = 0
    while(available_resources > 0 and geo < len(cluster_order)):
        cluster = 'C{}_Forecast'.format(cluster_order[geo])
        # only forecasts not negative
        if (forecast.loc[cluster] > 0):
            cluster_area = clusters.iloc[geo].Area
            # allocate as many resouces as needed, without exceeding the amount available
            amount = min(forecast.loc[cluster]*cluster_area //
                         cell_coverage_units, available_resources)
            allocation[geo] = amount
            # update available resources
            available_resources -= amount
        geo += 1

    return (available_resources, allocation)


def assignRemainingUniformly(allocation, available_resources=40):
    # distribute resources accross clusters uniformly (better on clusters with similar area)
    # a better approach would use the area of each cluster
    allocation = [x + available_resources//len(allocation) for x in allocation]
    available_resources = available_resources % len(allocation)

    # assign allocations in order
    geo = 0
    while(available_resources > 0 and geo < len(allocation)):
        # allocate remaining
        allocation[geo] += 1
        available_resources -= 1
        geo += 1

    return allocation

# compute the fixed metric for a specific date


def computeFixedMetricAllForecasts(forecasts, realCrimes, date_idx, clusters, cell_coverage_units=1, available_resources=10000):
    # forecast to compute
    forecast = forecasts.iloc[date_idx]

    # assign resources not exceeding predictions
    available, allocation = assignResources(
        forecast, clusters, cell_coverage_units, available_resources)
    # distribute remaining resources accross the clusters
    if available > 0:
        allocation = assignRemainingUniformly(allocation, available)

    #
    # computing the resource allocation evaluation
    #

    # potential crimes avoided
    potential = pd.Series([allocation[i] * cell_coverage_units /
                           clusters.Area.values[i] for i in range(len(clusters.Area.values))])
    # Result: real crimes avoided
    realCrimeIdx = len(realCrimes) - len(realCrimes) // 3 + date_idx
    avoided = sum([min(realCrimes.iloc[realCrimeIdx][i], potential[i]) for i in range(
        len(clusters.Area.values))])
    # avoided = pd.DataFrame([(realCrimes.iloc[realCrimeIdx], potential)]).min
    return avoided


# compute amount of stopped crimes for all forecasted dates

def computeStoppedCrime(forecasts, realCrimes, clusters, cell_coverage_units, available_resources):
    # avoided crimes for each date, ignoring training data
    return ([computeFixedMetricAllForecasts(forecasts, realCrimes, date_idx, clusters, cell_coverage_units, available_resources)
             for date_idx in range(len(forecasts))])

# compute the relative metric (0-100%]) of all forecasted dates


def computeRelativeStoppedCrime(forecasts, realCrimes, clusters, cell_coverage_units=1, available_resources=10000):
    # compute all next dates, ignore training data
    return pd.Series(data=np.array(computeStoppedCrime(forecasts, realCrimes, clusters, cell_coverage_units, available_resources)) /
                     np.array(realCrimes.T.sum().iloc[-len(realCrimes) // 3 + 1:]), index=forecasts.index, name='Score')

# compute stopped crimes only, fixing the available resources


def fixResourceAvailable(resource_indexes, ignoreFirst, forecasts, realCrimes, clusters, cell_coverage_units, unit_area):
    # sort clusters by minimum area
    clusters['Area'] = clusters.Geometry.map(lambda g: g.data.size * unit_area)
    clusters.sort_values(by=['Area', 'Crimes'], ascending=[
                         True, False], inplace=True)

    # compute for all amount of available resources
    return pd.Series(data=[
        computeRelativeStoppedCrime(forecasts, realCrimes, clusters,
                                    cell_coverage_units, resources).mean()
        for resources in resource_indexes], index=resource_indexes)
