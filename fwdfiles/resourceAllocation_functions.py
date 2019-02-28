import pandas as pd
import numpy as np

# assign resources optimally, for one specific date


def assignResources(forecast, clusters, cell_coverage_units=1, available_resources=40):
    # reset allocations
    allocation = np.zeros(len(clusters.Cluster.values))
    potentialCrimes = forecast.copy()

    # assign allocations in greedy fashion
    for _ in range(available_resources):
        maxUtility = -1
        maxLocation = -1
        for i in range(len(clusters.Cluster.values)):
            cluster = 'C{}_Forecast'.format(clusters.iloc[i].Cluster)
            currentUtility = min(
                (1 * cell_coverage_units / clusters.iloc[i].Area), potentialCrimes.loc[cluster])
            if currentUtility > maxUtility:
                maxLocation = i
                maxUtility = currentUtility
        if maxUtility <= 0:
            break
        allocation[maxLocation] += 1
        available_resources -= 1
        potentialCrimes.loc['C{}_Forecast'.format(
            clusters.iloc[maxLocation].Cluster)] -= maxUtility

    return (available_resources, allocation)


def assignRemainingUniformly(allocation, available_resources=40):
    # distribute resources accross clusters uniformly (better on clusters with similar area)
    # a better approach would use the area of each cluster
    allocation = [x + available_resources//len(allocation) for x in allocation]
    available_resources = available_resources % len(allocation)

    # assign allocations in order
    for geo in range(int(available_resources)):
        # allocate remaining
        allocation[geo] += 1

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

    # potential crimes avoided
    potential = pd.Series([allocation[i] * cell_coverage_units /
                           clusters.Area.values[i] for i in range(len(clusters.Area.values))])
    # Result: real crimes avoided
    realCrimeIdx = len(realCrimes) - len(realCrimes) // 3 + date_idx
    avoided = sum([min(realCrimes.iloc[realCrimeIdx][i], potential[i]) for i in range(
        len(clusters.Area.values))])
    return avoided


# compute amount of stopped crimes for all forecasted dates

def computeStoppedCrime(forecasts, realCrimes, clusters, cell_coverage_units, available_resources):
    # avoided crimes for each date, ignoring training data
    return ([computeFixedMetricAllForecasts(forecasts, realCrimes, date_idx, clusters, cell_coverage_units, available_resources)
             for date_idx in range(len(forecasts))])

# compute the relative metric (0-100%]) of all forecasted dates


def computeRelativeStoppedCrime(forecasts, realCrimes, clusters, cell_coverage_units=1, available_resources=10000):
    # compute all next dates, ignore training data
    return pd.Series(data=computeStoppedCrime(forecasts, realCrimes, clusters, cell_coverage_units, available_resources), index=forecasts.index, name='Score')

# compute stopped crimes only, fixing the available resources


def fixResourceAvailable(resource_indexes, forecasts, realCrimes, clusters, cell_coverage_units, unit_area):
    # sort clusters by minimum area
    clusters['Area'] = clusters.Geometry.map(lambda g: g.data.size * unit_area)
    clusters.sort_values(by=['Area', 'Crimes'], ascending=[
                         True, False], inplace=True)

    # compute for all amount of available resources
    return pd.Series(data=[
        (computeRelativeStoppedCrime(forecasts, realCrimes, clusters,
                                     cell_coverage_units, resources).sum()) / (np.array(realCrimes.T.sum().iloc[-(len(realCrimes) // 3):]).sum())
        for resources in resource_indexes], index=resource_indexes)
