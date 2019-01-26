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

# compute the full metric for a specific date


def computeFullMetricAllForecasts(forecasts, realCrimes, date_idx, clusters, cell_coverage_units=1):
    # forecast to compute
    forecast = forecasts.iloc[date_idx]

    # sort clusters by minimum area (compute outside the function)
    #clusters['Area'] = clusters.Geometry.map(lambda g: g.data.size)
    #clusters.sort_values(by=['Area','Crimes'], ascending=[True, False], inplace=True)

    # assign as many resources as required by the forecast
    _, allocation = assignResources(
        forecast, clusters, cell_coverage_units, available_resources=np.inf)

    #
    # computing the resource allocation evaluation
    #

    # potential crimes avoided is forecasted crimes (as many as required)
    potential = pd.Series(forecast.values, index=[
                          f[:-len('Forecast')]+'Crimes' for f in forecast.index])
    # Result: real crimes avoided
    avoided = pd.DataFrame([realCrimes.iloc[date_idx], potential]).min()

    return (avoided.sum(), allocation.sum())


#
# The following functions compute (different) scores for a period of evaluation (ignoring first dates)
#

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

# compute the full metric for all forecasted dates


def computeFullEfficiency(forecasts, realCrimes, clusters, cell_coverage_units=1):
    #
    # Full metric:
    #                   avoided          avoided
    #          =   (------------- + ---------------)*1/2    =   [(reduced-crime efficiency)+(allocation efficiency)]*1/2
    #               #real_crimes    #resources_used
    #
    #          = avoided * (#real_crimes + #resources_used)/(2 * #real_crimes * #resources_used)
    #

    # ignoring training data
    res = pd.DataFrame(data=[[*computeFullMetricAllForecasts(forecasts, realCrimes, date_idx, clusters, cell_coverage_units)]
                             for date_idx in range(0, forecasts.shape[0])], index=forecasts.index[0:], columns=['Avoided', 'Used'])

    real_crimes = realCrimes.iloc[0:].T.sum()
    x = res.Avoided.sum()*(real_crimes.sum() + res.Used.sum())
    y = real_crimes.sum()*res.Used.sum()
    return x/y if y > 0 else None


#
# The following functions should be the main interface with scripts
#

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

# compute final efficiency for crimes stopped and assigning precision


def fullMetric_(prop_indexes, ignoreFirst, forecasts, realCrimes, clusters, cell_coverage_units):
    # sort clusters by minimum area
    clusters['Area'] = clusters.Geometry.map(lambda g: g.data.size)
    clusters.sort_values(by=['Area', 'Crimes'], ascending=[
                         True, False], inplace=True)

    # mean of the full metric (final efficiency)
    # .mean()
    return computeFullEfficiency(forecasts, realCrimes, clusters, cell_coverage_units)


def fullMetric(ignoreFirst, forecasts, realCrimes, clusters, cell_coverage_units, A_coverage_units):
    clusters['Area'] = clusters.Geometry.map(lambda g: g.data.size)

    _error = (forecasts.applymap(lambda r: 0 if r <
                                 0 else r).values - realCrimes.values)[ignoreFirst:]
    errors = pd.DataFrame(data=_error,
                          columns=[c[:-len('Forecast')] +
                                   'Error' for c in forecasts.columns],
                          index=forecasts.index[ignoreFirst:])

    # compute the absolute error (for normal absolute mean)
    errors = errors.applymap(lambda c: -c if c < 0 else c)

    # normalize errors by the area of each cluster
    # 1 error for 1 cell  -> 1 error
    # 1 error for 2 cells -> 1 error * weight 2 = 2 (less precision)
    weights = pd.Series(data=clusters.Area.values,
                        index=['C{}_Error'.format(c) for c in clusters.Cluster]).reindex(errors.columns).values
    weighted_errors = errors*weights / cell_coverage_units

    # 'mais correto' (algoritmo de cluster remove celulas sem crimes)
    # return weighted_errors.sum().sum()/(weighted_errors.shape[0] * clusters.Area.sum()*cell_coverage_units)
    return weighted_errors.sum().sum()/(weighted_errors.shape[0] * A_coverage_units)
