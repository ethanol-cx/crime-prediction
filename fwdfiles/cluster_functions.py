import pysal as ps
from scipy import sparse
import pandas as pd
from fwdfiles.general_functions import gridLatLong, add_ElapsedWeeks
import numpy as np


def computeClustersAndOrganizeData(ts, gridshape, ignoreFirst, threshold, maxDist):

    # create grid: assign crimes to cells
    dataGrid = pd.DataFrame(gridLatLong(ts, gridshape))
    ts = pd.DataFrame({'Crimes': dataGrid.groupby(
        ['LatCell', 'LonCell', 'Date']).Date.count()}).reset_index()

    # compute clusters using threshold and max distance from the border
    clusters = computeClusters(
        ts, ignoreFirst, threshold=threshold, maxDist=maxDist, gridshape=gridshape)
    clusters = clusters.sort_values(by=['Crimes'], ascending=False)
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
        filter_list = [True if tuple([ts_elem.LatCell, ts_elem.LonCell])
                       in list(zip(*selGeometry)) else False for ts_elem in ts.itertuples()]
        weeks = ts.loc[filter_list]
        weeks = weeks.groupby(['Date']).Crimes.sum().reset_index()
        weeks.index = pd.DatetimeIndex(weeks.Date)
        # resample to weekly data and normalize index
        weeks = weeks.Crimes.reindex(dailyIdx).fillna(0).resample('W').mean().reindex(
            weeklyIdx).fillna(0).rename('C{}_Crimes'.format(selCluster))
        # save crimes in dataframe
        realCrimes = realCrimes.join(weeks)

    return (clusters, realCrimes)


# compute the border after dilatation
def computeDilatation(row, mask, gridshape):
    geo = row['Geometry']
    # given a geometry, dilatate its border
    dilatation = np.concatenate([(mask + (lat, lon))
                                 for (lat, lon) in zip(*geo.nonzero())])
    border = [(lat, lon) for lat, lon in set(map(tuple, dilatation.tolist())) if
              lat >= 0 and lat < gridshape[0] and lon >= 0 and lon < gridshape[1]]
    return border


# compute the colisions of the dilatation with the grid
def computeColisions(row, grid, gridshape):
    dilatation = row['Dilatation']
    colisions = set([grid[lat, lon] for (lat, lon)
                     in dilatation if (lat, lon) in zip(*grid.nonzero())])
    return list(colisions - {row.Cluster})


# innerloop responsible for the agglutination
def agglutinateCollisions(trainingGrid, threshold, grid, mask, gridshape):
    # compute dilatation border for all geometries
    trainingGrid['Dilatation'] = [computeDilatation(
        row, mask, gridshape) for key, row in trainingGrid.iterrows()]
    trainingGrid['Colisions'] = [computeColisions(
        row, grid, gridshape) for key, row in trainingGrid.iterrows()]
    i = 0
    # store expanded clusters
    alreadyExpanded = set()
    rowNamesToDrop = []
    for i in range(len(trainingGrid)):
        row = trainingGrid.iloc[i]
        # check if it is a new cluster
        if row.Cluster in alreadyExpanded:
            continue
        alreadyExpanded.add(row.Cluster)
        # retrieve colisions and sort by relevance
        colisions = trainingGrid.loc[trainingGrid.Cluster.isin(
            row.Colisions)].sort_values(by=['Crimes'], ascending=False)
        # do not use clusters already expanded
        colisions = colisions.loc[[(x not in alreadyExpanded)
                                   for x in colisions.Cluster]]
        potential_neighbor_idx = 0
        while trainingGrid.loc[row.name, 'Crimes'] < threshold and potential_neighbor_idx < colisions.shape[0]:
           # commit agglutination
            aggRow = colisions.iloc[potential_neighbor_idx]
            potential_neighbor_idx += 1
            # this is required when the cluster in the `collisions` is referred by `newcollisions` before it is added to `alreadyExpanded`
            if aggRow.Cluster in alreadyExpanded:
                continue
            alreadyExpanded.add(aggRow.Cluster)
            trainingGrid.loc[row.name, 'Crimes'] += aggRow.Crimes
            # add the colisions/neighbours of the newly merged cell that belong to no clusters into our potential merging list of thie "row" cell.
            newColisions = trainingGrid.loc[trainingGrid.Cluster.isin(
                aggRow.Colisions)].sort_values(by=['Crimes'], ascending=False)
            newColisions = newColisions.loc[[
                (x not in alreadyExpanded) for x in newColisions.Cluster]]
            colisions = pd.concat([colisions, newColisions], axis=0)
            # add the cluster label to row.Cluster
            newLabel = aggRow.Geometry.copy()
            trainingGrid.at[row.name,
                            'Geometry'] = trainingGrid.loc[row.name, 'Geometry'] + newLabel
            trainingGrid.at[row.name, 'Colisions'] = list(colisions.Cluster)
            rowNamesToDrop.append(aggRow.name)
    for name in rowNamesToDrop:
        trainingGrid.drop(name, inplace=True)

    return trainingGrid


def initializeGeometries(ts, ignoreFirst, gridshape):
    trainingGrid = add_ElapsedWeeks(ts.copy()).query('ElapsedWeeks < {}'.format(ignoreFirst))\
        .groupby(by=['LatCell', 'LonCell'])['Crimes'].sum().reset_index().sort_values(by=['Crimes'], ascending=False)
    trainingGrid.set_index(['LatCell', 'LonCell'], inplace=True)
    for i in range(gridshape[0]):
        for j in range(gridshape[1]):
            if (i, j) not in list(trainingGrid.index.values):
                trainingGrid.ix[(i, j), :] = 0
    trainingGrid['Cluster'] = np.array(range(trainingGrid.shape[0])) + 1
    trainingGrid['Geometry'] = [sparse.coo_matrix(
        ([g+1], ([lat], [lon])), shape=gridshape).tocsr() for (g, (lat, lon)) in enumerate(trainingGrid.index)]
    trainingGrid['Dilatation'] = None
    return trainingGrid


def computeClusters(ts, ignoreFirst, threshold, maxDist, gridshape):
    """Compute clusters agglutinating collisions

    Keyword arguments:
    ts        -- Date indexed DataFrame with ['LatCell', 'LonCell', 'Crimes']
    maxDist   -- consider up to maxDist(included) units ahead: [1, maxDist]
    threshold -- max amount of crimes per cluster

    Return: cluster of the training grid
    """

    # initialize the trainingGrid with geometries
    trainingGrid = initializeGeometries(ts, ignoreFirst, gridshape)

    # dilatation mask
    # * * *
    # * x *
    # * * *
    mask = np.array(
        [(-1, -1), (-1, 0), (0, -1), (1, 1), (1, 0), (0, 1), (-1, 1), (1, -1)])

    # create grid board with the clustering labels
    grid = sparse.csr_matrix(gridshape, dtype=np.int32)
    for row in trainingGrid.Geometry.values:
        grid = row + grid

    # first phase of clutering
    print("Starting Phase One of the clutering...")
    trainingGrid = agglutinateCollisions(
        trainingGrid, threshold, grid, mask, gridshape)
    print("Finished Phase One!")

    # phase two: clearing small "holds" by merging them with the largest neighbour
    print("Starting Phase Two of the clustering...")
    filter_threshold = threshold // 10
    droppedRowNames = set()
    for i in range(trainingGrid.shape[0]):
        row = trainingGrid.iloc[i]
        if row.Crimes < filter_threshold:
            parentColisions = trainingGrid.loc[[row.Cluster in colisions for colisions in
                                                trainingGrid.Colisions]].sort_values(by=['Crimes'], ascending=False)
            for _, parentClusterRow in parentColisions.iterrows():
                if parentClusterRow.name not in droppedRowNames:
                    parentClusterRow.Crimes += row.Crimes
                    # add the cluster label to parentRow.Cluster
                    newColisions = row.Colisions.copy()
                    colisions = parentClusterRow.Colisions
                    colisions += newColisions
                    colisions = list(set(colisions))
                    newLabel = row.Geometry.copy()
                    trainingGrid.at[parentClusterRow.name,
                                    'Geometry'] = trainingGrid.loc[parentClusterRow.name, 'Geometry'] + newLabel
                    trainingGrid.at[parentClusterRow.name,
                                    'Colisions'] = colisions
                    droppedRowNames.add(row.name)
                    break
    for name in droppedRowNames:
        trainingGrid.drop(name, inplace=True)
    print("Finished Phase Two!")

    return trainingGrid
