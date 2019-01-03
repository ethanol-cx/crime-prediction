import pysal as ps
from scipy import sparse
from fwdfiles.general_functions import *
import pandas as pd


def computeClustersAndOrganizeData(ts, gridshape, ignoreFirst, threshold, maxDist):

    # create grid: assign crimes to cells

    dataGrid = pd.DataFrame(gridLatLong(ts, gridshape))
    ts = pd.DataFrame({'Crimes': dataGrid.groupby(
        ['LatCell', 'LonCell', 'Date']).Date.count()}).reset_index()
    ts.index = pd.DatetimeIndex(ts.Date)

    # compute clusters using threshold and max distance from the border
    clusters = computeClusters(
        ts, ignoreFirst, threshold=threshold, maxDist=maxDist, gridshape=gridshape)
    clusters = clusters.sort_values(by=['Crimes'], ascending=False)

    #
    # Organize data to weekly crimes
    #
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
    iGeo = 0
    # store expanded clusters (do not copy the result in a new variable cuz it may resort the lines)
    alreadyExpanded = set()
    while(iGeo < trainingGrid.shape[0]):
        row = trainingGrid.iloc[iGeo]
        # check if it is a new cluster
        if row.Cluster in alreadyExpanded:
            continue
        alreadyExpanded.add(row.Cluster)
        # retrieve colisions and sort by relevance
        colisions = trainingGrid.loc[trainingGrid.Cluster.isin(
            row.Colisions)].sort_values(by=['Crimes'], ascending=False)
        # ignore agglutination if greater than threshold (or with itself)
        # colisions = colisions.loc[np.array(colisions.Cluster!=row.Cluster) & np.array((colisions.Crimes+row.Crimes)<=threshold)]
        colisions = colisions.loc[[(x not in alreadyExpanded)
                                   for x in colisions.Cluster]]

        possible = 0
        while trainingGrid.loc[row.name, 'Crimes'] < threshold and possible < colisions.shape[0]:
            # commit agglutination (using the smaller id possible)
            while (possible < colisions.shape[0]):
                aggRow = colisions.iloc[possible]
                possible += 1
                # do not use clusters already expanded
                if aggRow.Cluster in alreadyExpanded:
                    continue
                alreadyExpanded.add(aggRow.Cluster)

                # add the colisions/neighbours of the newly merged cell that belong to no clusters into our potential merging list of thie "row" cell.
                newColisions = trainingGrid.loc[trainingGrid.Cluster.isin(
                    aggRow.Colisions)].sort_values(by=['Crimes'], ascending=False)
                newColisions = newColisions.loc[[
                    (x not in alreadyExpanded) for x in newColisions.Cluster]]
                colisions = pd.concat([colisions, newColisions], axis=0)

                trainingGrid.loc[row.name, 'Crimes'] += aggRow.Crimes

                # change the cluster label to row.Cluster
                newLabel = aggRow.Geometry.copy()
                newLabel.data = np.array([row.Cluster])

                trainingGrid.set_value(
                    row.name, 'Geometry', trainingGrid.loc[row.name, 'Geometry'] + newLabel)
                trainingGrid.drop(aggRow.name, inplace=True)
        # compute next geometry
        iGeo += 1
    print(trainingGrid)
    return trainingGrid


def initializeGeometries(ts, ignoreFirst, gridshape):
    trainingGrid = add_ElapsedWeeks(ts.copy()).query('ElapsedWeeks < {}'.format(ignoreFirst))\
        .groupby(by=['LatCell', 'LonCell'])['Crimes'].sum().reset_index().sort_values(by=['Crimes'], ascending=False)
    trainingGrid.set_index(['LatCell', 'LonCell'], inplace=True)
    trainingGrid['Cluster'] = np.array(range(trainingGrid.shape[0])) + 1
    trainingGrid['Geometry'] = [sparse.coo_matrix(
        ([g+1], ([lat], [lon])), shape=gridshape).tocsr() for (g, (lat, lon)) in enumerate(trainingGrid.index)]
    print(trainingGrid['Geometry'])
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
    mask_unit = np.array(
        [(-1, -1), (-1, 0), (0, -1), (1, 1), (1, 0), (0, 1), (-1, 1), (1, -1)])
    mask = [(0, 0)]

    # create grid board with the clustering labels
    grid = sparse.csr_matrix(gridshape, dtype=np.int32)
    for row in trainingGrid.Geometry.values:
        grid = row + grid

    # Increase thickness of the border for colision
    for _ in range(maxDist):
        # expand border mask
        mask = np.array(list(set(map(tuple, np.concatenate(
            [coord + mask_unit for coord in mask]))) - {(0, 0)}))
        trainingGrid = agglutinateCollisions(
            trainingGrid, threshold, grid, mask, gridshape)

    return trainingGrid
