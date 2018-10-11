import pysal as ps
from scipy import sparse
from fwdfiles.general_functions import *
import pandas as pd


# compute the border after dilatation
def computeDilatation(row, mask, gridshape):
    geo = row['Geometry']

    # given a geometry, dilatate its border
    dilatation = np.concatenate( [ (mask + (lat, lon)) for (lat,lon) in zip(*geo.nonzero())] )
    border = [ (lat,lon) for lat,lon in set(map(tuple, dilatation.tolist())) if
              lat>=0 and lat<gridshape[0] and lon>=0 and lon<gridshape[1]]
    return border

# compute the colisions of the dilatation with the grid
def computeColisions(row, grid, gridshape):
    dilatation = row['Dilatation']
    colisions = set([grid[lat,lon] for (lat,lon) in dilatation if (lat,lon) in zip(*grid.nonzero())])
    return list(colisions - {row.Cluster})

# innerloop responsable to the agglutination
def agglutinateCollisions(trainingGrid, threshold, grid, mask, gridshape):
    # compute dilatation border for all geometries
    trainingGrid['Dilatation'] = [computeDilatation(row, mask, gridshape) for key,row in trainingGrid.iterrows()]
    trainingGrid['Colisions'] = [computeColisions(row, grid, gridshape) for key,row in trainingGrid.iterrows()]
    # expand one neighbor per geometry
    iGeo = 0
    alreadyExpanded = set() # store expanded clusters (do not copy the result in a new variable cuz it may resort the lines)
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
        colisions = colisions.loc[[(x not in alreadyExpanded) for x in colisions.Cluster]]

        possible = 0
        while trainingGrid.loc[row.name, 'Crimes'] < threshold and colisions.shape[0] > possible:
            # commit agglutination (using the smaller id possible)
            while (colisions.shape[0]>possible):
                aggRow = colisions.iloc[possible]
                possible += 1
                # do not use clusters already expanded
                if aggRow.Cluster in alreadyExpanded:
                    continue

                alreadyExpanded.add(aggRow.Cluster)

                #add the colisions/neighbours of the newly merged cell that belong to no clusters into our potential merging list of thie "row" cell.
                newColisions = trainingGrid.loc[trainingGrid.Cluster.isin(
                    aggRow.Colisions)].sort_values(by=['Crimes'], ascending=False)
                newColisions = newColisions.loc[[
                    (x not in alreadyExpanded) for x in newColisions.Cluster]]
                pd.concat([colisions, newColisions], axis=0)

                # identify the smaller cluster id
                # if (row.Cluster>aggRow.Cluster):
                #     row,aggRow = aggRow,row

                trainingGrid.loc[row.name,'Crimes'] += aggRow.Crimes

                # change the cluster label to row.Cluster
                newLabel = aggRow.Geometry.copy()
                newLabel.data = np.array([row.Cluster])
                # newLabel.data = np.array([row.Cluster]*len(newLabel.data))

                trainingGrid.set_value(row.name, 'Geometry', trainingGrid.loc[row.name,'Geometry'] + newLabel)
                trainingGrid.drop(aggRow.name, inplace=True)
                break # just expand once

            #else:
                #print('Geometry {} (cluster {}) without possible agglutination (this round)'.format(geo, row.Cluster))

        # compute next geometry
        iGeo += 1
    return (trainingGrid, grid)

def initializeGeometries(ts, ignoreFirst, gridshape):
    trainingGrid = add_ElapsedWeeks(ts.copy()).query('ElapsedWeeks < {}'.format(ignoreFirst))\
        .groupby(by=['LatCell', 'LonCell'])['Crimes'].sum().reset_index().sort_values(by=['Crimes'], ascending=False)
    trainingGrid.set_index(['LatCell', 'LonCell'], inplace=True)

    trainingGrid['Cluster'] = np.array(range(trainingGrid.shape[0])) + 1
    trainingGrid['Geometry'] = [sparse.coo_matrix( ([g+1], ([lat], [lon])), shape=gridshape).tocsr() for (g, (lat, lon)) in enumerate(trainingGrid.index)]
    trainingGrid['Dilatation'] = None
    return trainingGrid

def computeClusters(ts, ignoreFirst, threshold = 500, maxDist = 85, gridshape=(60,85)):
    """Compute clusters agglutinating collisions

    Keyword arguments:
    ts        -- Date indexed DataFrame with ['LatCell', 'LonCell', 'Crimes']
    maxDist   -- consider up to maxDist(included) units ahead: [1, maxDist]
    threshold -- max amount of crimes per cluster

    Return: cluster of the training grid
    """

    # initialize the trainingGrid with geometries
    trainingGrid = initializeGeometries(ts, ignoreFirst, gridshape)
    print(trainingGrid['Crimes'])

    # dilatation mask
    # * * *
    # * x *        while (initial != trainingGrid.shape[0]):
    # * * *
    mask_unit = np.array([(-1,-1), (-1,0), (0,-1), (1,1), (1,0), (0,1), (-1,1), (1,-1)])
    mask = [(0,0)]

    # create grid board with the clustering labels
    grid = sparse.csr_matrix(gridshape, dtype=np.int32)
    for row in trainingGrid.Geometry.values:
        grid = row.tocsr() + grid

    # Increase thickness of the border for colision: [1, maxDist]
    for sqrt_dist in range (1, maxDist+1):

        # expand border mask
        mask = np.array( list(set(map(tuple, np.concatenate([coord + mask_unit for coord in mask]) )) - {(0,0)}) )

        trainingGrid, grid = agglutinateCollisions(trainingGrid, threshold, grid, mask, gridshape)

    return (trainingGrid, grid)
