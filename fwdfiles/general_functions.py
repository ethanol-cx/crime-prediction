import numpy as np

from datetime import date
from datetime import timedelta

# Group the DataFrame in a grid (Latitude, Longitude) of size gridshape


def gridLatLong(data, gridshape=[10, 10], inplace=False, verbose=False):
    if not inplace:
        data = data.copy()
    # classify range into bins: [0,bins[, vmax is considered inside bin: bins-1

    def defBin(data, vmin, vmax, bins): return int((data-vmin) //
                                                   ((vmax-vmin) / bins)) if data < vmax else int(bins-1)

    data.Latitude = np.array(data.Latitude).flatten()
    data.Longitude = np.array(data.Longitude).flatten()
    LatMin, LatMax = np.array(
        data.Latitude).min(), np.array(data.Latitude).max()
    LonMin, LonMax = np.array(
        data.Longitude).min(), np.array(data.Longitude).max()
    data['LatCell'] = np.array([defBin(i, LatMin, LatMax, gridshape[0])
                                for i in np.array(data.Latitude)]).flatten()
    data['LonCell'] = np.array([defBin(i, LonMin, LonMax, gridshape[1])
                                for i in np.array(data.Longitude)]).flatten()
    if verbose:
        print('Cells with',
              haversine(((LatMax-LatMin)/gridshape[0], 0), (0, 0))*1000,
              'per',
              haversine((0, (LonMax-LonMin)/gridshape[1]), (0, 0))*1000,
              'meters')
    return data
