import numpy as np

from datetime import date
from datetime import timedelta



def haversine(loc1, loc2, miles=False):
    lon1, lat1, lon2, lat2 = [*loc1, *loc2]
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    #c = 2 * np.arcsin(np.sqrt(a))
    c = 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a) )
    
    km = 6371.008 * c # mean ratio
    #km = 6373 * c # optimized for Whashington DC
    
    return km*0.621371 if miles else km

def reject_outliers_IQR_range(data, iq_range=[0.25, 0.75], returnMask=False):
    qlow, qhigh = data.dropna().quantile(iq_range)
    return data[(qlow<=data) & (data<=qhigh)] if not returnMask else (qlow<=data) & (data<=qhigh)

# count the amount of weeks elapsed since the begining of the database
def add_ElapsedWeeks(data, inplace=False):
    if not inplace:
        data = data.copy()
        
    minDate = data.Date.min()
    minDate = (minDate - timedelta(days=minDate.weekday())) # start the week on Monday
    
    data['ElapsedWeeks'] = (data.Date-minDate).apply(lambda f: f.days // 7)
    
    return data

# Group the DataFrame in a grid (Latitude, Longitude) of size gridshape
def gridLatLong(data, gridshape = [10,10], inplace=False, verbose=False):
    if not inplace:
        data = data.copy()
    # classify range into bins: [0,bins[, vmax is considered inside bin: bins-1
    defBin = lambda data, vmin, vmax, bins: int((data-vmin) // ((vmax-vmin) / bins)) if data < vmax else int(bins-1)
    
    data.Latitude = np.array(data.Latitude).flatten()
    data.Longitude = np.array(data.Longitude).flatten()
    LatMin, LatMax = np.array(data.Latitude).min(), np.array(data.Latitude).max()
    LonMin, LonMax = np.array(data.Longitude).min(), np.array(data.Longitude).max()
    data['LatCell'] = np.array([defBin(i, LatMin, LatMax, gridshape[0]) for i in np.array(data.Latitude)]).flatten()
    data['LonCell'] = np.array([defBin(i, LonMin, LonMax, gridshape[1]) for i in np.array(data.Longitude)]).flatten()
    # data['LatCell'] = np.array(data['LatCell']).flatten()
    # data['LonCell'] = np.array(data['LonCell']).flatten()
    if verbose:
        print('Cells with',
              haversine( ((LatMax-LatMin)/gridshape[0],0), (0,0) )*1000,
              'per',
              haversine( (0,(LonMax-LonMin)/gridshape[1]), (0,0) )*1000,
              'meters')
    return data
