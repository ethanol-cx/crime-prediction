import argparse

def checkSyntax():
    
    # Script syntax
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="CSV database with crimes", type=argparse.FileType('r'))
    parser.add_argument("q_lat", help="amount of cell along latitude", type=int)
    parser.add_argument("q_lon", help="amount of cells along longitude", type=int)
    parser.add_argument("qt_train", help="min amount of training periods", type=int)
    parser.add_argument("ahead", help="periods ahead to forecast", type=int)
    parser.add_argument("threshold", help="threshold of clustering", type=int)
    parser.add_argument("border", help="neighborhood distance of clustering", type=int)
    parser.add_argument('-no_cluster', '--no_cluster', help='perform predictions without clustering. Similar to threshold=0 and border=0.', action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-f_h', '--f_harmonic', help='harmonic with ARIMA error', action="store_true")
    group.add_argument('-f_ar', '--f_autoregressive', help='AR(5) model', action="store_true")
    group.add_argument('-f_mm', '--f_mm_window', nargs=1, type=int, default=[5], help='(default) forecast method: moving mean. Window value F_MM_WINDOW (default 5)')
    
    class Configuration:
        args = None
        filename = None
        gridshape = None
        ignoreFirst = None
        periodsAhead = None
        threshold = None
        maxDist = None
        f_mm_window = None
        no_cluster = None
    conf = Configuration()

    args = parser.parse_args()
    conf.args = args
    conf.filename = args.file
    conf.gridshape = (args.q_lat,args.q_lon)
    conf.ignoreFirst = args.qt_train
    conf.periodsAhead = args.ahead
    conf.threshold = args.threshold
    conf.maxDist = args.border
    conf.f_mm_window = args.f_mm_window[0]
    conf.no_cluster = args.no_cluster

    if (conf.no_cluster):
        conf.threshold = 0
        conf.maxDist = 0

    if (args.f_harmonic):
        print('Using harmonic prediction with ARIMA error')
    elif (conf.args.f_autoregressive):
        print('Using AR(5) model')
    elif(conf.f_mm_window>0):
        print('Using moving mean with window {}'.format(conf.f_mm_window))
    else:
        print('Incorrect window value for moving mean forecasts')
        exit(-1)
        
    return conf
