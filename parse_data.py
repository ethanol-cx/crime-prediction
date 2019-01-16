import pandas as pd
import sys
import os
import config


def parse_data(ifilename, ofilename):
    ofilename = os.path.abspath(ofilename)
    data = pd.read_csv(ifilename)
    data = data[['Category', 'Latitude', 'Longitude', 'Date']]

    print("Minimum latitude: %f" % min(data["Latitude"]))
    print("Maximum latitude: %f" % max(data["Latitude"]))
    print()
    print("Minimum longitude: %f" % min(data["Longitude"]))
    print("Maximum longitude: %f" % max(data["Longitude"]))
    print("Number of datapoints before selecting: {}".format(len(data.index)))
    # Select square window
    data = data[(config.lon_min <= data.Latitude) & (data.Latitude <= config.lon_max)
                & (config.lat_min <= data.Longitude) & (data.Longitude <= config.lat_max)]
    print("Number of datapoints after selecting: {}".format(len(data.index)))
    data.to_pickle(ofilename)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise Exception(
            "Usage: python parse_data.py [input file name].pkl [output file name].pkl")
    parse_data(sys.argv[1], sys.argv[2])
