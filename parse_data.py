import pandas as pd
import sys
import os


def parse_data(ifilename, ofilename):
    ofilename = os.path.abspath(ofilename)
    data = pd.read_pickle(ifilename)
    data = data[['CaseNbr', 'Category', 'Latitude', 'Longitude', 'Date', ]]

    print("Minimum latitude: %f" % min(data["Latitude"]))
    print("Maximum latitude: %f" % max(data["Latitude"]))
    print()
    print("Minimum longitude: %f" % min(data["Longitude"]))
    print("Maximum longitude: %f" % max(data["Longitude"]))
    print("Number of datapoints before selecting: {}".format(len(data.index)))
    # Select square window
    data = data[(34.015 <= data.Latitude) & (data.Latitude <= 34.038)
                & (-118.297 <= data.Longitude) & (data.Longitude <= -118.27)]
    data.columns = [["#", "Category", "Latitude", "Longitude", "Date"]]
    data = data.set_index("#")
    print("Number of datapoints after selecting: {}".format(len(data.index)))
    data.to_pickle(ofilename)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise Exception(
            "Usage: python parse_data.py [input file name].pkl [output file name].pkl")
    parse_data(sys.argv[1], sys.argv[2])
