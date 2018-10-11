import pandas as pd
import sys
import os


def parse_data(ifilename, ofilename):
    ofilename = os.path.abspath(ofilename)

    data = pd.read_pickle(ifilename)
    data = data[["DR Number", "Crime Code Description", "Location ",
                 "timeAndDate"]]
    data.columns = ["CaseNbr", "Category", "Location", "time"]

    def get_latitude(row):
        return float(row["Location"].split(",")[0][1:])

    def get_longitude(row):
        return float(row["Location"].split(",")[1][:-1])

    data["latitude"] = data.apply(get_latitude, axis=1)
    data["longitude"] = data.apply(get_longitude, axis=1)

    data = data[["CaseNbr", "Category", "latitude", "longitude", "time"]]
    data = data[(data["latitude"] != 0) & (data["longitude"] != 0)]

    print("Minimum latitude: %f" % min(data["latitude"]))
    print("Maximum latitude: %f" % max(data["latitude"]))
    print()
    print("Minimum longitude: %f" % min(data["longitude"]))
    print("Maximum longitude: %f" % max(data["longitude"]))

    # Select square window
    data = data[(data["latitude"] > 33.35) &
                (data["latitude"] < 34.35) &
                (data["longitude"] > -118.75) &
                (data["longitude"] < -117.75)]
    data.columns = [["#", "Category", "Latitude", "Longitude", "Timestamp"]]

    def get_date(row):
        # print(row["Timestamp"])
        return row["Timestamp"].dt.date

    def get_hour(row):
        return row["Timestamp"].dt.time

    data["Date"] = data.apply(get_date, axis=1)
    data["Hour"] = data.apply(get_hour, axis=1)

    data = data.set_index("#")
    data.to_pickle(ofilename)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise Exception("Usage: python parse_data.py [input file name].pkl [output file name].pkl")
    parse_data(sys.argv[1], sys.argv[2])
