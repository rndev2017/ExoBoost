import os
import pandas as pd
import numpy as np
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive as query

# reads a data file (.csv) from a designated file path
params_df = pd.read_csv("\\path\\to\\data")

# retrieves data from the archive through astroquery
exoplanet_archive_table = query.get_confirmed_planets_table()

# pl_hostname refers to the the planet's star host names
# pl_pnum refers to the number of planets around the host
star_names = exoplanet_archive_table["pl_hostname"]
num_of_planets = exoplanet_archive_table["pl_pnum"]


def create_new_df(names, num_of_planets):
    """Creates a new pandas DataFrame that will be used to add to an existing
       data set

    Arguments:
        names {astropy.table.column.MaskedColumn} -- a column of host star names
        num_of_planets {astropy.table.column.MaskedColumn} -- a column of the
            number of planets for each star in the names column

    Returns:
        pandas.DataFrame -- consisting of the star names and number of planets
            as two columns
    """
    names_df = pd.Series(data=star_names)
    pl_num = pd.Series(data=num_of_planets)
    return pd.DataFrame({"host_name": names_df, "num_of_planets": num_of_planets}).drop_duplicates()


def save_df(data_frame, file_path):
    """Saves the DataFrame to a specified path

        Arguments:
            data_frame {pandas.DataFrame} -- DataFrame
            file_path {str} -- specifies the data file's desired location to be
                saved

        Returns:
            None -- File Saved

    """
    data_frame.to_csv(file_path)
    return None

def extract_unique_values(existing_df, new_df):
    """Extract unique values (i.e number of planets)
        from an external data frame

        Arguments:
            existing_df {pandas.DataFrame} -- old DataFrame that will be used to
                compared to newer DataFrame
            new_df {pandas.DataFrame} -- new DataFrame that was extracted
                through the astroquery's call to NasaExoplanetArchive

        Returns:
            None -- unique values added to existing DataFrame
    """
    num_planets_series = []
    for star_name in existing_df["Name"]:
        host_look_up = star_name[:-4]
        num_of_planets = new_df["num_of_planets"][new_df["host_name"]== \
            host_look_up].values
        num_planets_series.append(num_of_planets)

    params_df["num_of_planets"] = pd.Series(data=num_planets_series)
