import os
import pandas as pd
import numpy as np
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive as query


# retrieves data from the archive through astroquery
exoplanet_archive_table = query.get_confirmed_planets_table()

# pl_hostname refers to the the planet's star host names
# pl_pnum refers to the number of planets around the host
star_names = exoplanet_archive_table["pl_hostname"]
num_of_planets = exoplanet_archive_table["pl_pnum"]


def create_new_df(names, num_of_planets):
    """Creates a new pandas DataFrame

    Arguments:
        names {astropy.table.column.MaskedColumn} -- a column of host star names
        num_of_planets {astropy.table.column.MaskedColumn} -- a column of the
            number of planets for each star in the names column

    Returns:
        Pandas DataFrame -- consisting of the star names and number of planets
            as two columns
    """
    names_df = pd.Series(data=star_names)
    pl_num = pd.Series(data=num_of_planets)
    return pd.DataFrame(columns=["host_name", "num_of_planets"])



df["host_name"] = names_df
df["num_of_planets"] = pl_num
df.to_csv("\\path\\to\\save\\new\\file")



params_df = pd.read_csv("\\path\\to\\data")


num_planets_series = []
for star_name in params_df["Name"]:
    host_look_up = star_name[:-4]
    num_of_planets = df["num_of_planets"][df["host_name"]==host_look_up].values
    print("Appending: {}".format(star_name))
    num_planets_series.append(num_of_planets)


params_df["num_of_planets"] = pd.Series(data=num_planets_series)

params_df.to_csv("\\path\\to\\save\\new\\file.csv")
