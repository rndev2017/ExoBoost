from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive as query
import pandas as pd
import numpy as np
import os

exoplanet_archive_table = query.get_confirmed_planets_table()
star_names = exoplanet_archive_table["pl_hostname"]
num_of_planets = exoplanet_archive_table["pl_pnum"]

names_df = pd.Series(data=star_names)
pl_num = pd.Series(data=num_of_planets)
df = pd.DataFrame(columns=["host_name", "num_of_planets"])



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
