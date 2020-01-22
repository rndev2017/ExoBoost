import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import planet as pl
import star

uni_grav_const = 6.67e-11 # Units: N * m^2 * kg^-2


def create_length_of_observation(n):
    """Creates a certain length of n `observations`.

    Parameters
    ----------
    n : integer
        How many days you want to observe the simulated star.

    Returns
    -------
    list
        An n-dimensional list counting up to n.

    """
    return np.linspace(0, n-1, n)

def calcualte_K(period, star_mass, planet_mass, e = 0, i = math.pi/2):
    """Calcuate the radial velocity semiamplitude based on Wright, Jason T.

    Parameters
    ----------
    period : double
        Represents the orbital period of the planet. (seconds)
    star_mass : double
        Mass of the host star. (kilograms)
    planet_mass : double
        Mass of the planet. (kilograms)
    e : double
        Eccentricty of the orbit (default: e = 0).
    i : integer
        Inclination of the observation (default: i = 90) (degrees).

    Returns k
    -------
    double
        The Radial Velocity Semi-Amplitude of the star system (m/s).
    """
    return ((2 * math.pi * uni_grav_const)/(period * (star_mass**2)))**(1/3) \
      * ((planet_mass * math.sin(i))/(math.sqrt(1 - e**2)))


def calculate_phase(time, period):
    """Calculates phase based on period.

    Parameters
    ----------
    time : type
        Description of parameter `time`.
    period : type
        Description of parameter `period`.

    Returns
    -------
    list
        Orbital phase of the object orbiting the star.

    """

    return (time % period) / period


def create_sim_props(k, phase):
    """Create simulation properties for planet.

    Parameters
    ----------
    k : integer
        The Radial Velocity semiamplitude.
    phase : list
        Orbital phase of the object orbiting the star.

    Returns
    -------
    dictionary
        Dictionary containing RV Semi. and Phase.

    """
    return {"K": k, "Phase": phase}


def radvel(n_planets, sim_props, n):
    """Calculates the radial velocity of a star.

    Parameters
    ----------
    n_planets : integer
        The number of planets.
    sim_props : list
        A list of simulation properties for each planet in the system.

    Returns rv
    -------
    list
        list of radial velocities over time

    """
    error = np.random.normal(0, 10, n)

    if n_planets == 1:
        rv = sim_props["K"] * np.sin(2 * np.pi * sim_props["Phase"]) + error
        return rv

    else:
        rv = np.zeros(n)
        for i in range(n_planets):
            rv += sim_props[i]["K"] * np.sin(2 * np.pi * sim_props[i]["Phase"]) + error
        return rv


def create_dataframe(time, rv_data):
    """Creates a Pandas DataFrame to hold time series data.

    Parameters
    ----------
    time : list
        An array of `observation` points.
    rv_data : list
        An array of all radial velocities over set observation period.

    Returns data
    -------
    Pandas DataFrame
        A Pandas-based time series data frame

    """
    rv_error = [1 for i in range(len(time))]
    data_merge = list(zip(time, rv_data, rv_error))

    data = pd.DataFrame(data_merge, columns=["JD", "RV", "RV_ERROR"])

    return data


def save_to_csv(path, dataframe):
    """Saves dataframe to desired path.

    Parameters
    ----------
    path : string
        Path to the save directory.
    dataframe : Pandas DataFrame
        DataFrame containing time series data.
    """
    dataframe.to_csv(path)
