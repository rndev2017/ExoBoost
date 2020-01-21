import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

uni_grav_const = 6.67e-11 # Units: N * m^2 * kg^-2

n = 500

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


def calculate_phase(period):
    """Calculates phase based on period.

    Parameters
    ----------
    period : double
        Orbital period of observation (days).

    Returns phase
    -------
    list
        Orbital phase of the object orbiting the star.

    """
    t = np.linspace(0, n-1, num=n)

    return (t % period) / period


def rv(k, phase):
    """Calculates the radial velocity of the simulated host star.

    Parameters
    ----------
    k : double
        Radial Velocity Semi-Amplitude (m/s).
    phase : list
        Time of one orbital cycle.

    Returns rv
    -------
    list
        Radial velocities corresponding to each phase.

    """
    error = np.random.normal(0, 10, n)
    rv = k * np.sin(2 * np.pi * phase) + error

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
    data_merge = list(zip(time, rv_data))

    data = pd.DataFrame(data_merge, columns=["JD", "RV"])

    return data

def save_to_csv(path, dataframe):
    """Short summary.

    Parameters
    ----------
    path : type
        Description of parameter `path`.
    dataframe : type
        Description of parameter `dataframe`.

    Returns
    -------
    type
        Description of returned object.

    """
    dataframe.to_csv(path)



period_b = 452.8
period_c = 883.0

phase_b = calculate_phase(period_b)
phase_c = calculate_phase(period_c)


planet_b = calcualte_K(
    period = period_b * 365 * 24 * 60 * 60,
    star_mass = 1.989e30 * 1.11,
    planet_mass = 1.898e27 * 1.99,
    e = 0.09
)

planet_c = calcualte_K(
    period = period_c * 365 * 24 * 60 * 60,
    star_mass = 1.989e30 * 1.11,
    planet_mass = 1.898e27 * 0.86,
    e = 0.29
)

rv_t = rv(planet_b, phase_b) + rv(planet_c, phase_c)

data = create_dataframe(np.linspace(0, n-1, n), rv_t)

print(data)
save_to_csv(os.path.join(os.getcwd(), "data", "sim1.csv"), data)
