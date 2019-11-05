import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit, fsolve
from sklearn.preprocessing import MinMaxScaler
import name_extract as name
import math
import os
# inspiration from https://www.astro.princeton.edu/~jgreene/ast303/HW07_code/exoplanet.py


class Star():

    def __init__(self, file_path, star_name):
        self.file_path = file_path
        self.star_name = star_name

        # extracting csv data from file path
        self.data = pd.read_csv(self.file_path)
        self.t = self.data["JD"].values
        self.rv = self.data["RV"].values
        self.rv_err = self.data["RV_ERROR"].values

        # initializes period as none; will be initialized after
        self.period = None


    def fill_na(self, column_name):
        self.data[column_name] = self.data.fillna(
            self.data[column_name].mean())


    def solve_kepler(self, M, e):
        eanom = np.zeros(M.shape)
        for i, mi in enumerate(M):
            # solves the Kepler's equation
            tmp, = fsolve(lambda E: E-e*np.sin(E)-mi, mi)
            eanom[i] = tmp
        return eanom


    def plot_rv(self, t, rv, err, fmt):
        title = "Radial Velocity Curve of {}".format(self.star_name)
        plt.figure(title)
        plt.title(title)
        plt.xlabel("JD [Julian Dates]")
        plt.ylabel("Radial Velocity [m/s]")
        plt.errorbar(x=t, y=rv, fmt=fmt, yerr=err)
        plt.savefig("path\\to\\some\\directory".format(self.star_name))
        plt.close()


    def keplerian_fit(self, t, K, P, e, w, tau, vr0):
        e_anomaly = self.solve_kepler((t-tau)*2*np.pi/P, e)
        theta = 2*np.arctan2(np.sqrt(1.+e)*np.sin(0.5*e_anomaly),
                             np.sqrt(1.-e)*np.cos(0.5*e_anomaly))

        return K*(np.cos(theta+w)+e*np.cos(w))+vr0


    def compute_periodiogram(self, t, rv, err):
        frequency = np.linspace(0.001, 1, 100000)
        power = LombScargle(t, rv, err).power(frequency)
        frequency, power = LombScargle(t, rv, err).autopower()

        return frequency, power


    def plot_periodogram(self, freq, power, fmt):
        title = "Periodiogram plot"
        plt.figure(title)
        plt.title(title)
        plt.xlabel("Period [Days]")
        plt.ylabel("Power")
        plt.semilogx(1./freq, power, fmt)
        plt.savefig(
            "path\\to\\some\\directory".format(self.star_name))
        plt.close()


    def compute_period(self, freq, power):
        self.period = 1./freq[np.argmax(power)]

        return self.period


    def create_fit(self, period, freq, power, t, rv, err):
        time_fit = np.linspace(0, period, 1000)
        phase = (t % period) % period
        rad_fit = LombScargle(t, rv, err).model(time_fit, 1/period)

        semi_amplitude = 0.5*(np.max(rad_fit)-np.min(rad_fit))
        voffset = np.mean(rad_fit)

        return phase, semi_amplitude, voffset


    def create_initial_params(self, k, p, e, w, tau, vr0):
        return (k, p, e, w, tau, vr0)


    def radial_vel_fit(self, t, rv, rv_err, initial_params):
        params = curve_fit(self.keplerian_fit, t, rv,
                           sigma=rv_err, absolute_sigma=True,
                           p0=initial_params)

        k, p, e, w, tau, vr0 = params[0]
        return self.star_name, k, p, e, w, tau, vr0


    def plot_rvc_fit(self, phase, rv, err, params):
        self.star_name, k, p, e, w, tau, vr0 = params

        if e < 0:
            w -= np.pi
            e *= -1

        if k < 0:
            k *= -1
            w += np.pi

        tfit = np.linspace(0, p, 1000)
        rvfit = self.keplerian_fit(tfit, k, p, e, w, tau, vr0)

        plt.xlabel('Orbital Phase [days]')
        plt.ylabel("Radial Velocity [m/s]")
        plt.errorbar(phase, rv, err, fmt='.k')
        plt.plot(tfit, rvfit, 'r--')
        plt.savefig(
            "path\\to\\some\\directory".format(self.star_name))
        plt.close()


    def get_t(self):
        return self.t


    def get_rv(self):
        return self.rv


    def get_rv_err(self):
        return self.rv_err
