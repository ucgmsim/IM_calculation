from rspectra_calculations import rspectra as rspectra
import numpy as np
from qcore import timeseries


def get_max_nd(data):
    return np.max(np.abs(data), axis=0)


def get_spectral_acceleration(acceleration, period, NT, DT):
    # pSA
    deltat = 0.005
    c = 0.05
    M = 1.0
    beta = 0.25
    gamma = 0.5

    acc_step = np.zeros(NT + 1)
    acc_step[1:] = acceleration

    # interpolation additions
    Npts = np.shape(acceleration)[0]
    Nstep = np.floor(Npts * DT / deltat)
    t_orig = np.arange(Npts + 1) * DT
    t_solve = np.arange(Nstep + 1) * deltat
    acc_step = np.interp(t_solve, t_orig, acc_step)
    return rspectra.Response_Spectra(acc_step, deltat, c, period, M, gamma, beta)


def get_spectral_acceleration_nd(acceleration, period, NT, DT):
    # pSA
    if acceleration.ndim != 1:
        dims = acceleration.shape[1]
        values = np.zeros((period.size, dims))
        for i in range(dims):
            values[:, i] = get_spectral_acceleration(acceleration[:, i], period, NT, DT)
        return values
    else:
        return get_spectral_acceleration(acceleration, period, NT, DT)


def get_cumulative_abs_velocity_nd(acceleration, times):
    return np.trapz(np.abs(acceleration), times, axis=0)


def get_arias_intensity_nd(acceleration, g, times):
    acc_in_cms = acceleration * g
    integrand = acc_in_cms ** 2
    return np.pi / (2 * g) * np.trapz(integrand, times, axis=0)


def calculate_MMI_nd(velocities):
    pgv = get_max_nd(velocities)
    return timeseries.pgv2MMI(pgv)


def getDs(dt, fx, percLow=5, percHigh=75):
    """Computes the percLow-percHigh% sign duration for a single ground motion component
    Based on getDs575.m
    Inputs:
        dt - the time step (s)
        fx - a vector (of acceleration)
        percLow - The lower percentage bound (default 5%)
        percHigh - The higher percentage bound (default 75%)
    Outputs:
        Ds - The duration (s)    """
    nsteps = np.size(fx)
    husid = np.zeros(nsteps)
    husid[0] = 0  # initialize first to 0
    for i in xrange(1, nsteps):
        husid[i] = husid[i - 1] + dt * (fx[i] ** 2)  # note that pi/(2g) is not used as doesnt affect the result
    AI = husid[-1]
    Ds = dt * (np.sum(husid / AI <= percHigh / 100.) - np.sum(husid / AI <= percLow / 100.))
    return Ds


def getDs_nd(dt, accelerations, percLow=5, percHigh=75):
    """Computes the percLow-percHigh% sign duration for a nd(>1) ground motion component
    Based on getDs575.m
    Inputs:
        dt - the time step (s)
        fx - a vector (of acceleration)
        percLow - The lower percentage bound (default 5%)
        percHigh - The higher percentage bound (default 75%)
    Outputs:
        Ds - The duration (s)    """
    if accelerations.ndim == 1:
        return getDs(dt, accelerations, percLow, percHigh)
    else:
        values = np.zeros(3)
        i = 0
        for fx in accelerations.transpose():
            values[i] = getDs(dt, fx, percLow, percHigh)
            i += 1
        return values


def get_geom(d1, d2):
    """
    get geom value from the 090 and 000 components
    :param d1: 090
    :param d2: 000
    :return: geom value
    """
    return np.sqrt(d1 * d2)