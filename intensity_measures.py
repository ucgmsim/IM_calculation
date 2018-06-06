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
    values = []

    if acceleration.ndim != 1:
        for i in range(3):
            value = get_spectral_acceleration(acceleration[:, i], period, NT, DT)
            values.append(value)
        value = values
    else:
        value = get_spectral_acceleration(acceleration, period, NT, DT)
    print("respectra values",value)
    return value


def get_cumulative_abs_velocity(acceleration, times):
    # CAV
    return np.trapz(np.abs(acceleration), times)


def get_cumulative_abs_velocity_nd(acceleration, times):
    return np.trapz(np.abs(acceleration), times, axis=0)


def get_arias_intensity(acceleration, g, times):
    # AI
    # I_{A}=\frac {\pi }{2g}\int _{0}^{T_{d}}a(t)^{2}dt on http://asciimath.org/
    # Where the acceleration units for a and g are the same. Below they are considered in cm/s
    acc_in_cms = acceleration * g
    integrand = acc_in_cms ** 2
    return np.pi / (2 * g) * np.trapz(integrand, times)


def get_arias_intensity_nd(acceleration, g, times):
    acc_in_cms = acceleration * g
    integrand = acc_in_cms ** 2
    return np.pi / (2 * g) * np.trapz(integrand, times, axis=0)


def calculate_MMI(velocities):
    # MMI
    pgv = get_max_nd(velocities)
    return np.float(timeseries.pgv2MMI(pgv))


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
    """Computes the percLow-percHigh% sign duration for a single ground motion component
    Based on getDs575.m
    Inputs:
        dt - the time step (s)
        fx - a vector (of acceleration)
        percLow - The lower percentage bound (default 5%)
        percHigh - The higher percentage bound (default 75%)
    Outputs:
        Ds - The duration (s)    """
    if accelerations.ndim == 1:
        return getDs(dt, accelerations, percLow=percLow, percHigh=percHigh)
    else:
        ds_values = []
        for fx in accelerations.transpose():
            ds = getDs(dt, fx, percLow, percHigh)
            ds_values.append(ds)
        return ds_values


def get_geom(d1, d2):
    return np.sqrt(d1 * d2)