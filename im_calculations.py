from Cython import rspectra as rspectra
import numpy as np
from qcore import timeseries

# pSA
DELTA = 0.005
C = 0.05
M = 1.0
BETA = 0.25
GAMMA = 0.5
EXT_PERIOD = np.logspace(start=np.log10(0.01), stop=np.log10(10.), num=100, base=10)
BSC_PERIOD = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0])
DIMS_AXIS = {1:0, 2:0}


def get_max(data):
    # PGV / PGA
    return np.max(np.abs(data))


def get_max_nd(data):
    return np.max(np.abs(data), axis=0)


def get_spectral_acceleration(acceleration, period, NT, DT):
    # pSA
    deltat = 0.005
    c = 0.05
    M = 1.0
    beta = 0.25
    gamma = 0.5
    # extended_period = np.logspace(start=np.log10(0.01), stop=np.log10(10.), num=100, base=10)
    # basic_period = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]

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
    pgv = get_max(velocities)
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
    ds_values = []
    for fx in accelerations.transpose():
        nsteps = np.size(fx)
        husid = np.zeros(fx.shape)
        for i in xrange(1, nsteps):
            husid[i] = husid[i - 1] + dt * (fx[i] ** 2)  # note that pi/(2g) is not used as doesnt affect the result
        AI = husid[-1]
        ds = dt * (np.sum(husid / AI <= percHigh / 100., axis=0) - np.sum(husid / AI <= percLow / 100.))
        ds_values.append(ds)
    return ds_values


# def getDs(dt, fx, percLow=5, percHigh=75):
#     results, cols = get_cols(fx)
#     print(results, cols)
#
#     for col in range(cols):
#         print(col)
#         fx_col = fx[:, col]
#         nsteps = np.size(fx_col)
#         print(nsteps)
#         husid = np.zeros(nsteps)
#         husid[0] = 0  # initialize first to 0
#         for i in xrange(1, nsteps):
#             husid[i] = husid[i - 1] + dt * (fx_col[i] ** 2)  # note that pi/(2g) is not used as doesnt affect the result
#         AI = husid[-1]
#         Ds = dt * (np.sum(husid / AI <= percHigh / 100.) - np.sum(husid / AI <= percLow / 100.))
#
#         results = get_result(results, Ds)
#
#     return results
#
#
# def get_cols(fx):
#     print(fx.shape)
#     try:
#         cols = fx.shape[1]
#         results = []
#     except IndexError:
#         cols = 1
#         results = None
#     return results, cols
#
#
# def get_result(results, result):
#     try:
#         results.append(result)
#     except AttributeError:
#         print("only one col in fx")
#         return result
#     return results