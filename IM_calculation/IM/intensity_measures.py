import numpy as np

from IM_calculation.IM.rspectra_calculations import rspectra as rspectra
from qcore import timeseries

DELTA_T = 0.005


def get_max_nd(data):
    return np.max(np.abs(data), axis=0)


def get_spectral_acceleration(acceleration, period, NT, DT, Nstep):
    # pSA
    c = 0.05
    M = 1.0
    beta = 0.25
    gamma = 0.5

    acc_step = np.zeros(NT + 1)
    acc_step[1:] = acceleration

    # interpolation additions
    t_orig = np.arange(NT + 1) * DT
    t_solve = np.arange(Nstep) * DELTA_T
    acc_step = np.interp(t_solve, t_orig, acc_step)
    return rspectra.Response_Spectra(acc_step, DELTA_T, c, period, M, gamma, beta)


def get_spectral_acceleration_nd(acceleration, period, NT, DT):
    # pSA
    if acceleration.ndim != 1:
        ts, dims = acceleration.shape
        values = np.zeros((period.size, dims))
        Nstep = calculate_Nstep(DT, NT)

        displacements = np.zeros((period.size, Nstep, dims))
        for i in range(dims):
            psa, u = get_spectral_acceleration(
                acceleration[:, i], period, NT, DT, Nstep
            )
            values[:, i] = psa
            displacements[:, :, i] = u

        return values, displacements
    else:
        return get_spectral_acceleration(acceleration, period, NT, DT)


def calculate_Nstep(DT, NT):
    return NT * int(round(DT / DELTA_T))


def get_rotations(
    spectral_displacements,
    func=lambda x: np.max(np.abs(x), axis=1),
    delta_theta: int = 1,
    min_angle: int = 0,
    max_angle: int = 180,
):
    """
    Calculates the rotd values for the given spectral displacements
    Works across multiple periods at once
    For each angle in the range [min_angle, max_angle) with step size delta theta,
    the displacement at each timestep is rotated to get the component in the direction of the given angle
    The absolute value is taken as it does not matter if the displacement is in the direction of the angle,
    or 180 degrees out of phase of it
    For each angle the maximum displacement is taken.
    :param spectral_displacements: An array with shape [periods.size, nt, 2] where nt is the number of timesteps in the original waveform
    :param func: The function to be applied to each waveform after rotations have been applied. Default takes the maximum of each angle for each period
    :param delta_theta: The difference between each angle to take. Defaults to 2 degrees
    :param min_angle: The minimum angle in degrees to calculate from, 0 is due East
    :param max_angle: The maximum angle in degrees to calculate to, 180 is due West. This value is not included in the calculations
    :return: An array of shape [periods.size, nt, (max_angle-min_angle)/delta_theta] containing rotd values
    """

    thetas = np.deg2rad(np.arange(min_angle, max_angle, delta_theta))
    rotation_matrices = np.asarray([np.cos(thetas), np.sin(thetas)])
    periods, nt, xy = spectral_displacements.shape
    rotds = np.zeros((periods, thetas.size))

    # Magic number empirically determined from runs on Maui
    step = int(np.floor(86000000 / (thetas.size * nt)))
    step = np.min([np.max([step, 1]), periods])

    for period in range(0, periods, step):
        rotds[period : period + step] = func(
            np.dot(spectral_displacements[period : period + step], rotation_matrices)
        )

    return rotds


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
        Ds - The duration (s)"""
    nsteps = np.size(fx)
    husid = np.zeros(nsteps)
    husid[0] = 0  # initialize first to 0
    for i in range(1, nsteps):
        husid[i] = husid[i - 1] + dt * (
            fx[i] ** 2
        )  # note that pi/(2g) is not used as doesnt affect the result
    AI = husid[-1]
    Ds = dt * (
        np.sum(husid / AI <= percHigh / 100.0) - np.sum(husid / AI <= percLow / 100.0)
    )
    return Ds


def getDs_nd(accelerations, dt, percLow=5, percHigh=75):
    """Computes the percLow-percHigh% sign duration for a nd(>1) ground motion component
    Based on getDs575.m
    Inputs:
        dt - the time step (s)
        fx - a vector (of acceleration)
        percLow - The lower percentage bound (default 5%)
        percHigh - The higher percentage bound (default 75%)
    Outputs:
        Ds - The duration (s)"""
    if accelerations.ndim == 1:
        return getDs(dt, accelerations, percLow, percHigh)
    else:
        values = np.zeros(
            accelerations.shape[-1]
        )  # Ds575 shouldn't return [1., 2., 0.] if only 2 columns are needed
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
