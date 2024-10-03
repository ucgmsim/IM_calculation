import numba
import numpy as np
from qcore import timeseries

from IM_calculation.IM.Burks_Baker_2013_elastic_inelastic import Bilinear_Newmark_withTH
from IM_calculation.IM.rspectra_calculations import rspectra as rspectra

DELTA_T = 0.005
G = 981.0  # cm/s^2


def get_max_nd(data):
    return np.max(np.abs(data), axis=0)


@numba.njit
def response_spectra(
    acc: np.ndarray,
    dt: float,
    period: np.ndarray,
    xi: float = 0.05,
    m: float = 1.0,
    gamma: float = 0.5,
    beta=0.25,
) -> np.ndarray:
    w = 2 * np.pi / period
    c = 2 * xi * m * w
    k = m * w**2
    k1 = k + gamma * c / (beta * dt) + m / (beta * dt**2)
    a = m / (beta * dt) + gamma * c / beta
    b = 0.5 * m / beta + dt * (gamma * 0.5 / beta - 1) * c
    p = -m * acc
    u1 = 0
    u2 = 0
    u = np.zeros((period.size, acc.size))
    dp = np.diff(p)

    for i_T in range(acc.size - 1):
        dp = p[i_T + 1] - p[i_T]
        u1 = 0
        u2 = 0
        for i_s in range(period.size - 1):
            dp1 = dp + a * u1 + b * u2
            du: float = dp1 / k1
            du1 = (
                gamma * du / (beta * dt)
                - gamma * u1 / beta
                + dt * (1.0 - 0.5 * gamma / beta) * u2
            )
            du2 = du / (beta * dt**2) - u1 / (beta * dt) - 0.5 * u2 / beta
            u[i_T, i_s + 1] = u[i_T, i_s] + du
            u1 += du1
            u2 += du2

    return (u.T * np.square(w)).T


def get_spectral_acceleration(acceleration, period, NT, DT, Nstep, delta_t=DELTA_T):
    # pSA

    acc_step = np.zeros(NT + 1)
    acc_step[1:] = acceleration

    # interpolation additions
    t_orig = np.arange(NT + 1) * DT
    t_solve = np.arange(Nstep) * delta_t
    acc_step = np.interp(t_solve, t_orig, acc_step)

    xi = 0.05
    m = 1.0
    gamma = 0.5
    beta = 0.25
    return rspectra.Response_Spectra(acc_step, delta_t, xi, period, m, gamma, beta)


def get_spectral_acceleration_nd(acceleration, period, NT, DT):
    # pSA
    # Uses string conversion to convert from a float32 to a python float while retaining prescision
    # https://stackoverflow.com/questions/41967222/how-to-retain-precision-when-converting-from-float32-to-float
    delta_t = min(float(str(DT)), DELTA_T)

    if acceleration.ndim != 1:
        ts, dims = acceleration.shape
        Nstep = calculate_Nstep(DT, NT, delta_t)
        accelerations = np.zeros((period.size, Nstep, dims))

        for i in range(dims):
            accelerations[:, :, i] = get_spectral_acceleration(
                acceleration[:, i], period, NT, DT, Nstep, delta_t
            )

        return accelerations
    else:
        return get_spectral_acceleration(acceleration, period, NT, DT, delta_t)


def get_SDI(acceleration, period, DT, z, alpha, dy, dt):
    return Bilinear_Newmark_withTH(
        period,
        z,
        dy,
        alpha,
        acceleration * G / 100,
        DT,
        dt,  # Input is in m/s^2
    ).T


def get_SDI_nd(acceleration, period, DT, z, alpha, dy, dt):
    # SDI
    if acceleration.ndim != 1:
        ts, dims = acceleration.shape
        displacements = None

        for i in range(dims):
            res = get_SDI(acceleration[:, i], period, DT, z, alpha, dy, dt)
            if (
                displacements is None
            ):  # shape of displacements is determined after the first call
                displacements = np.zeros((*res.shape, dims))
            displacements[:, :, i] = res

        return displacements
    else:
        return get_SDI(acceleration, period, DT, z, alpha, dy, dt)


def calculate_Nstep(DT, NT, delta_t=DELTA_T):
    return NT * int(round(DT / delta_t))


@numba.njit
def get_rotations(
    accelerations,
    func=lambda x: np.abs(x, out=x).max(axis=-2),
    delta_theta: int = 1,
    min_angle: int = 0,
    max_angle: int = 180,
):
    """
    Calculates the rotd values for the given accelerations
    Works across multiple periods at once
    For each angle in the range [min_angle, max_angle) with step size delta theta,
    the acceleration at each timestep is rotated to get the component in the direction of the given angle
    The absolute value is taken as it does not matter if the acceleration is in the direction of the angle,
    or 180 degrees out of phase of it
    For each angle the maximum acceleration is taken.
    :param accelerations: An array with shape [periods.size, nt, 2] where nt is the number of timesteps in the original waveform
    :param func: The function to be applied to each waveform after rotations have been applied. Default takes the maximum of each angle for each period
    :param delta_theta: The difference between each angle to take. Defaults to 2 degrees
    :param min_angle: The minimum angle in degrees to calculate from, 0 is due East
    :param max_angle: The maximum angle in degrees to calculate to, 180 is due West. This value is not included in the calculations
    :return: An array of shape [periods.size, nt, (max_angle-min_angle)/delta_theta] containing rotd values
    """
    thetas = np.deg2rad(np.arange(min_angle, max_angle, delta_theta))
    rotation_matrices = np.asarray([np.cos(thetas), np.sin(thetas)])
    return func(np.dot(accelerations, rotation_matrices))


def get_cumulative_abs_velocity_nd(acceleration, times):
    return np.trapz(np.abs(acceleration), times, axis=0)


def get_arias_intensity_nd(acceleration, times):
    acc_in_cms = acceleration * G
    integrand = acc_in_cms**2
    return np.pi / (2 * G) * np.trapz(integrand, times, axis=0)


def get_specific_energy_density_nd(velocity, times):
    integrand = velocity**2
    return np.trapz(integrand, times, axis=0)


def calculate_MMI_nd(velocities):
    pgv = get_max_nd(velocities)
    return timeseries.pgv2MMI(pgv)


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
    arias_intensity = np.cumsum(np.square(accelerations), axis=0)
    arias_intensity /= arias_intensity[:, -1][:, np.newaxis]
    return np.apply_along_axis(
        lambda component: dt
        * (np.diff(np.searchsorted(component, [percLow / 100, percHigh / 100])) + 1),
        0,
        arias_intensity,
    ).ravel()


def get_geom(d1, d2):
    """
    get geom value from the 090 and 000 components
    :param d1: 090
    :param d2: 000
    :return: geom value
    """
    return np.sqrt(d1 * d2)


def get_euclidean_dist(d1, d2):
    """
    get euclidean distance from the 090 and 000 components
    :param d1: 090
    :param d2: 000
    :return: euclidean distance
    ----------
    """
    return np.sqrt(d1**2 + d2**2)
