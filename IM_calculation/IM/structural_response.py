#!/usr/bin/env python

import math

import numpy as np


def load():
    # temporary to load time series
    station_data = [
        "/home/vap30/Downloads/Jae/20191201 CHCH Feb ground motion input/Observed_Records/ADCS_000_obs.m",
        "/home/vap30/Downloads/Jae/20191201 CHCH Feb ground motion input/Simulated_Records/ADCS_000_sim.m",
    ]
    obssim = []
    for f in station_data:
        with open(f, "r") as g:
            g.readline()
            g.readline()
            obssim.append(
                np.array(
                    list(
                        map(
                            float,
                            " ".join(list(map(str.rstrip, g.readlines()))).split(),
                        )
                    )
                )
            )
    return obssim


def sa_sd_time(
    acc, dt, t1_range=None, t1min=1e-06, t1max=5, nt=100, c=0.05, g=9.81, m=1
):
    # t1_range: give values instead of calculating them from min, max, nt
    # t1min: minimum time period to be considered in the design acceleration spectra (seconds).
    #        0 not chosen to avoid placing 0 in the denominator in later calculations.
    # t1max: maximum time period to be considered in the design acceleration spectra (seconds).
    # nt: number of points to be considered in the design acceleration spectra.
    # c: damping  (c=0.05 represents 5% damping).
    # g: 9.81 converts acceleration to ms-2;  (if g=1, unit will be in g because the unit of the acceleration record input is in g).
    # m: 1 for fixed mass values (stiffness varied)

    # period range of design acceleration spectra
    if t1_range is None:
        t1_range = np.linspace(t1min, t1max, nt)
    else:
        # allow single values and python lists
        t1_range = np.atleast_1d(np.array(t1_range))
    sa = np.zeros(acc.size, dtype=np.float64)
    sd = np.zeros_like(sa)
    # psd = np.zeros(len(t1_range), dtype=np.float64)
    sa_time = np.empty((len(t1_range), acc.size))
    sd_time = np.empty_like(sa_time)

    # newmark constant acceleration
    kt = m * (2 * math.pi / t1_range) ** 2
    omegat = np.sqrt(kt / m)
    ct = 2 * c * omegat * m
    deltaaccg = np.diff(acc) * g

    for i in range(len(t1_range)):
        # spectral acceleration terms use Newmark beta (Chopra, p177)
        u0 = 0.0
        du0 = 0.0
        d2u0 = 0.0
        for j in range(1, acc.size):
            deltad2u = (
                -deltaaccg[j - 1] * m
                - dt * ct[i] * d2u0
                - dt * kt[i] * (du0 + dt / 2.0 * d2u0)
            ) / (m + dt / 2.0 * ct[i] + dt ** 2 / 4.0 * kt[i])
            deltadu = dt * d2u0 + dt / 2 * deltad2u
            deltau = dt * du0 + dt ** 2 / 2.0 * d2u0 + dt ** 2 / 4.0 * deltad2u

            d2u0 += deltad2u
            du0 += deltadu
            u0 += deltau

            sa[j] = d2u0
            sd[j] = u0
        # psd(i) = max(abs(sd))
        sa_time[i] = sa
        sd_time[i] = sd

    # in m/s2
    # psa = ((2 * pi) / t1_range) ** 2 * psd

    return sa_time, sd_time


def eigenvalue_test(alpha, gamma_approximation):
    # this code checks if the gamma values for the vibration periods have been correctly estimated

    squared_term = alpha ** 2 + gamma_approximation ** 2

    # f(x)
    eigenvalue = (
        2.0
        + (2.0 + alpha ** 4 / (gamma_approximation ** 2 * squared_term))
        * math.cos(gamma_approximation)
        * math.cosh(math.sqrt(squared_term))
        + (alpha ** 2 / (gamma_approximation * math.sqrt(squared_term)))
        * math.sin(gamma_approximation)
        * math.sinh(math.sqrt(squared_term))
    )

    # f'(x)
    term_1 = (
        -2.0 * math.sin(gamma_approximation) * math.cosh(math.sqrt(squared_term))
        + 2
        * math.cos(gamma_approximation)
        * math.sinh(math.sqrt(squared_term))
        * (squared_term ** -0.5)
        * gamma_approximation
    )
    term_2 = (
        (
            (
                -(alpha ** 4)
                * math.sin(gamma_approximation)
                * math.cosh(math.sqrt(squared_term))
                + (alpha ** 4)
                * math.cos(gamma_approximation)
                * math.sinh(math.sqrt(squared_term))
                * (squared_term ** -0.5)
                * gamma_approximation
            )
            * (gamma_approximation ** 2)
            * (squared_term)
        )
        - (
            (2.0 * gamma_approximation * squared_term)
            + 2.0 * (gamma_approximation ** 3)
        )
        * (alpha ** 4)
        * math.cos(gamma_approximation)
        * math.cosh(math.sqrt(squared_term))
    ) / ((gamma_approximation ** 2) * (squared_term)) ** 2
    term_3 = (
        (
            (
                (alpha ** 2)
                * math.cos(gamma_approximation)
                * math.sinh(math.sqrt(squared_term))
                + (alpha ** 2)
                * math.sin(gamma_approximation)
                * math.cosh(math.sqrt(squared_term))
                * (squared_term ** -0.5)
                * gamma_approximation
            )
            * gamma_approximation
            * math.sqrt(squared_term)
        )
        - (
            (
                math.sqrt(squared_term)
                + (gamma_approximation ** 2) * (squared_term ** -0.5)
            )
            * (alpha ** 2)
            * math.sin(gamma_approximation)
            * math.sinh(math.sqrt(squared_term))
        )
    ) / ((gamma_approximation ** 2) * squared_term)
    eigenvalue_prime = term_1 + term_2 + term_3

    return eigenvalue, eigenvalue_prime


def calculate_vibration_periods(alpha, t1):
    """
    more accurate estimation of vibration periods appears to be achieved when alpha is limited to 25
    """

    c = 0
    increment = 0.01
    if alpha >= 25:
        gamma_max = 35
        tolerance = 2.5
    else:
        gamma_max = 25
        tolerance = 0.01

    gamma_incremental_approximation = np.empty(round(gamma_max / increment))
    gamma = np.zeros(round(gamma_max / increment) - 1)
    # this is horrible, dual loop reused arrays with dependencies
    eigenvalue = np.zeros(max(round(gamma_max / increment), 100))
    eigenvalue_prime = np.zeros_like(eigenvalue)
    gamma_NR_approximation = np.empty(101)

    for i in range(round(gamma_max / increment)):
        # function script that calls out Equation 24 of Miranda 2005
        eigenvalue[i], eigenvalue_prime[i] = eigenvalue_test(alpha, increment * (i + 1))

        # find the approximate gamma which yields eigenvalue = 0
        if i > 0:
            # positive slope
            if eigenvalue[i] > 0:
                if eigenvalue[i - 1] < 0:
                    gamma_initial_NR_approximation = (
                        increment * i + increment * (i + 1)
                    ) / 2.0
                    gamma_NR_approximation[0] = gamma_initial_NR_approximation
                    for j in range(100):
                        eigenvalue[j], eigenvalue_prime[j] = eigenvalue_test(
                            alpha, gamma_NR_approximation[j]
                        )
                        # no need to calculate at j + 1 == 100
                        gamma_NR_approximation[j + 1] = gamma_NR_approximation[j] - (
                            eigenvalue[j] / eigenvalue_prime[j]
                        )

                        # not sure if necessary, matlab called clear, also below
                        eigenvalue[i - 1] = 0
                        if j > 0:
                            if abs(eigenvalue[j]) < tolerance:
                                if (
                                    abs(
                                        abs(gamma_NR_approximation[j])
                                        - abs(gamma_initial_NR_approximation)
                                    )
                                    < tolerance
                                ):
                                    gamma[c] = gamma_NR_approximation[j]
                                    c += 1
                                    break

            # negative slope
            elif eigenvalue[i] < 0:
                if eigenvalue[i - 1] > 0:
                    gamma_initial_NR_approximation = (
                        increment * i + increment * (i + 1)
                    ) / 2.0
                    gamma_NR_approximation[0] = gamma_initial_NR_approximation
                    for j in range(100):
                        eigenvalue[j], eigenvalue_prime[j] = eigenvalue_test(
                            alpha, gamma_NR_approximation[j]
                        )
                        gamma_NR_approximation[j + 1] = gamma_NR_approximation[j] - (
                            eigenvalue[j] / eigenvalue_prime[j]
                        )

                        eigenvalue[i - 1] = 0
                        if j > 0:
                            if abs(eigenvalue[j]) < tolerance:
                                if (
                                    abs(
                                        abs(gamma_NR_approximation[j])
                                        - abs(gamma_initial_NR_approximation)
                                    )
                                    < tolerance
                                ):
                                    gamma[c] = gamma_NR_approximation[j]
                                    c += 1
                                    break

    gamma = gamma[: c]
    for i in range(len(gamma)):
        if abs(eigenvalue_test(alpha, gamma[i])[0]) > tolerance:
            print("eigenvalue approximation incorrect at", i, "th vibration period")

    # vibration periods of the structural system
    vibration_period = np.empty(len(gamma))
    vibration_period[0] = t1
    for i in range(1, vibration_period.size):
        vibration_period[i] = (
            t1
            * (gamma[0] / gamma[i])
            * np.sqrt((gamma[0] ** 2 + alpha ** 2) / (gamma[i] ** 2 + alpha ** 2))
        )

    return gamma, vibration_period


def calculate_mode_shapes(alpha, gamma, storey):

    # fixed mass values (stiffness varied)
    rho = 1

    storey_height = np.arange(storey + 1, dtype=np.float64) / storey
    beta = np.sqrt(alpha ** 2 + gamma ** 2)
    eta = ((gamma ** 2) * np.sin(gamma) + gamma * beta * np.sinh(beta)) / (
        gamma ** 2 * np.cos(gamma) + (beta ** 2) * np.cosh(beta)
    )
    denominator_1 = (
        np.sin(gamma)
        - (gamma / beta) * np.sinh(beta)
        + eta * (np.cosh(beta) - np.cos(gamma))
    )
    phi = np.zeros((storey + 1, len(gamma)))
    phi_1 = np.zeros_like(phi)
    phi_2 = np.zeros_like(phi)
    phi_3 = np.zeros_like(phi)
    phi_4 = np.zeros_like(phi)
    participation_factor = np.zeros_like(gamma)
    # doesn't really need to be an array
    effective_mass = np.zeros_like(gamma)
    participating_mass_ratio = 0.0

    for i in range(len(gamma)):
        for j in range(storey + 1):
            # normalised mode shape, with respect to phi at storey_height = 1
            phi[j, i] = (
                math.sin(gamma[i] * storey_height[j])
                - (gamma[i] / beta[i]) * math.sinh(storey_height[j] * beta[i])
                + eta[i]
                * (
                    math.cosh(storey_height[j] * beta[i])
                    - math.cos(gamma[i] * storey_height[j])
                )
            ) / denominator_1[i]
            phi_1[j, i] = (
                gamma[i] * math.cos(gamma[i] * storey_height[j])
                - gamma[i] * math.cosh(storey_height[j] * beta[i])
                + eta[i]
                * (
                    beta[i] * math.sinh(storey_height[j] * beta[i])
                    + gamma[i] * math.sin(gamma[i] * storey_height[j])
                )
            ) / denominator_1[i]
            phi_2[j, i] = (
                -(gamma[i] ** 2) * math.sin(gamma[i] * storey_height[j])
                - gamma[i] * beta[i] * math.sinh(storey_height[j] * beta[i])
                + eta[i]
                * (
                    (beta[i] ** 2) * math.cosh(storey_height[j] * beta[i])
                    + (gamma[i] ** 2) * math.cos(gamma[i] * storey_height[j])
                )
            ) / denominator_1[i]
            phi_3[j, i] = (
                -(gamma[i] ** 3) * math.cos(gamma[i] * storey_height[j])
                - gamma[i] * (beta[i] ** 2) * math.cosh(storey_height[j] * beta[i])
                + eta[i]
                * (
                    (beta[i] ** 3) * math.sinh(storey_height[j] * beta[i])
                    - (gamma[i] ** 3) * math.sin(gamma[i] * storey_height[j])
                )
            ) / denominator_1[i]
            phi_4[j, i] = (
                (gamma[i] ** 4) * math.sin(gamma[i] * storey_height[j])
                - gamma[i] * (beta[i] ** 3) * math.sinh(storey_height[j] * beta[i])
                + eta[i]
                * (
                    (beta[i] ** 4) * math.cosh(storey_height[j] * beta[i])
                    - (gamma[i] ** 4) * math.cos(gamma[i] * storey_height[j])
                )
            ) / denominator_1[i]

        # modal participation factor
        participation_factor[i] = (
            np.dot(
                np.dot(phi[1 : storey + 1, i], (rho * np.eye(storey))), np.ones(storey)
            )
        ) / np.dot(
            np.dot(phi[1 : storey + 1, i], (rho * np.eye(storey))),
            phi[1 : storey + 1, i],
        )
        effective_mass[i] = (participation_factor[i] ** 2) * np.dot(
            np.dot(phi[1 : storey + 1, i], (rho * np.eye(storey))),
            phi[1 : storey + 1, i],
        )
        participating_mass_ratio += effective_mass[i] / (rho * storey)

    # check if 90% or more participating mass has been considered
    if participating_mass_ratio < 0.9:
        # not sure what the point of this is, should it assert?
        print("check participating_mass > 0.9")

    return (phi, phi_1, phi_2, phi_3, phi_4, participation_factor)


def calculate_structural_response_B(
    participation_factor,
    phi,
    phi_1,
    phi_2,
    phi_3,
    phi_4,
    sa_time,
    sd_time,
    alpha,
    storey,
):
    modal_disp = np.zeros(phi.shape)
    modal_slope = np.zeros_like(modal_disp)
    modal_moment = np.zeros_like(modal_disp)
    modal_shear = np.zeros_like(modal_disp)
    modal_load = np.zeros_like(modal_disp)
    modal_acc = np.zeros_like(modal_disp)

    disp_time_history_per_EQ = np.zeros((phi.shape[0], sd_time.shape[1]))
    moment_time_history_per_EQ = np.zeros_like(disp_time_history_per_EQ)
    shear_time_history_per_EQ = np.zeros_like(disp_time_history_per_EQ)
    load_time_history_per_EQ = np.zeros_like(disp_time_history_per_EQ)
    rel_accel_time_history_per_EQ = np.zeros_like(disp_time_history_per_EQ)
    slope_time_history_per_EQ = np.zeros_like(disp_time_history_per_EQ)

    for j in range(sd_time.shape[1]):
        modal_disp = participation_factor * phi * sd_time[:, j]
        modal_slope = participation_factor * phi_1 * sd_time[:, j]
        modal_moment = participation_factor * phi_2 * sd_time[:, j]
        modal_shear = (
            participation_factor * phi_3 * sd_time[:, j]
            - participation_factor * alpha ** 2 * phi_1 * sd_time[:, j]
        )
        modal_load = (
            participation_factor * phi_4 * sd_time[:, j]
            - participation_factor * alpha ** 2 * phi_2 * sd_time[:, j]
        )

        modal_acc = participation_factor * phi * sa_time[:, j]

        # system response at each time step (response from each vibration step combined)
        disp_time_history_per_EQ[:, j] = np.sum(modal_disp, axis=1)
        moment_time_history_per_EQ[:, j] = np.sum(modal_moment, axis=1)
        shear_time_history_per_EQ[:, j] = np.sum(modal_shear, axis=1)
        load_time_history_per_EQ[:, j] = np.sum(modal_load, axis=1)
        rel_accel_time_history_per_EQ[:, j] = np.sum(modal_acc, axis=1)

    # calculate interstorey drift from displacement
    for k in range(1, storey + 1):
        slope_time_history_per_EQ[k, :] = (
            disp_time_history_per_EQ[k, :] - disp_time_history_per_EQ[k - 1, :]
        ) / (1.0 / storey)

    return (
        disp_time_history_per_EQ,
        slope_time_history_per_EQ,
        moment_time_history_per_EQ,
        shear_time_history_per_EQ,
        load_time_history_per_EQ,
        rel_accel_time_history_per_EQ,
    )


def calculate_structural_response(
    gamma,
    vibration_period,
    acc_time_history,
    dt,
    storey,
    participation_factor,
    phi,
    phi_1,
    phi_2,
    phi_3,
    phi_4,
    alpha,
    c=0.05,
    g=9.81,
):

    # calculate spectral acceleration and spectral displacements
    sa_time, sd_time = sa_sd_time(
        acc_time_history, dt, t1_range=vibration_period, c=c, g=g
    )

    # calculate structural response
    disp_time_history_per_EQ, slope_time_history_per_EQ, moment_time_history_per_EQ, shear_time_history_per_EQ, load_time_history_per_EQ, rel_accel_time_history_per_EQ = calculate_structural_response_B(
        participation_factor,
        phi,
        phi_1,
        phi_2,
        phi_3,
        phi_4,
        sa_time,
        sd_time,
        alpha,
        storey,
    )

    # creates matrices for storey shear and overturning moment
    storey_shear_time_history_per_EQ = np.zeros_like(shear_time_history_per_EQ)
    storey_moment_time_history_per_EQ = np.zeros_like(moment_time_history_per_EQ)
    for k in range(storey, -1, -1):
        if k == storey:
            storey_shear_time_history_per_EQ[k, :] = shear_time_history_per_EQ[k, :]
            storey_moment_time_history_per_EQ[k, :] = moment_time_history_per_EQ[k, :]
        else:
            storey_shear_time_history_per_EQ[k, :] = (
                storey_shear_time_history_per_EQ[k + 1, :]
                + shear_time_history_per_EQ[k, :]
            )
            storey_moment_time_history_per_EQ[k, :] = (
                storey_moment_time_history_per_EQ[k + 1, :]
                + moment_time_history_per_EQ[k, :]
            )

    # calculates total acceleration (floor acceleration) matrix
    ground_accel_time_history_per_EQ = np.empty((storey + 1, len(acc_time_history)))
    for k in range(storey + 1):
        ground_accel_time_history_per_EQ[k, :] = g * acc_time_history

    total_accel_time_history_per_EQ = (
        ground_accel_time_history_per_EQ + rel_accel_time_history_per_EQ
    )

    return (
        disp_time_history_per_EQ,
        slope_time_history_per_EQ,
        moment_time_history_per_EQ,
        storey_moment_time_history_per_EQ,
        shear_time_history_per_EQ,
        storey_shear_time_history_per_EQ,
        load_time_history_per_EQ,
        ground_accel_time_history_per_EQ,
        rel_accel_time_history_per_EQ,
        total_accel_time_history_per_EQ,
    )


def extract_peak_structural_response(
    storey,
    disp_time_history_EQ_matrix,
    slope_time_history_EQ_matrix,
    moment_time_history_EQ_matrix,
    storey_moment_time_history_EQ_matrix,
    shear_time_history_EQ_matrix,
    storey_shear_time_history_EQ_matrix,
    load_time_history_EQ_matrix,
    ground_accel_time_history_EQ_matrix,
    rel_accel_time_history_EQ_matrix,
    total_accel_time_history_EQ_matrix,
):
    # extracts the peak response structural response from the time history response matrices

    disp_time_history_peak = np.max(abs(disp_time_history_EQ_matrix), axis=1)
    slope_time_history_peak = np.max(abs(slope_time_history_EQ_matrix), axis=1)
    moment_time_history_peak = np.max(abs(moment_time_history_EQ_matrix), axis=1)
    storey_moment_time_history_peak = np.max(
        abs(storey_moment_time_history_EQ_matrix), axis=1
    )
    shear_time_history_peak = np.max(abs(shear_time_history_EQ_matrix), axis=1)
    storey_shear_time_history_peak = np.max(
        abs(storey_shear_time_history_EQ_matrix), axis=1
    )
    load_time_history_peak = np.max(abs(load_time_history_EQ_matrix), axis=1)
    ground_accel_time_history_peak = np.max(
        abs(ground_accel_time_history_EQ_matrix), axis=1
    )
    rel_accel_time_history_peak = np.max(abs(rel_accel_time_history_EQ_matrix), axis=1)
    total_accel_time_history_peak = np.max(
        abs(total_accel_time_history_EQ_matrix), axis=1
    )

    # shear is effectively zero at top, set it to zero
    shear_time_history_peak[-1] = 0.0
    storey_shear_time_history_peak[-1] = 0.0
    # moment is effectively zero at top, set it to zero
    moment_time_history_peak[-1] = 0.0
    storey_moment_time_history_peak[-1] = 0.0

    return (
        disp_time_history_peak,
        slope_time_history_peak,
        moment_time_history_peak,
        storey_moment_time_history_peak,
        shear_time_history_peak,
        storey_shear_time_history_peak,
        load_time_history_peak,
        ground_accel_time_history_peak,
        rel_accel_time_history_peak,
        total_accel_time_history_peak,
    )


# read timeseries
dt_obs = 0.005
dt_sim = 0.005
acc_obs, acc_sim = load()

# largest translational period of the structure.
t1 = 2.0
# non-dimensional flexure-shear coefficient of the structure.  alpha=0 represents shear wall buildings.  alpha=30 represents moment frame buildings.
alpha = 30
# represents the number of equally spaced heights along the structure from which the analysis outputs will be recorded  (height of the structure is non-dimensional; 0 at base, 1 at top).
storey = 10
c=0.1

# 3a calculate the vibration periods of the structure
gamma, vibration_period = calculate_vibration_periods(alpha, t1)
# 3b calculate the mode shapes of the structure
phi, phi_1, phi_2, phi_3, phi_4, participation_factor = calculate_mode_shapes(
    alpha, gamma, storey
)
print("VIBRATION PERIOD")
print(",".join(map(str, vibration_period)))
print("PARTICIPATION FACTOR")
print(",".join(map(str, participation_factor)))
varsp = {"PHI":phi}
for var in varsp:
    print(var)
    for i in range(phi.shape[0]):
        print(
            ",".join(map(str, varsp[var][i]))
        )
# 3c calculate structural response
disp_time_history_EQ_matrix_obs, slope_time_history_EQ_matrix_obs, moment_time_history_EQ_matrix_obs, storey_moment_time_history_EQ_matrix_obs, shear_time_history_EQ_matrix_obs, storey_shear_time_history_EQ_matrix_obs, load_time_history_EQ_matrix_obs, ground_accel_time_history_EQ_matrix_obs, rel_accel_time_history_EQ_matrix_obs, total_accel_time_history_EQ_matrix_obs = calculate_structural_response(
    gamma,
    vibration_period,
    acc_obs,
    dt_obs,
    storey,
    participation_factor,
    phi,
    phi_1,
    phi_2,
    phi_3,
    phi_4,
    alpha,
    c=c,
)
disp_time_history_EQ_matrix_sim, slope_time_history_EQ_matrix_sim, moment_time_history_EQ_matrix_sim, storey_moment_time_history_EQ_matrix_sim, shear_time_history_EQ_matrix_sim, storey_shear_time_history_EQ_matrix_sim, load_time_history_EQ_matrix_sim, ground_accel_time_history_EQ_matrix_sim, rel_accel_time_history_EQ_matrix_sim, total_accel_time_history_EQ_matrix_sim = calculate_structural_response(
    gamma,
    vibration_period,
    acc_sim,
    dt_sim,
    storey,
    participation_factor,
    phi,
    phi_1,
    phi_2,
    phi_3,
    phi_4,
    alpha,
    c=c,
)
# 3d extract peak structural response values from each earthquake ground motion record
disp_time_history_peak_obs, slope_time_history_peak_obs, moment_time_history_peak_obs, storey_moment_time_history_peak_obs, shear_time_history_peak_obs, storey_shear_time_history_peak_obs, load_time_history_peak_obs, ground_accel_time_history_peak_obs, rel_accel_time_history_peak_obs, total_accel_time_history_peak_obs = extract_peak_structural_response(
    storey,
    disp_time_history_EQ_matrix_obs,
    slope_time_history_EQ_matrix_obs,
    moment_time_history_EQ_matrix_obs,
    storey_moment_time_history_EQ_matrix_obs,
    shear_time_history_EQ_matrix_obs,
    storey_shear_time_history_EQ_matrix_obs,
    load_time_history_EQ_matrix_obs,
    ground_accel_time_history_EQ_matrix_obs,
    rel_accel_time_history_EQ_matrix_obs,
    total_accel_time_history_EQ_matrix_obs,
)
disp_time_history_peak_sim, slope_time_history_peak_sim, moment_time_history_peak_sim, storey_moment_time_history_peak_sim, shear_time_history_peak_sim, storey_shear_time_history_peak_sim, load_time_history_peak_sim, ground_accel_time_history_peak_sim, rel_accel_time_history_peak_sim, total_accel_time_history_peak_sim = extract_peak_structural_response(
    storey,
    disp_time_history_EQ_matrix_sim,
    slope_time_history_EQ_matrix_sim,
    moment_time_history_EQ_matrix_sim,
    storey_moment_time_history_EQ_matrix_sim,
    shear_time_history_EQ_matrix_sim,
    storey_shear_time_history_EQ_matrix_sim,
    load_time_history_EQ_matrix_sim,
    ground_accel_time_history_EQ_matrix_sim,
    rel_accel_time_history_EQ_matrix_sim,
    total_accel_time_history_EQ_matrix_sim,
)

print("disp_time_history_peak, slope_time_history_peak, moment_time_history_peak, storey_moment_time_history_peak, shear_time_history_peak, storey_shear_time_history_peak, load_time_history_peak, ground_accel_time_history_peak, rel_accel_time_history_peak, total_accel_time_history_peak")
print("SIM")
for i in range(len(disp_time_history_peak_sim)):
    print(
        ",".join(map(str, [disp_time_history_peak_sim[i],
        slope_time_history_peak_sim[i],
        moment_time_history_peak_sim[i],
        storey_moment_time_history_peak_sim[i],
        shear_time_history_peak_sim[i],
        storey_shear_time_history_peak_sim[i],
        load_time_history_peak_sim[i],
        ground_accel_time_history_peak_sim[i],
        rel_accel_time_history_peak_sim[i],
        total_accel_time_history_peak_sim[i]]))
    )
print("OBS")
for i in range(len(disp_time_history_peak_obs)):
    print(
        ",".join(map(str, [disp_time_history_peak_obs[i],
        slope_time_history_peak_obs[i],
        moment_time_history_peak_obs[i],
        storey_moment_time_history_peak_obs[i],
        shear_time_history_peak_obs[i],
        storey_shear_time_history_peak_obs[i],
        load_time_history_peak_obs[i],
        ground_accel_time_history_peak_obs[i],
        rel_accel_time_history_peak_obs[i],
        total_accel_time_history_peak_obs[i]]))
    )
