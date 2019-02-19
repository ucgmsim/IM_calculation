from IM.rspectra_calculations import rspectra as rspectra
import numpy as np
from qcore import timeseries
import pickle
import os


test_data_save_dir = '/home/jpa198/test_space/im_calc_test/pickled'
REALISATION = 'PangopangoF29_HYP01-10_S1244'
data_taken = {'get_max_nd': False,
              'get_spectral_acceleration': False,
              'get_spectral_acceleration_nd': False,
              'get_cumulative_abs_velocity_nd': False,
              'get_arias_intensity_nd': False,
              'calculate_MMI_nd': False,
              'getDs': False,
              'getDs_nd': False,
              'get_geom': False,
              }

def get_max_nd(data):
    function = 'get_max_nd'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_data.P'), 'wb') as save_file:
            pickle.dump(data, save_file)
    ret_val = np.max(np.abs(data), axis=0)
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_ret_val.P'), 'wb') as save_file:
            pickle.dump(ret_val, save_file)
    return ret_val


def get_spectral_acceleration(acceleration, period, NT, DT):
    function = 'get_spectral_acceleration'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_acceleration.P'), 'wb') as save_file:
            pickle.dump(acceleration, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_period.P'), 'wb') as save_file:
            pickle.dump(period, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_NT.P'), 'wb') as save_file:
            pickle.dump(NT, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_DT.P'), 'wb') as save_file:
            pickle.dump(DT, save_file)

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
    ret_val = rspectra.Response_Spectra(acc_step, deltat, c, period, M, gamma, beta)

    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_ret_val.P'), 'wb') as save_file:
            pickle.dump(ret_val, save_file)
        data_taken[function] = True
    return ret_val


def get_spectral_acceleration_nd(acceleration, period, NT, DT):

    # pSA
    if acceleration.ndim != 1:
        #Only useful if >2, otherwise just mirrors non nd version
        function = 'get_spectral_acceleration_nd'
        if not data_taken[function]:
            with open(os.path.join(test_data_save_dir, REALISATION, function + '_acceleration.P'), 'wb') as save_file:
                pickle.dump(acceleration, save_file)
            with open(os.path.join(test_data_save_dir, REALISATION, function + '_period.P'), 'wb') as save_file:
                pickle.dump(period, save_file)
            with open(os.path.join(test_data_save_dir, REALISATION, function + '_NT.P'), 'wb') as save_file:
                pickle.dump(NT, save_file)
            with open(os.path.join(test_data_save_dir, REALISATION, function + '_DT.P'), 'wb') as save_file:
                pickle.dump(DT, save_file)
        dims = acceleration.shape[1]
        values = np.zeros((period.size, dims))

        for i in range(dims):
            values[:, i] = get_spectral_acceleration(acceleration[:, i], period, NT, DT)

        if not data_taken[function]:
            with open(os.path.join(test_data_save_dir, REALISATION, function + '_values.P'), 'wb') as save_file:
                pickle.dump(values, save_file)
            data_taken[function] = True
        return values
    else:
        return get_spectral_acceleration(acceleration, period, NT, DT)


def get_cumulative_abs_velocity_nd(acceleration, times):
    function = 'get_cumulative_abs_velocity_nd'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_acceleration.P'), 'wb') as save_file:
            pickle.dump(acceleration, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_times.P'), 'wb') as save_file:
            pickle.dump(times, save_file)
    ret_val = np.trapz(np.abs(acceleration), times, axis=0)
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_ret_val.P'), 'wb') as save_file:
            pickle.dump(ret_val, save_file)
        data_taken[function] = True
    return ret_val


def get_arias_intensity_nd(acceleration, g, times):
    function = 'get_arias_intensity_nd'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_acceleration.P'), 'wb') as save_file:
            pickle.dump(acceleration, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_g.P'), 'wb') as save_file:
            pickle.dump(g, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_times.P'), 'wb') as save_file:
            pickle.dump(times, save_file)
    acc_in_cms = acceleration * g
    integrand = acc_in_cms ** 2
    ret_val = np.pi / (2 * g) * np.trapz(integrand, times, axis=0)
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_ret_val.P'), 'wb') as save_file:
            pickle.dump(ret_val, save_file)
        data_taken[function] = True
    return ret_val


def calculate_MMI_nd(velocities):
    function = 'calculate_MMI_nd'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_velocities.P'), 'wb') as save_file:
            pickle.dump(velocities, save_file)
    pgv = get_max_nd(velocities)
    ret_val = timeseries.pgv2MMI(pgv)
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_ret_val.P'), 'wb') as save_file:
            pickle.dump(ret_val, save_file)
        data_taken[function] = True
    return ret_val


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
    function = 'getDs'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_dt.P'), 'wb') as save_file:
            pickle.dump(dt, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_fx.P'), 'wb') as save_file:
            pickle.dump(fx, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_percLow.P'), 'wb') as save_file:
            pickle.dump(percLow, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_percHigh.P'), 'wb') as save_file:
            pickle.dump(percHigh, save_file)
    nsteps = np.size(fx)
    husid = np.zeros(nsteps)
    husid[0] = 0  # initialize first to 0
    for i in range(1, nsteps):
        husid[i] = husid[i - 1] + dt * (fx[i] ** 2)  # note that pi/(2g) is not used as doesnt affect the result
    AI = husid[-1]
    Ds = dt * (np.sum(husid / AI <= percHigh / 100.) - np.sum(husid / AI <= percLow / 100.))
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_Ds.P'), 'wb') as save_file:
            pickle.dump(Ds, save_file)
        data_taken[function] = True
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
        function = 'getDs_nd'
        if not data_taken[function]:
            with open(os.path.join(test_data_save_dir, REALISATION, function + '_dt.P'), 'wb') as save_file:
                pickle.dump(dt, save_file)
            with open(os.path.join(test_data_save_dir, REALISATION, function + '_accelerations.P'), 'wb') as save_file:
                pickle.dump(accelerations, save_file)
            with open(os.path.join(test_data_save_dir, REALISATION, function + '_percLow.P'), 'wb') as save_file:
                pickle.dump(percLow, save_file)
            with open(os.path.join(test_data_save_dir, REALISATION, function + '_percHigh.P'), 'wb') as save_file:
                pickle.dump(percHigh, save_file)
        values = np.zeros(3)
        i = 0
        for fx in accelerations.transpose():
            values[i] = getDs(dt, fx, percLow, percHigh)
            i += 1

        if not data_taken[function]:
            with open(os.path.join(test_data_save_dir, REALISATION, function + '_values.P'), 'wb') as save_file:
                pickle.dump(values, save_file)
            data_taken[function] = True
        return values


def get_geom(d1, d2):
    """
    get geom value from the 090 and 000 components
    :param d1: 090
    :param d2: 000
    :return: geom value
    """
    function = 'get_geom'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_d1.P'), 'wb') as save_file:
            pickle.dump(d1, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_d2.P'), 'wb') as save_file:
            pickle.dump(d2, save_file)
    ret_val = np.sqrt(d1 * d2)
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_ret_val.P'), 'wb') as save_file:
            pickle.dump(ret_val, save_file)
        data_taken[function] = True
    return ret_val