"""Intensity Measure Implementations."""

import gc
import multiprocessing
import warnings
from collections.abc import Callable
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Optional

import numba
import numexpr as ne
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
import xarray as xr
from pyfftw.interfaces import numpy_fft as fft

from IM import ko_matrices


class Component(IntEnum):
    """Component index enumeration."""

    COMP_0 = 0
    """Index for the 0° component of a waveform."""
    COMP_90 = 1
    """Index for the 90° component of a waveform."""
    COMP_VER = 2
    """Index for the vertical component of a waveform."""


class IM(StrEnum):
    """Intensity Measure enumeration."""

    PGA = "PGA"
    PGV = "PGV"
    CAV = "CAV"
    CAV5 = "CAV5"
    Ds575 = "Ds575"
    Ds595 = "Ds595"
    AI = "AI"
    pSA = "pSA"  # noqa: N815
    FAS = "FAS"


@numba.njit(cache=True)
def newmark_estimate_psa(
    waveforms: npt.NDArray[np.float32],
    dt: np.float32,
    w: npt.NDArray[np.float32],
    xi: np.float32 = np.float32(0.05),
    gamma: np.float32 = np.float32(1 / 2),
    beta: np.float32 = np.float32(1 / 4),
    m: np.float32 = np.float32(1),
) -> npt.NDArray[np.float32]:  # pragma: no cover
    """Compute pseudo-spectral acceleration using the Newmark-beta method [1]_ [2]_.

    Parameters
    ----------
    waveforms : ndarray of float32 with shape `(n_stations, n_timesteps)`
        Acceleration waveforms array.
    dt : float32
        Time step between consecutive samples (s).
    w : ndarray of float32
        Angular frequencies of single-degree-of-freedom oscillators (Hz).
    xi : float32, optional
        Damping coefficient, by default 0.05
    gamma : float32, optional
        Newmark method gamma parameter, by default 0.5
    beta : float32, optional
        Newmark method beta parameter, by default 0.25
    m : float32, optional
        Mass parameter, by default 1.0

    Returns
    -------
    ndarray of float32 with shape `(n_stations, n_timesteps, n_frequencies)`
        Response curves for SDOF oscillators.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Newmark-beta_method
    .. [2] ENCI335 (Structural Analysis), Chapter 4: Course notes on Newmark-beta method
    """
    c = np.float32(2 * xi) * w
    a = 1 / (beta * dt) * m + (gamma / beta) * c
    b = 1 / (2 * beta) * m + dt * (gamma / (2 * beta) - 1) * c
    k = np.square(w)
    kbar = k + (gamma / (beta * dt)) * c + 1 / (beta * dt**2) * m
    u = np.zeros(
        shape=(waveforms.shape[0], waveforms.shape[1], w.size), dtype=np.float32
    )
    # calculations for each time step
    dudt = np.zeros_like(w, dtype=np.float32)
    #         ns         np x ns    np x ns
    for i in range(waveforms.shape[0]):
        d2udt2 = (-m * waveforms[i, 0] - c * dudt - k * u[i, 0]) / m
        for j in range(1, waveforms.shape[1]):
            d_pti = -m * (waveforms[i, j] - waveforms[i, j - 1])
            d_pbari = d_pti + a * dudt + b * d2udt2
            d_ui = d_pbari / kbar
            d_dudti = (
                (gamma / (beta * dt)) * d_ui
                - (gamma / beta) * dudt
                + dt * (1 - gamma / (2 * beta)) * d2udt2
            )
            d_d2udt2i = (
                1 / (beta * dt**2) * d_ui
                - 1 / (beta * dt) * dudt
                - 1 / (2 * beta) * d2udt2
            )
            # Convert from incremental formulation and store values in vector
            u[i, j] = u[i, j - 1] + d_ui
            dudt += d_dudti
            d2udt2 += d_d2udt2i

        dudt[:] = np.float32(0.0)
    return u


def rotate_components(
    step_000: np.ndarray,
    step_090: np.ndarray,
    theta: np.ndarray,
    out: Optional[np.ndarray] = None,
    use_numexpr: bool = True,
):
    """
    Helper function to handle rotation computation using either numexpr or numpy.

    Parameters
    ----------
    step_000 : np.ndarray
        Array containing the 000 component of the waveforms.
    step_090 : np.ndarray
        Array containing the 090 component of the waveforms.
    theta : np.ndarray
        Array containing the angles at which to rotate the components.
    out : np.ndarray, optional
        Array to store the output of the computation, by default None.
    use_numexpr : bool, optional
        Use numexpr for computation, by default True.
    """
    if use_numexpr:
        return ne.evaluate(
            "abs(comp_000 * cos(theta) + comp_090 * sin(theta))",
            {
                "comp_000": step_000[..., np.newaxis],
                "theta": theta[np.newaxis, ...],
                "comp_090": step_090[..., np.newaxis],
            },
            # The out parameter should accept None, but doesn't because of a bug in numexpr
            out=out,  # type: ignore
        )
    else:
        return np.abs(
            step_000[..., np.newaxis] * np.cos(theta)[np.newaxis, ...]
            + step_090[..., np.newaxis] * np.sin(theta)[np.newaxis, ...]
        )


def rotd_psa_values(
    comp_000: npt.NDArray[np.float32],
    comp_090: npt.NDArray[np.float32],
    w: npt.NDArray[np.float32],
    use_numexpr: bool = True,
) -> npt.NDArray[np.float32]:
    """Compute rotated pseudo-spectral acceleration statistics.

    Parameters
    ----------
    comp_000 : ndarray of float32 with shape `(n_periods, n_stations, n_timesteps)`
        PSA in 000 component (cm/s^2).
    comp_090 : ndarray of float32 with shape `(n_periods, n_stations, n_timesteps)`
        PSA in 090 component (cm/s^2).
    w : ndarray of float32
        Natural angular frequencies of oscillators (Hz).
    use_numexpr : bool, optional
        Use numexpr for computation, by default True.

    Returns
    -------
    ndarray of float32 with shape `(n_stations, n_periods, 3)`
        Array containing minimum (rotd0), median (rotd50) and maximum (rotd100) PSA values.
    """
    theta = np.linspace(0, np.pi, num=180, dtype=np.float32)
    psa = np.zeros((comp_000.shape[0], comp_000.shape[-1], 3), dtype=np.float32)
    out = np.zeros((comp_000.shape[0], *comp_000.shape[1:], 180), dtype=np.float32)
    w2 = np.square(w, dtype=np.float32)

    psa = np.transpose(
        np.percentile(
            np.max(
                rotate_components(
                    comp_000,
                    comp_090,
                    theta,
                    out=out[: len(comp_000)],
                    use_numexpr=use_numexpr,
                )[: len(comp_000)],
                axis=1,
            ),
            [0, 50, 100],
            axis=-1,
        ),
        [1, 2, 0],
    ).astype(np.float32)

    del out
    gc.collect()  # This is required because Python's GC is too lazy to remove the out array when it should
    return w2[np.newaxis, :, np.newaxis] * psa


def pseudo_spectral_acceleration(
    waveforms: npt.NDArray[np.float32],
    periods: npt.NDArray[np.float32],
    dt: np.float32,
    psa_rotd_maximum_memory_allocation: Optional[float] = None,
    cores: int = multiprocessing.cpu_count(),
    use_numexpr: bool = True,
) -> xr.DataArray:
    """Compute pseudo-spectral acceleration statistics for waveforms.

    Parameters
    ----------
    waveforms : ndarray of float32 with shape `(n_stations, n_timesteps, n_components)`
        Acceleration waveforms array (g).
    periods : ndarray of float32
        Periods for PSA computation (s). These correspond to SDOF oscillator natural frequencies.
    dt : float
        Timestep resolution of waveform array (s).
    psa_rotd_maximum_memory_allocation : float, optional
        Maximum memory allocation for PSA rotation calculations (bytes).
    cores : int, optional
        Number of CPU cores to use, by default all available cores.
    use_numexpr : bool, optional
        Use numexpr for computation, by default True.

    Returns
    -------
    xr.DataArray
        DataArray containing PSA statistics for each
        station, period and component ['000', '090', 'ver', 'geom', 'rotd0', 'rotd50', 'rotd100'].
    """
    w = 2 * np.pi / periods

    # Step size is the minimum of either the CPU count, or the maximum number of
    # steps that fits within the psa_rotd_maximum_memory_allocation.
    if psa_rotd_maximum_memory_allocation:
        step = min(
            int(
                psa_rotd_maximum_memory_allocation
                / (180 * len(w) * waveforms[0].nbytes)
            ),
            cores,
        )
    else:
        step = cores
    if step < 1:
        raise ValueError(
            "PSA rotd memory allocation is too small (cannot even calculate a single station's pSA)."
        )

    rotd_psa = np.zeros((waveforms.shape[0], len(w), 3), dtype=np.float32)

    comp_0_psa = np.zeros((waveforms.shape[0], len(w)), dtype=np.float32)
    comp_90_psa = np.zeros((waveforms.shape[0], len(w)), dtype=np.float32)
    comp_ver_psa = np.zeros((waveforms.shape[0], len(w)), dtype=np.float32)

    for i in range(0, waveforms.shape[0], step):
        comp_0 = newmark_estimate_psa(
            waveforms[i : i + step, :, Component.COMP_0.value],
            dt,
            w,
        )

        comp_90 = newmark_estimate_psa(
            waveforms[i : i + step, :, Component.COMP_90.value],
            dt,
            w,
        )

        rotd_psa[i : i + step, :, :] = rotd_psa_values(
            comp_0, comp_90, w, use_numexpr=use_numexpr
        )

        conversion_factor = np.square(w)[np.newaxis, :]
        comp_0_psa[i : i + step, :] = conversion_factor * np.abs(comp_0).max(axis=1)
        comp_90_psa[i : i + step, :] = conversion_factor * np.abs(comp_90).max(axis=1)

        comp_ver_psa[i : i + step, :] = conversion_factor * np.abs(
            newmark_estimate_psa(
                waveforms[i : i + step, :, Component.COMP_VER.value], dt, w
            )
        ).max(axis=1)

    geom_psa = np.sqrt(comp_0_psa * comp_90_psa)

    return xr.DataArray(
        np.stack(
            [
                comp_0_psa,
                comp_90_psa,
                comp_ver_psa,
                geom_psa,
                rotd_psa[:, :, 0],
                rotd_psa[:, :, 1],
                rotd_psa[:, :, 2],
            ],
            axis=0,
        ),
        name=IM.pSA.value,
        dims=("component", "station", "period"),
        coords={
            "station": np.arange(waveforms.shape[0]),
            "period": periods,
            "component": ["000", "090", "ver", "geom", "rotd0", "rotd50", "rotd100"],
        },
    )


def compute_intensity_measure_rotd(
    waveforms: npt.NDArray[np.float32],
    intensity_measure: Callable,
    use_numexpr: bool = True,
) -> pd.DataFrame:
    """Compute rotated intensity measure statistics for multiple waveforms.

    Parameters
    ----------
    waveforms : ndarray of float32 with shape `(n_stations, n_timesteps, n_components)`
        Acceleration waveforms array (g).
    intensity_measure : callable
        Function that computes the intensity measure. Should accept a 2D array
        of shape `(n_stations, n_timesteps)` and return a 1D array of shape
        `(n_stations,)`.
    use_numexpr : bool, optional
        Use numexpr for computation, by default True.

    Returns
    -------
    pandas.DataFrame with columns `['000', '090', 'ver', 'geom', 'rotd100', 'rotd50', 'rotd0']`
        DataFrame containing intensity measure statistics. Each row represents
        statistics for a single station.
    """
    (stations, _, _) = waveforms.shape
    values = np.zeros(shape=(stations, 180), dtype=waveforms.dtype)

    comp_0 = waveforms[:, :, Component.COMP_0.value]
    comp_90 = waveforms[:, :, Component.COMP_90.value]
    for i in range(180):
        theta = np.deg2rad(i).astype(np.float32)

        if use_numexpr:
            values[:, i] = intensity_measure(
                ne.evaluate(
                    "cos(theta) * comp_0 + sin(theta) * comp_90",
                    {"comp_0": comp_0, "comp_90": comp_90, "theta": theta},
                ),
            )
        else:
            values[:, i] = intensity_measure(
                np.cos(theta) * comp_0 + np.sin(theta) * comp_90
            )

    comp_0 = values[:, 0]
    comp_90 = values[:, 90]
    comp_ver = waveforms[:, :, Component.COMP_VER.value]
    ver = intensity_measure(comp_ver)
    rotated_max = np.max(values, axis=1)
    rotated_median = np.median(values, axis=1)
    rotated_min = np.min(values, axis=1)
    return pd.DataFrame(
        {
            "000": comp_0,
            "090": comp_90,
            "ver": ver,
            "geom": np.sqrt(comp_0 * comp_90),
            "rotd100": rotated_max,
            "rotd50": rotated_median,
            "rotd0": rotated_min,
        }
    )


@numba.njit(parallel=True, cache=True)
def trapz(
    waveforms: npt.NDArray[np.float32], dt: float
) -> npt.NDArray[np.float32]:  # pragma: no cover
    """Compute parallel trapezium numerical integration.

    Parameters
    ----------
    waveforms : ndarray of float32 with shape `(n_stations, n_timesteps)`
        Waveform accelerations to integrate (units).
    dt : float
        Timestep resolution of the waveform array (t).

    Returns
    -------
    ndarray of float32 with shape (n_stations,)
        Integrated values for each waveform (units-sec).

    Notes
    -----
    This is a parallel implementation equivalent to np.trapz, optimized for
    performance with numba.
    """
    sums = np.zeros((waveforms.shape[0],), np.float32)
    for i in numba.prange(waveforms.shape[0]):  # type: ignore
        for j in range(waveforms.shape[1]):
            if j == 0 or j == waveforms.shape[1] - 1:
                sums[i] += waveforms[i, j] / 2
            else:
                sums[i] += waveforms[i, j]
    return sums * dt


def significant_duration(
    waveforms: npt.NDArray[np.float32],
    dt: float,
    percent_low: float,
    percent_high: float,
    use_numexpr: bool = True,
) -> npt.NDArray[np.float32]:
    """Compute significant duration based on Arias Intensity accumulation.

    Parameters
    ----------
    waveforms : ndarray of float32 with shape `(n_stations, n_timesteps)`
        Waveform accelerations (g).
    dt : float
        Timestep resolution of the waveform array (s).
    percent_low : float
        Lower bound percentage for significant duration (e.g., 5 for 5%).
    percent_high : float
        Upper bound percentage for significant duration (e.g., 95 for 95%).
    use_numexpr : bool, optional
        Use numexpr for computation, by default True.

    Returns
    -------
    ndarray of float32
        Significant duration values in seconds. Shape: (n_stations,).
    """
    arias_intensity = _cumulative_arias_intensity(waveforms, dt)
    arias_intensity /= arias_intensity[:, -1][:, np.newaxis]
    if use_numexpr:
        sum_mask = ne.evaluate(
            "(arias_intensity >= percent_low / 100) & (arias_intensity <= percent_high / 100)"
        )
    else:
        sum_mask = (arias_intensity >= percent_low / 100) & (
            arias_intensity <= percent_high / 100
        )
    threshold_values = np.count_nonzero(sum_mask, axis=1) * dt
    return threshold_values.ravel()


def fourier_amplitude_spectra(
    waveforms: npt.NDArray[np.float32],
    dt: float,
    freqs: npt.NDArray[np.float32],
    ko_directory: Path,
    cores: int = multiprocessing.cpu_count(),
) -> xr.DataArray:
    """Compute Fourier Amplitude Spectrum (FAS) of seismic waveforms.

    The FAS is computed using FFT and then smoothed using the Konno-Ohmachi
    smoothing algorithm.

    Parameters
    ----------
    waveforms : ndarray of float32 with shape `(n_stations, n_timesteps, n_components)`
        Waveform array (g).
    dt : float
        Timestep resolution of the waveforms (s).
    freqs : ndarray of float32
        Frequencies at which to compute FAS (Hz).
    ko_directory : Path
        Directory containing precomputed Konno-Ohmachi matrices.
    cores : int, optional
        Number of CPU cores to use, by default all available cores.

    Returns
    -------
    xr.DataArray
        DataArray containing FAS values for each station, frequency and component ['000', '090', 'ver', 'eas'].
    """
    nyquist_frequency = 1 / (2 * dt)
    max_frequency = freqs.max()
    if max_frequency > nyquist_frequency:
        warnings.warn(
            RuntimeWarning(
                f"Attempting to compute FAS for frequencies above Nyquist frequency {nyquist_frequency:.2e} Hz. Results only include frequencies at or below Nyquist frequency {nyquist_frequency:.2e} Hz."
            ),
        )
        freqs = freqs[freqs <= nyquist_frequency]

    n_fft = 2 ** int(np.ceil(np.log2(waveforms.shape[1])))
    # Swap the first and last axes to ensure array has shape
    # (n_components, n_stations, nt) or (n_components, nt).
    waveforms = np.moveaxis(waveforms, -1, 0)
    # Essential! Repack the waveform array so that the rows are
    # contiguous in memory.
    waveforms = np.ascontiguousarray(waveforms)
    fa_frequencies = np.fft.rfftfreq(n_fft, dt)
    waveform_shape = list(waveforms.shape)

    waveform_shape[-1] = len(fa_frequencies)
    n_components = waveform_shape[0]
    fa_spectrum = np.empty(waveform_shape, dtype=waveforms.dtype)
    for i in range(n_components):
        fa_spectrum[i] = np.abs(
            fft.rfft(waveforms[i], n=n_fft, axis=-1, threads=cores) * dt
        )

    # Get appropriate konno ohmachi matrix
    konno = ko_matrices.get_konno_matrix(fa_spectrum.shape[-1], ko_directory)
    # For optimal matrix-product calculation, repack the matrix in column-major order
    # (i.e. Fortran order) to optimise cache efficiency and allow
    # multi-threaded BLAS if enabled.
    #
    # NOTE: for matrices generated by new versions of
    # gen_ko_matrix.py, this is a no-op because the arrays are already
    # Fortran contiguous. Hence it creates no copy in memory.
    konno = np.asfortranarray(konno)
    fas_smooth = fa_spectrum @ konno
    if np.ndim(fas_smooth) == 2:
        fas_smooth = np.expand_dims(fas_smooth, axis=1)
    interpolator = sp.interpolate.make_interp_spline(
        fa_frequencies, fas_smooth, axis=-1, k=1
    )
    fas_smooth = interpolator(freqs)

    eas = np.sqrt(
        0.5
        * (
            np.square(fas_smooth[Component.COMP_0.value])
            + np.square(fas_smooth[Component.COMP_90.value])
        )
    )
    geom_fas = np.sqrt(
        fas_smooth[Component.COMP_0.value] * fas_smooth[Component.COMP_90.value]
    )

    return xr.DataArray(
        np.stack(
            [
                fas_smooth[Component.COMP_0.value],
                fas_smooth[Component.COMP_90.value],
                fas_smooth[Component.COMP_VER.value],
                geom_fas,
                eas,
            ],
            axis=0,
        ),
        name=IM.FAS.value,
        dims=("component", "station", "frequency"),
        coords={
            "component": ["000", "090", "ver", "geom", "eas"],
            "frequency": freqs,
            "station": np.arange(fas_smooth.shape[1]),
        },
    )


@numba.njit(parallel=True, cache=True)
def _cumulative_absolute_velocity(
    waveform: npt.NDArray[np.float32], dt: float
) -> npt.NDArray[np.float32]:  # pragma: no cover
    """Compute Cumulative Absolute Velocity (CAV) of waveforms.

    Parameters
    ----------
    waveform : ndarray of float32 with shape `(n_stations, n_timesteps)`
        Waveform accelerations (g).
    dt : float
        Timestep resolution of the waveform array (s).

    Returns
    -------
    ndarray of float32 with shape `(n_stations,)`
        CAV values for each waveform (m/s).

    Notes
    -----
    Implementation optimized by Jérôme Richard[0]_ for accurate integration of
    signals with sign changes.

    References
    ----------
    .. [0] https://stackoverflow.com/questions/79164983/numerically-integrating-signals-with-absolute-value/79173972#79173972
    """
    cav = np.zeros((waveform.shape[0],), dtype=np.float32)
    dtf = np.float32(dt)
    half = np.float32(0.5)
    for i in numba.prange(np.int32(waveform.shape[0])):  # type: ignore
        tmp = np.float32(0)
        for j in range(np.int32(waveform.shape[1] - 1)):
            v1 = waveform[i, j]
            v2 = waveform[i, j + 1]
            if min(v1, v2) >= 0 or max(v1, v2) <= 0:
                tmp += dtf * (np.abs(v1) + np.abs(v2))
            else:
                inv_slope = dtf / (v2 - v1)
                x0 = -v1 * inv_slope
                tmp += x0 * np.abs(v1) + (dtf - x0) * np.abs(v2)
        cav[i] = tmp * half
    g = np.float32(9.81)
    return g * cav


@numba.njit(parallel=True, cache=True)
def _arias_intensity(
    waveform: npt.NDArray[np.float32], dt: float
) -> npt.NDArray[np.float32]:  # pragma: no cover
    """Compute Arias Intensity (AI) of waveforms.

    Parameters
    ----------
    waveform : ndarray of float32 with shape `(n_stations, n_timesteps)`
        Waveform accelerations (g).
    dt : float
        Timestep resolution of the waveform array (s).

    Returns
    -------
    ndarray of float32 with shape `(n_stations,)`
        AI values for each waveform (m/s).
    """
    ai = np.zeros((waveform.shape[0],), dtype=np.float32)
    dtf = np.float32(dt)
    half = np.float32(0.5)
    for i in numba.prange(np.int32(waveform.shape[0])):  # type: ignore
        tmp = np.float32(0)
        for j in range(np.int32(waveform.shape[1] - 1)):
            v1 = waveform[i, j]
            v2 = waveform[i, j + 1]
            if min(v1, v2) >= 0 or max(v1, v2) <= 0:
                tmp += dtf * (np.square(v1) + np.square(v2))
            else:
                inv_slope = dtf / (v2 - v1)
                x0 = -v1 * inv_slope
                tmp += x0 * np.square(v1) + (dtf - x0) * np.square(v2)
        ai[i] = tmp * half
    g = np.float32(9.81)
    return np.pi * half * g * ai


@numba.njit(parallel=True, cache=True)
def _cumulative_arias_intensity(
    waveform: npt.NDArray[np.float32], dt: float
) -> npt.NDArray[np.float32]:  # pragma: no cover
    """Compute the cumulative Arias Intensity (AI) of a waveform.

    Parameters
    ----------
    waveform : npt.NDArray[np.float32]
        A 3D array of shape `(n_stations, n_samples, n_components)` for each
        waveform in each measured component (g).
    dt : float
        The time step (in seconds) between consecutive samples in the waveform (s).

    Returns
    -------
    npt.NDArray[np.float32]
        A 1D array of shape `(n_signals,)` containing the AI values for each
        input waveform.
    """
    ai = np.zeros_like(waveform, dtype=np.float32)
    dtf = np.float32(dt)
    half = np.float32(0.5)
    for i in numba.prange(np.int32(waveform.shape[0])):  # type: ignore
        tmp = np.float32(0)
        for j in range(np.int32(waveform.shape[1] - 1)):
            v1 = waveform[i, j]
            v2 = waveform[i, j + 1]
            if min(v1, v2) >= 0 or max(v1, v2) <= 0:
                tmp += dtf * (np.square(v1) + np.square(v2))
            else:
                inv_slope = dtf / (v2 - v1)
                x0 = -v1 * inv_slope
                tmp += x0 * np.square(v1) + (dtf - x0) * np.square(v2)

            ai[i, j + 1] = tmp

    g = np.float32(9.81)
    return ai * np.pi * half * g


def peak_ground_acceleration(
    waveform: npt.NDArray[np.float32], use_numexpr: bool = True
) -> pd.DataFrame:
    """Compute Peak Ground Acceleration (PGA) for waveforms.

    Parameters
    ----------
    waveform : ndarray of float32 with shape `(n_stations, n_timesteps, n_components)`
        Acceleration waveforms in g units.
    use_numexpr : bool, optional
        Use numexpr for computation, by default True.

    Returns
    -------
    pandas.DataFrame with columns `['000', '090', 'ver', 'geom', 'rotd100', 'rotd50', 'rotd0']`
        DataFrame containing PGA values with rotated components in g-units.
    """
    return compute_intensity_measure_rotd(
        waveform, lambda v: np.abs(v).max(axis=1), use_numexpr=use_numexpr
    )


def peak_ground_velocity(
    waveform: npt.NDArray[np.float32], dt: float, use_numexpr: bool = True
) -> pd.DataFrame:
    """Compute Peak Ground Velocity (PGV) for waveforms.

    Parameters
    ----------
    waveform : ndarray of float32 with shape `(n_stations, n_timesteps, n_components)`
        Acceleration waveforms in g units.
    dt : float
        Timestep resolution of the waveform array.
    use_numexpr: bool, optional
        Use numexpr for computation, by default True.

    Returns
    -------
    pandas.DataFrame with columns `['000', '090', 'ver', 'geom', 'rotd100', 'rotd50', 'rotd0']`
        DataFrame containing PGV values with rotated components. Values are
        in cm/s.
    """
    g = 981
    return compute_intensity_measure_rotd(
        sp.integrate.cumulative_trapezoid(waveform, dx=dt, axis=1),
        lambda v: g * np.abs(v).max(axis=1),
        use_numexpr=use_numexpr,
    )


def cumulative_absolute_velocity(
    waveform: npt.NDArray[np.float32], dt: float, threshold: Optional[float] = None
) -> pd.DataFrame:
    """Compute Cumulative Absolute Velocity (CAV) for waveforms.

    Parameters
    ----------
    waveform : ndarray of float32 with shape `(n_stations, n_timesteps, n_components)`
        Acceleration waveforms (g).
    dt : float
        Timestep resolution of the waveform array (s).
    threshold : float, optional
        The minimum acceleration threshold, in cm/s^2. CAV5 is found by using
        `threshold` = 5. Acceleration values below `threshold` are set to zero.

    Returns
    -------
    pandas.DataFrame with columns `['000', '090', 'ver', 'geom', 'rotd100', 'rotd50', 'rotd0']`
        DataFrame containing CAV values (m/s) with rotated components.
    """

    comp_0 = waveform[:, :, Component.COMP_0]
    comp_90 = waveform[:, :, Component.COMP_90]
    comp_ver = waveform[:, :, Component.COMP_VER]

    if threshold:
        g = 981
        comp_0 = np.where(np.abs(comp_0) < threshold / g, np.float32(0), comp_0)
        comp_90 = np.where(np.abs(comp_90) < threshold / g, np.float32(0), comp_90)
        comp_ver = np.where(np.abs(comp_ver) < threshold / g, np.float32(0), comp_ver)

    comp_0_cav = _cumulative_absolute_velocity(comp_0, dt)
    comp_90_cav = _cumulative_absolute_velocity(comp_90, dt)
    comp_ver_cav = _cumulative_absolute_velocity(comp_ver, dt)

    return pd.DataFrame(
        {
            "000": comp_0_cav,
            "090": comp_90_cav,
            "ver": comp_ver_cav,
            "geom": np.sqrt(comp_0_cav * comp_90_cav),
        }
    )


def arias_intensity(waveform: npt.NDArray[np.float32], dt: float) -> pd.DataFrame:
    """Compute Arias Intensity (AI) for waveforms.

    Parameters
    ----------
    waveform : ndarray of float32 with shape `(n_stations, n_timesteps, n_components)`
        Acceleration waveforms (g).
    dt : float
        Timestep resolution of the waveform array (s).

    Returns
    -------
    pandas.DataFrame with columns `['intensity_measure', '000', '090', 'ver', 'geom']`
        DataFrame containing Arias Intensity values (m/s). The 'geom' component
        is the geometric mean of the 000 and 090 components.
    """
    arias_intensity_0 = _arias_intensity(waveform[:, :, Component.COMP_0.value], dt)
    arias_intensity_90 = _arias_intensity(waveform[:, :, Component.COMP_90.value], dt)
    arias_intensity_ver = _arias_intensity(waveform[:, :, Component.COMP_VER.value], dt)

    return pd.DataFrame(
        {
            "000": arias_intensity_0,
            "090": arias_intensity_90,
            "ver": arias_intensity_ver,
            "geom": np.sqrt(arias_intensity_0 * arias_intensity_90),
        }
    )


def ds575(
    waveform: npt.NDArray[np.float32], dt: float, use_numexpr: bool = True
) -> pd.DataFrame:
    """Compute 5-75% Significant Duration (DS575) for waveforms.

    Parameters
    ----------
    waveform : ndarray of float32 with shape `(n_stations, n_timesteps, n_components)`
        Acceleration waveforms (g).
    dt : float
        Timestep resolution of the waveform array (s).
    use_numexpr: bool, optional
        Use numexpr for computation, by default True.

    Returns
    -------
    pandas.DataFrame with columns `['000', '090', 'ver', 'geom', 'rotd100', 'rotd50', 'rotd0']`
        DataFrame containing DS575 values (in seconds) with rotated components.
    """
    significant_duration_0 = significant_duration(
        waveform[:, :, Component.COMP_0.value], dt, 5, 75, use_numexpr
    )
    significant_duration_90 = significant_duration(
        waveform[:, :, Component.COMP_90.value], dt, 5, 75, use_numexpr
    )
    significant_duration_ver = significant_duration(
        waveform[:, :, Component.COMP_VER.value], dt, 5, 75, use_numexpr
    )

    return pd.DataFrame(
        {
            "000": significant_duration_0,
            "090": significant_duration_90,
            "ver": significant_duration_ver,
            "geom": np.sqrt(significant_duration_0 * significant_duration_90),
        }
    )


def ds595(
    waveform: npt.NDArray[np.float32], dt: float, use_numexpr: bool = True
) -> pd.DataFrame:
    """Compute 5-95% Significant Duration (DS595) for waveforms.

    Parameters
    ----------
    waveform : ndarray of float32 with shape `(n_stations, n_timesteps, n_components)`
        Acceleration waveforms (g).
    dt : float
        Timestep resolution of the waveform array (s).
    use_numexpr: bool, optional
        Use numexpr for computation, by default True.

    Returns
    -------
    pandas.DataFrame with columns `['000', '090', 'ver', 'geom', 'rotd100', 'rotd50', 'rotd0']`
        DataFrame containing DS595 values (in seconds) with rotated components.
    """
    significant_duration_0 = significant_duration(
        waveform[:, :, Component.COMP_0.value], dt, 5, 95, use_numexpr
    )
    significant_duration_90 = significant_duration(
        waveform[:, :, Component.COMP_90.value], dt, 5, 95, use_numexpr
    )
    significant_duration_ver = significant_duration(
        waveform[:, :, Component.COMP_VER.value], dt, 5, 95, use_numexpr
    )

    return pd.DataFrame(
        {
            "000": significant_duration_0,
            "090": significant_duration_90,
            "ver": significant_duration_ver,
            "geom": np.sqrt(significant_duration_0 * significant_duration_90),
        }
    )
