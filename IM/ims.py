"""Intensity Measure Implementations."""

import itertools
import multiprocessing
import os
import warnings
from collections.abc import Callable, Generator, MutableMapping
from contextlib import contextmanager
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Optional

import numexpr as ne
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
import tqdm
import xarray as xr
from pyfftw.interfaces import numpy_fft as fft

from IM import (
    _utils,  # type: ignore[unresolved-import]
    ko_matrices,
)


@contextmanager
def environment(
    **variables: str,
    # NOTE: the type here could be the os._Environ type defined in the
    # os module, but this means we don't rely on any specific
    # behaviour of that object which might change later down the line
    # (or indeed, they may remove the os._Environ object at any time
    # because it is an internal class).
) -> Generator[MutableMapping[str, str]]:
    """Update an environment and revert after exit

    Parameters
    ----------
    **variables : str or bytes
        Environment values to update inside the context manager.

    Yields
    ------
    MutableMapping
        The mapping object representing `os.environ`.
    """
    # Code to acquire resource, e.g.:
    old_environment: dict[str, str] = os.environ.copy()
    try:
        os.environ.update(variables)  # type: ignore[no-matching-overload]
        yield os.environ
    finally:
        for key in set(os.environ) - set(old_environment):
            del os.environ[key]
        os.environ.update(old_environment)


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


def pseudo_spectral_acceleration(
    waveforms: npt.NDArray[np.float64 | np.float32],
    periods: npt.NDArray[np.float64],
    dt: np.float64,
    psa_rotd_maximum_memory_allocation: Optional[float] = None,
    cores: int = multiprocessing.cpu_count(),
    step: int | None = None,
    use_tqdm: bool = False,
) -> xr.DataArray:
    """Compute pseudo-spectral acceleration statistics for waveforms.

    Parameters
    ----------
    waveforms : ndarray of float32 with shape `(n_components, n_stations, n_timesteps)`
        Acceleration waveforms array (g).
    periods : ndarray of float32
        Periods for PSA computation (s). These correspond to SDOF oscillator natural frequencies.
    dt : float
        Timestep resolution of waveform array (s).
    psa_rotd_maximum_memory_allocation : float, optional
        Maximum memory allocation for PSA rotation calculations (bytes).
    cores : int, optional
        Number of CPU cores to use, by default all available cores.
    step : int, optional
        Number of steps to use. If `None`, will use the number of cores as the default number of steps.
    use_tqdm : bool, optional
        If true, show tqdm progress bar.

    Returns
    -------
    xr.DataArray
        DataArray containing PSA statistics for each
        station, period and component ['000', '090', 'ver', 'geom', 'rotd0', 'rotd50', 'rotd100'].
    """
    waveforms = np.ascontiguousarray(waveforms)
    angular_frequencies = 2 * np.pi / periods

    # Step size *used* to be based on the cores available but that no
    # longer holds because Rayon, the rust parallel work scheduler,
    # manages this on its own. So the real bound on step size is now
    # how much memory we have available.
    step = step or cores
    n_stations = waveforms.shape[1]
    n_frequencies = len(angular_frequencies)
    rotd_psa = np.zeros((n_frequencies, n_stations, 3), dtype=np.float64)

    comp_0_psa = np.zeros((n_frequencies, n_stations), dtype=np.float64)
    comp_90_psa = np.zeros((n_frequencies, n_stations), dtype=np.float64)
    comp_ver_psa = np.zeros((n_frequencies, n_stations), dtype=np.float64)
    xi = 0.05
    with environment(RAYON_NUM_THREADS=str(cores)):
        station_iter = range(0, n_stations, step)
        n_steps = len(station_iter) * len(angular_frequencies)
        station_period_iterator = itertools.product(
            range(len(angular_frequencies)), station_iter
        )
        if use_tqdm:
            station_period_iterator = tqdm.tqdm(station_period_iterator, total=n_steps)
        j_last: int | None = None
        for j, i in station_period_iterator:
            w = angular_frequencies[j]
            if use_tqdm and j_last != j:
                assert isinstance(station_period_iterator, tqdm.tqdm)
                j_last = j
                t0 = periods[j]
                station_period_iterator.set_description(f"Period {t0:g}")

            comp_0_chunk = waveforms[Component.COMP_0.value, i : i + step].astype(
                np.float64
            )
            comp_90_chunk = waveforms[Component.COMP_90.value, i : i + step].astype(
                np.float64
            )
            comp_0_response = _utils._newmark_beta_method(comp_0_chunk, dt, w, xi)
            comp_90_response = _utils._newmark_beta_method(comp_90_chunk, dt, w, xi)
            conversion_factor = w * w

            rotd_psa[j, i : i + step] = conversion_factor * _utils._rotd_parallel(
                comp_0_response, comp_90_response
            )

            comp_0_psa[j, i : i + step] = conversion_factor * np.abs(
                comp_0_response
            ).max(axis=1)
            comp_90_psa[j, i : i + step] = conversion_factor * np.abs(
                comp_90_response
            ).max(axis=1)

            z = waveforms[Component.COMP_VER.value, i : i + step].astype(np.float64)
            z_response = _utils._newmark_beta_method(z, dt, w, xi)
            comp_ver_psa[j, i : i + step] = conversion_factor * np.abs(z_response).max(
                axis=1
            )

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
        dims=(
            "component",
            "period",
            "station",
        ),
        coords={
            "station": np.arange(waveforms.shape[1]),
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
    arias_intensity = _utils._cumulative_arias_intensity(waveforms, dt)
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
    use_numexpr : bool, optional
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

    comp_0_cav = _utils._cumulative_absolute_velocity(comp_0, dt)
    comp_90_cav = _utils._cumulative_absolute_velocity(comp_90, dt)
    comp_ver_cav = _utils._cumulative_absolute_velocity(comp_ver, dt)

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
    arias_intensity_0 = _utils._arias_intensity(
        waveform[:, :, Component.COMP_0.value], dt
    )
    arias_intensity_90 = _utils._arias_intensity(
        waveform[:, :, Component.COMP_90.value], dt
    )
    arias_intensity_ver = _utils._arias_intensity(
        waveform[:, :, Component.COMP_VER.value], dt
    )

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
    use_numexpr : bool, optional
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
    use_numexpr : bool, optional
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
