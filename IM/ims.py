"""Intensity Measure Implementations."""

import itertools
import multiprocessing
import os
import warnings
from collections.abc import Generator, MutableMapping
from contextlib import contextmanager
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
import tqdm
import xarray as xr
from pyfftw.interfaces import numpy_fft as fft

from IM import (
    _core,  # type: ignore[unresolved-import]
    ko_matrices,
)

# (n_components, n_stations, nt)
ChunkedWaveformArray = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
# (n_stations, nt)
SingleWaveformArray = np.ndarray[tuple[int, int], np.dtype[np.float64]]
WaveformArray = ChunkedWaveformArray | SingleWaveformArray
Array1D = np.ndarray[tuple[int], np.dtype[np.float64]]


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
    PGD = "PGD"
    CAV = "CAV"
    CAV5 = "CAV5"
    Ds575 = "Ds575"
    Ds595 = "Ds595"
    AI = "AI"
    pSA = "pSA"  # noqa: N815
    FAS = "FAS"


def pseudo_spectral_acceleration(
    waveforms: ChunkedWaveformArray,
    periods: Array1D,
    dt: np.float64,
    psa_rotd_maximum_memory_allocation: Optional[float] = None,
    cores: int = multiprocessing.cpu_count(),
    step: int | None = None,
    use_tqdm: bool = False,
) -> xr.DataArray:
    """Compute pseudo-spectral acceleration (PSA) statistics.

    Calculates PSA for single-degree-of-freedom oscillators across various
    periods using the Newmark-beta method and computes rotated (RotD) statistics.

    Parameters
    ----------
    waveforms : ChunkedWaveformArray
        Acceleration waveforms (g) with shape (n_components, n_stations, nt).
    periods : Array1
        Natural periods of the oscillators (s).
    dt : np.float64
        Timestep resolution of the waveforms (s).
    psa_rotd_maximum_memory_allocation : float, optional
        Target maximum memory limit for rotation calculations.
    cores : int, optional
        Number of CPU cores for parallel processing via Rayon.
    step : int, optional
        Station chunk size for processing. Defaults to `cores` if None.
    use_tqdm : bool, optional
        Whether to display a progress bar.

    Returns
    -------
    xr.DataArray
        A 3D DataArray (component, period, station) containing PSA for
        ['000', '090', 'ver', 'geom', 'rotd0', 'rotd50', 'rotd100'].
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
        # Coverage tests don't cover this interactive usage (because
        # it doesn't change the calculations).
        if use_tqdm:  # pragma no cover
            station_period_iterator = tqdm.tqdm(station_period_iterator, total=n_steps)
        j_last: int | None = None
        for j, i in station_period_iterator:
            w = angular_frequencies[j]
            if use_tqdm and j_last != j:  # pragma: no cover
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
            comp_0_response = _core._newmark_beta_method(comp_0_chunk, dt, w, xi)
            comp_90_response = _core._newmark_beta_method(comp_90_chunk, dt, w, xi)
            conversion_factor = w * w

            rotd_psa[j, i : i + step] = conversion_factor * _core._rotd_parallel(
                comp_0_response, comp_90_response
            )

            comp_0_psa[j, i : i + step] = conversion_factor * np.abs(
                comp_0_response
            ).max(axis=1)
            comp_90_psa[j, i : i + step] = conversion_factor * np.abs(
                comp_90_response
            ).max(axis=1)

            z = waveforms[Component.COMP_VER.value, i : i + step].astype(np.float64)
            z_response = _core._newmark_beta_method(z, dt, w, xi)
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


def significant_duration(
    waveforms: ChunkedWaveformArray,
    dt: float,
    percent_low: float,
    percent_high: float,
    cores: int,
) -> pd.DataFrame:
    """Compute significant duration based on Arias Intensity accumulation.

    Parameters
    ----------
    waveforms : ChunkedWaveformArray
        Acceleration waveforms (g) with shape (n_components, n_stations, nt).
    dt : float
        Timestep resolution (s).
    percent_low : float
        Lower bound percentage (e.g., 5.0 for 5%).
    percent_high : float
        Upper bound percentage (e.g., 95.0 for 95%).
    cores : int
        Number of CPU cores for parallel execution.

    Returns
    -------
    pd.DataFrame
        Significant duration (s) for components ['000', '090', 'ver', 'geom'].
    """
    (_, n_stations, _) = waveforms.shape
    comp_0 = waveforms[Component.COMP_0]
    comp_90 = waveforms[Component.COMP_90]
    comp_ver = waveforms[Component.COMP_VER]
    quant_low = percent_low / 100
    quant_high = percent_high / 100

    if (
        cores == 1 or n_stations < 1000
    ):  # from benchmarks: for < 1000 stations the parallel overhead is not worth it.
        significant_duration_0 = _core._significant_duration(
            comp_0, dt, quant_low, quant_high
        )
        significant_duration_90 = _core._significant_duration(
            comp_90, dt, quant_low, quant_high
        )
        significant_duration_ver = _core._significant_duration(
            comp_ver, dt, quant_low, quant_high
        )
    else:
        # Testing is not big enough for multi-core execution so this codepath is not covered.
        with environment(RAYON_NUM_THREADS=str(cores)):  # pragma: no cover
            significant_duration_0 = _core._parallel_significant_duration(
                comp_0, dt, quant_low, quant_high
            )
            significant_duration_90 = _core._parallel_significant_duration(
                comp_90, dt, quant_low, quant_high
            )
            significant_duration_ver = _core._parallel_significant_duration(
                comp_ver, dt, quant_low, quant_high
            )

    return pd.DataFrame(
        {
            "000": significant_duration_0,
            "090": significant_duration_90,
            "ver": significant_duration_ver,
            "geom": np.sqrt(significant_duration_0 * significant_duration_90),
        }
    )


def fourier_amplitude_spectra(
    waveforms: ChunkedWaveformArray,
    dt: float,
    freqs: npt.NDArray[np.float64],
    ko_directory: Path,
    cores: int = multiprocessing.cpu_count(),
) -> xr.DataArray:
    """Compute Fourier Amplitude Spectrum (FAS) of seismic waveforms.

    The FAS is computed using FFT and then smoothed using the Konno-Ohmachi
    smoothing algorithm.

    Parameters
    ----------
    waveforms : ndarray of float64 with shape `(n_components, n_stations, n_timesteps)`
        Waveform array (g).
    dt : float
        Timestep resolution of the waveforms (s).
    freqs : ndarray of float64
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

    n_fft = 2 ** int(np.ceil(np.log2(waveforms.shape[-1])))
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


def compute_intensity_measure_rotd(
    waveforms: ChunkedWaveformArray, cores: int
) -> pd.DataFrame:
    """Generic wrapper to compute peak values and RotD statistics for IMs.

    Parameters
    ----------
    waveforms : ChunkedWaveformArray
        Waveform data with shape (n_components, n_stations, nt).
    cores : int
        Number of CPU cores for parallel rotation computation.

    Returns
    -------
    pd.DataFrame
        Peak ground values with columns ['000', '090', 'ver', 'geom',
        'rotd100', 'rotd50', 'rotd0'].
    """
    comp_0 = waveforms[Component.COMP_0]
    comp_90 = waveforms[Component.COMP_90]
    comp_ver = waveforms[Component.COMP_VER]
    if cores == 1:
        rotd_stats = _core._rotd(comp_0, comp_90)
    else:
        with environment(RAYON_NUM_THREADS=str(cores)):
            rotd_stats = _core._rotd_parallel(comp_0, comp_90)
    pga_comp_0 = np.abs(comp_0).max(axis=1)
    pga_comp_90 = np.abs(comp_90).max(axis=1)
    pga_ver = np.abs(comp_ver).max(axis=1)
    rotd0 = rotd_stats[:, 0]
    rotd50 = rotd_stats[:, 1]
    rotd100 = rotd_stats[:, 2]
    return pd.DataFrame(
        {
            "000": pga_comp_0,
            "090": pga_comp_90,
            "ver": pga_ver,
            "geom": np.sqrt(pga_comp_0 * pga_comp_90),
            "rotd100": rotd100,
            "rotd50": rotd50,
            "rotd0": rotd0,
        }
    )


def peak_ground_acceleration(
    waveform: ChunkedWaveformArray, cores: int
) -> pd.DataFrame:
    """Compute Peak Ground Acceleration (PGA) in g.

    Parameters
    ----------
    waveform : ChunkedWaveformArray
        Acceleration waveforms with shape (n_components, n_stations, nt).
    cores : int
        Number of CPU cores for parallel processing.

    Returns
    -------
    pd.DataFrame
        PGA values (g) for standard and rotated components.
    """
    if waveform.ndim != 3:
        raise TypeError(
            f"Waveform must have shape (n_components, n_stations, nt), but {waveform.shape=}"
        )
    elif waveform.dtype != np.float64:
        raise TypeError(f"Waveform must have dtype float64, but {waveform.dtype=}")
    return compute_intensity_measure_rotd(waveform, cores=cores)


def peak_ground_velocity(
    waveform: ChunkedWaveformArray, dt: float, cores: int
) -> pd.DataFrame:
    """Compute Peak Ground Velocity (PGV) in cm/s via trapezoidal integration.

    Parameters
    ----------
    waveform : ChunkedWaveformArray
        Acceleration waveforms (g) with shape (n_components, n_stations, nt).
    dt : float
        Timestep resolution (s).
    cores : int
        Number of CPU cores for parallel processing.

    Returns
    -------
    pd.DataFrame
        PGV values (cm/s) for standard and rotated components.
    """
    g = 981
    return compute_intensity_measure_rotd(
        g * sp.integrate.cumulative_trapezoid(waveform, dx=dt, axis=-1), cores=cores
    )


def peak_ground_displacement(
    waveform: ChunkedWaveformArray, dt: float, cores: int
) -> pd.DataFrame:
    """Compute Peak Ground Displacement (PGD) for waveforms.

    Parameters
    ----------
    waveform : ChunkedWaveformArray
        Acceleration waveforms in g units.
    dt : float
        Timestep resolution of the waveform array.
    cores : int
        Number of CPU cores for parallel processing.

    Returns
    -------
    pandas.DataFrame with columns `['000', '090', 'ver', 'geom', 'rotd100', 'rotd50', 'rotd0']`
        DataFrame containing PGD values with rotated components. Values are
        in cm.
    """
    g = 981
    # Integrate twice to get displacement in cm
    velocity = sp.integrate.cumulative_trapezoid(waveform, dx=dt, axis=-1, initial=0)
    # In-place multiplication to avoid yet another allocation
    np.multiply(g, velocity, out=velocity)
    displacement = sp.integrate.cumulative_trapezoid(
        velocity, dx=dt, axis=-1, initial=0
    )
    return compute_intensity_measure_rotd(displacement, cores=cores)


def cumulative_absolute_velocity(
    waveform: ChunkedWaveformArray,
    dt: float,
    cores: int,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Compute Cumulative Absolute Velocity (CAV) in m/s.

    Parameters
    ----------
    waveform : ChunkedWaveformArray
        Acceleration waveforms (g) with shape (n_components, n_stations, nt).
    dt : float
        Timestep resolution (s).
    cores : int
        Number of CPU cores for parallel processing.
    threshold : float, optional
        Acceleration threshold ($cm/s^2$). Values below this are ignored (e.g. 5 for CAV5).

    Returns
    -------
    pd.DataFrame
        CAV values (m/s) for ['000', '090', 'ver', 'geom'].
    """

    comp_0 = waveform[Component.COMP_0]
    comp_90 = waveform[Component.COMP_90]
    comp_ver = waveform[Component.COMP_VER]

    if threshold:
        g = 981
        comp_0 = np.where(np.abs(comp_0) < threshold / g, np.float64(0), comp_0)
        comp_90 = np.where(np.abs(comp_90) < threshold / g, np.float64(0), comp_90)
        comp_ver = np.where(np.abs(comp_ver) < threshold / g, np.float64(0), comp_ver)

    if cores == 1:
        comp_0_cav = _core._cav(comp_0, dt)
        comp_90_cav = _core._cav(comp_90, dt)
        comp_ver_cav = _core._cav(comp_ver, dt)
    else:
        with environment(RAYON_NUM_THREADS=str(cores)):
            comp_0_cav = _core._parallel_cav(comp_0, dt)
            comp_90_cav = _core._parallel_cav(comp_90, dt)
            comp_ver_cav = _core._parallel_cav(comp_ver, dt)

    return pd.DataFrame(
        {
            "000": comp_0_cav,
            "090": comp_90_cav,
            "ver": comp_ver_cav,
            "geom": np.sqrt(comp_0_cav * comp_90_cav),
        }
    )


def arias_intensity(
    waveform: ChunkedWaveformArray, dt: float, cores: int
) -> pd.DataFrame:
    """Compute Arias Intensity (AI) in m/s.

    Parameters
    ----------
    waveform : ChunkedWaveformArray
        Acceleration waveforms (g) with shape (n_components, n_stations, nt).
    dt : float
        Timestep resolution (s).
    cores : int
        Number of CPU cores for parallel processing.

    Returns
    -------
    pd.DataFrame
        AI values (m/s) for ['000', '090', 'ver', 'geom'].
    """
    comp_0 = waveform[Component.COMP_0]
    comp_90 = waveform[Component.COMP_90]
    comp_ver = waveform[Component.COMP_VER]

    if cores == 1:
        comp_0_ai = _core._arias_intensity(comp_0, dt)
        comp_90_ai = _core._arias_intensity(comp_90, dt)
        comp_ver_ai = _core._arias_intensity(comp_ver, dt)
    else:
        with environment(RAYON_NUM_THREADS=str(cores)):
            comp_0_ai = _core._parallel_arias_intensity(comp_0, dt)
            comp_90_ai = _core._parallel_arias_intensity(comp_90, dt)
            comp_ver_ai = _core._parallel_arias_intensity(comp_ver, dt)

    return pd.DataFrame(
        {
            "000": comp_0_ai,
            "090": comp_90_ai,
            "ver": comp_ver_ai,
            "geom": np.sqrt(comp_0_ai * comp_90_ai),
        }
    )


def ds575(waveform: ChunkedWaveformArray, dt: float, cores: int) -> pd.DataFrame:
    """Compute 5-75% Significant Duration (DS575) in seconds.

    Parameters
    ----------
    waveform : ChunkedWaveformArray
        Acceleration waveforms (g) with shape (n_components, n_stations, nt).
    dt : float
        Timestep resolution (s).
    cores : int
        Number of CPU cores for parallel processing.

    Returns
    -------
    pd.DataFrame
        Duration values (s) for ['000', '090', 'ver', 'geom'].
    """
    return significant_duration(waveform, dt, 5, 75, cores)


def ds595(waveform: ChunkedWaveformArray, dt: float, cores: int) -> pd.DataFrame:
    """Compute 5-95% Significant Duration (DS595) in seconds.

    Parameters
    ----------
    waveform : ChunkedWaveformArray
        Acceleration waveforms (g) with shape (n_components, n_stations, nt).
    dt : float
        Timestep resolution (s).
    cores : int
        Number of CPU cores for parallel processing.

    Returns
    -------
    pd.DataFrame
        Duration values (s) for ['000', '090', 'ver', 'geom'].
    """
    return significant_duration(waveform, dt, 5, 95, cores)
