"""Intensity Measure Implementations."""

import functools
import warnings
from collections.abc import Callable, Mapping, Sequence
from enum import IntEnum, StrEnum
from pathlib import Path

import numpy as np
import numpy.typing as npt
import scipy as sp
import xarray as xr
from pyfftw.interfaces import numpy_fft as fft

from IM import (
    _core,  # ty: ignore[unresolved-import]
    konno_ohmachi,
)

ChunkedWaveformArray = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]

Waveform = xr.DataArray | np.ndarray

WAVEFORM_DIMS = ("component", "station", "time")
ROTD_COMPONENTS = (
    "000",
    "090",
    "ver",
    "geom",
    "rotd0",
    "rotd50",
    "rotd100",
    "rotd0_orientation",
    "rotd100_orientation",
)
GEOM_COMPONENTS = ("000", "090", "ver", "geom")
FAS_COMPONENTS = ("000", "090", "ver", "geom", "eas")

DAMPING = 0.05
G = 981


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


def _as_waveform(waveform: Waveform) -> xr.DataArray:
    """Normalise a waveform into a DataArray with `component` and `time` dims.

    Parameters
    ----------
    waveform : Waveform
        Either a bare `(n_components, n_stations, nt)` ndarray, or an
        `xr.DataArray` with `component` and `time` dimensions.

    Returns
    -------
    xr.DataArray
        The waveform as a DataArray suitable to pass to `xr.apply_ufunc`.

    Raises
    ------
    TypeError
        If the waveform has the wrong number of dimensions, is missing the
        `component`/`time` dimensions, or does not have 3 components.
    """
    if not isinstance(waveform, xr.DataArray):
        array = np.asarray(waveform)
        if array.ndim != 3:
            raise TypeError(
                "Waveform must have shape (n_components, n_stations, nt), "
                f"but {array.shape=}"
            )
        waveform = xr.DataArray(array, dims=WAVEFORM_DIMS)

    missing = {"component", "time"}.difference(waveform.dims)
    if missing:
        raise TypeError(f"Waveform is missing dimensions {sorted(missing)}")
    if waveform.sizes["component"] != len(Component):
        raise TypeError(
            f"Waveform must have {len(Component)} components, "
            f"but {waveform.sizes['component']=}"
        )
    if waveform.chunks is not None:
        waveform = waveform.chunk({"component": -1, "time": -1})
    return waveform


def _components(block: np.ndarray) -> tuple[ChunkedWaveformArray, tuple[int, ...]]:
    """Split an `apply_ufunc` block into contiguous per-component matrices.

    Parameters
    ----------
    block : ndarray
        A block as received by an `apply_ufunc` kernel, of shape
        `(*lead, n_components, nt)`.

    Returns
    -------
    ChunkedWaveformArray
        The per-component matrices, contiguous float64, shape
        `(n_components, prod(lead), nt)`.
    tuple of int
        The original leading shape, to reshape kernel output back into.
    """
    n_components, nt = block.shape[-2:]
    lead = block.shape[:-2]
    components = np.ascontiguousarray(np.moveaxis(block, -2, 0), dtype=np.float64)
    return components.reshape(n_components, -1, nt), lead


def _im_dataset(
    kernel: Callable[..., np.ndarray],
    waveform: Waveform,
    components: Sequence[str],
    *,
    name: str,
    extra_dims: Mapping[str, npt.NDArray] | None = None,
    kwargs: Mapping[str, object] | None = None,
) -> xr.Dataset:
    """Run a per-station kernel over a waveform, one data variable per component.

    Parameters
    ----------
    kernel : callable
        Function to apply to each waveform block.
    waveform : Waveform
        The waveform to compute the IM for.
    components : sequence of str
        Names of the components the kernel produces, in output order.
    name : str
        Name recorded in `dataset.attrs["name"]`, identifying the IM. This is
        the only place the IM name is carried, since `components` become
        data variables rather than a `component` dimension.
    extra_dims : mapping of str to ndarray, optional
        Extra output dimensions the kernel introduces (e.g. `period` for pSA,
        `frequency` for FAS), mapping dimension name to coordinate values.
    kwargs : mapping, optional
        Extra keyword arguments passed through to `kernel`.

    Returns
    -------
    xr.Dataset
        One data variable per component, sharing the input's `station`
        dimension and any non-dimension coordinates (e.g. real station
        names, `latitude`, `longitude`).
    """
    extra_dims = extra_dims or {}
    kwargs = kwargs or {}
    waveform = _as_waveform(waveform)

    result = xr.apply_ufunc(
        kernel,
        waveform,
        input_core_dims=[["component", "time"]],
        output_core_dims=[[*extra_dims, "im_component"]],
        kwargs=dict(kwargs),
        keep_attrs=False,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={
            "output_sizes": {"im_component": len(components)}
            | {dim: len(values) for dim, values in extra_dims.items()}
        },
    )
    result = result.assign_coords(im_component=list(components), **extra_dims)
    dataset = result.to_dataset("im_component")
    dataset.attrs = {"name": name}
    return dataset


def _rotd_kernel(
    block: np.ndarray,
    *,
    transform: Callable[[ChunkedWaveformArray], ChunkedWaveformArray] | None = None,
) -> np.ndarray:
    """Kernel for `compute_intensity_measure_rotd`.

    Parameters
    ----------
    block : ndarray
        A `(*lead, n_components, nt)` waveform block.
    transform : callable, optional
        Applied to the `(3, n_rows, nt)` component matrices before taking
        peaks (e.g. integration for PGV/PGD). Must preserve the leading
        `(3, n_rows, ...)` shape.

    Returns
    -------
    ndarray
        A `(*lead, len(ROTD_COMPONENTS))` array.
    """
    components, lead = _components(block)
    if transform is not None:
        components = transform(components)
    comp_0, comp_90, comp_ver = components
    peak_0 = np.abs(comp_0).max(axis=-1)
    peak_90 = np.abs(comp_90).max(axis=-1)
    peak_ver = np.abs(comp_ver).max(axis=-1)
    # (rows, 5) = rotd0, rotd50, rotd100 then the RotD0 and RotD100 orientations.
    stats = _core._rotd(comp_0, comp_90)
    peaks = np.stack([peak_0, peak_90, peak_ver, np.sqrt(peak_0 * peak_90)], axis=-1)
    out = np.concatenate([peaks, stats], axis=-1)
    return out.reshape(lead + (len(ROTD_COMPONENTS),))


def compute_intensity_measure_rotd(
    waveforms: Waveform,
    name: str,
    *,
    transform: Callable[[ChunkedWaveformArray], ChunkedWaveformArray] | None = None,
) -> xr.Dataset:
    """Generic wrapper to compute peak values and RotD statistics for IMs.

    Parameters
    ----------
    waveforms : Waveform
        Waveform data with shape (n_components, n_stations, nt).
    name : str
        Name of the resulting dataset (recorded in `dataset.attrs["name"]`).
    transform : callable, optional
        Applied to the acceleration components before taking peaks (e.g.
        integration to velocity/displacement for PGV/PGD).

    Returns
    -------
    xr.Dataset
        One data variable per component in `ROTD_COMPONENTS`: peak values for
        `['000', '090', 'ver', 'geom', 'rotd0', 'rotd50', 'rotd100']`, then
        `rotd0_orientation` and `rotd100_orientation` holding the angle
        (degrees) at which RotD0 and RotD100 occur.
    """
    return _im_dataset(
        functools.partial(_rotd_kernel, transform=transform),
        waveforms,
        ROTD_COMPONENTS,
        name=name,
    )


def _velocity(components: ChunkedWaveformArray, dt: float) -> ChunkedWaveformArray:
    """Integrate acceleration (g) to velocity (cm/s).

    Parameters
    ----------
    components : ChunkedWaveformArray
        Per-component acceleration matrices, shape `(3, n_rows, nt)`.
    dt : float
        Timestep resolution (s).

    Returns
    -------
    ChunkedWaveformArray
        Velocity matrices (cm/s), shape `(3, n_rows, nt - 1)`.
    """
    return G * sp.integrate.cumulative_trapezoid(components, dx=dt, axis=-1)


def _displacement(components: ChunkedWaveformArray, dt: float) -> ChunkedWaveformArray:
    """Integrate acceleration (g) to displacement (cm).

    Parameters
    ----------
    components : ChunkedWaveformArray
        Per-component acceleration matrices, shape `(3, n_rows, nt)`.
    dt : float
        Timestep resolution (s).

    Returns
    -------
    ChunkedWaveformArray
        Displacement matrices (cm), shape `(3, n_rows, nt)`.
    """
    velocity = sp.integrate.cumulative_trapezoid(components, dx=dt, axis=-1, initial=0)
    # In-place multiplication to avoid yet another allocation
    np.multiply(G, velocity, out=velocity)
    return sp.integrate.cumulative_trapezoid(velocity, dx=dt, axis=-1, initial=0)


def peak_ground_acceleration(waveform: Waveform) -> xr.Dataset:
    """Compute Peak Ground Acceleration (PGA) in g.

    Parameters
    ----------
    waveform : Waveform
        Acceleration waveforms with shape (n_components, n_stations, nt).

    Returns
    -------
    xr.Dataset
        One data variable per component containing PGA values (g) for
        standard and rotated components.
    """
    return compute_intensity_measure_rotd(waveform, IM.PGA.value)


def peak_ground_velocity(waveform: Waveform, dt: float) -> xr.Dataset:
    """Compute Peak Ground Velocity (PGV) in cm/s via trapezoidal integration.

    Parameters
    ----------
    waveform : Waveform
        Acceleration waveforms (g) with shape (n_components, n_stations, nt).
    dt : float
        Timestep resolution (s).

    Returns
    -------
    xr.Dataset
        One data variable per component containing PGV values (cm/s) for
        standard and rotated components.
    """
    return compute_intensity_measure_rotd(
        waveform, IM.PGV.value, transform=functools.partial(_velocity, dt=dt)
    )


def peak_ground_displacement(waveform: Waveform, dt: float) -> xr.Dataset:
    """Compute Peak Ground Displacement (PGD) for waveforms.

    Parameters
    ----------
    waveform : Waveform
        Acceleration waveforms in g units.
    dt : float
        Timestep resolution of the waveform array.

    Returns
    -------
    xr.Dataset
        One data variable per component containing PGD values (cm) with
        rotated components.
    """
    return compute_intensity_measure_rotd(
        waveform, IM.PGD.value, transform=functools.partial(_displacement, dt=dt)
    )


def _cav_kernel(block: np.ndarray, *, dt: float, threshold: float | None) -> np.ndarray:
    """Kernel for `cumulative_absolute_velocity`.

    Parameters
    ----------
    block : ndarray
        A `(*lead, n_components, nt)` acceleration block (g).
    dt : float
        Timestep resolution (s).
    threshold : float or None
        Acceleration threshold ($cm/s^2$). Samples below it are zeroed
        before integrating. `None` or zero integrates the record as-is.

    Returns
    -------
    ndarray
        A `(*lead, len(GEOM_COMPONENTS))` array of CAV values (m/s).
    """
    components, lead = _components(block)
    if threshold:
        components = np.where(np.abs(components) < threshold / G, 0.0, components)
    comp_0, comp_90, comp_ver = components
    cav_0 = _core._cav(comp_0, dt)
    cav_90 = _core._cav(comp_90, dt)
    cav_ver = _core._cav(comp_ver, dt)
    out = np.stack([cav_0, cav_90, cav_ver, np.sqrt(cav_0 * cav_90)], axis=-1)
    return out.reshape(lead + (len(GEOM_COMPONENTS),))


def cumulative_absolute_velocity(
    waveform: Waveform,
    dt: float,
    threshold: float | None = None,
) -> xr.Dataset:
    """Compute Cumulative Absolute Velocity (CAV) in m/s.

    Parameters
    ----------
    waveform : Waveform
        Acceleration waveforms (g) with shape (n_components, n_stations, nt).
    dt : float
        Timestep resolution (s).
    threshold : float, optional
        Acceleration threshold ($cm/s^2$). Values below this are ignored (e.g. 5 for CAV5).

    Returns
    -------
    xr.Dataset
        One data variable per component (`attrs["name"]` is `CAV5` if
        `threshold` is set, else `CAV`) containing CAV values (m/s) for
        ['000', '090', 'ver', 'geom'].
    """
    name = IM.CAV5.value if threshold else IM.CAV.value
    return _im_dataset(
        _cav_kernel,
        waveform,
        GEOM_COMPONENTS,
        name=name,
        kwargs={"dt": dt, "threshold": threshold},
    )


def _arias_kernel(block: np.ndarray, *, dt: float) -> np.ndarray:
    """Kernel for `arias_intensity`.

    Parameters
    ----------
    block : ndarray
        A `(*lead, n_components, nt)` acceleration block (g).
    dt : float
        Timestep resolution (s).

    Returns
    -------
    ndarray
        A `(*lead, len(GEOM_COMPONENTS))` array of Arias intensities (m/s).
    """
    components, lead = _components(block)
    comp_0, comp_90, comp_ver = components
    ai_0 = _core._arias_intensity(comp_0, dt)
    ai_90 = _core._arias_intensity(comp_90, dt)
    ai_ver = _core._arias_intensity(comp_ver, dt)
    out = np.stack([ai_0, ai_90, ai_ver, np.sqrt(ai_0 * ai_90)], axis=-1)
    return out.reshape(lead + (len(GEOM_COMPONENTS),))


def arias_intensity(waveform: Waveform, dt: float) -> xr.Dataset:
    """Compute Arias Intensity (AI) in m/s.

    Parameters
    ----------
    waveform : Waveform
        Acceleration waveforms (g) with shape (n_components, n_stations, nt).
    dt : float
        Timestep resolution (s).

    Returns
    -------
    xr.Dataset
        One data variable per component containing AI values (m/s) for
        ['000', '090', 'ver', 'geom'].
    """
    return _im_dataset(
        _arias_kernel, waveform, GEOM_COMPONENTS, name=IM.AI.value, kwargs={"dt": dt}
    )


def _duration_kernel(
    block: np.ndarray, *, dt: float, quantile_low: float, quantile_high: float
) -> np.ndarray:
    """Kernel for `significant_duration`.

    Parameters
    ----------
    block : ndarray
        A `(*lead, n_components, nt)` acceleration block (g).
    dt : float
        Timestep resolution (s).
    quantile_low : float
        Lower bound of the Arias intensity accumulation window, as a
        fraction of the total (e.g. 0.05 for Ds595).
    quantile_high : float
        Upper bound of that window (e.g. 0.95 for Ds595).

    Returns
    -------
    ndarray
        A `(*lead, len(GEOM_COMPONENTS))` array of durations (s).
    """
    components, lead = _components(block)
    comp_0, comp_90, comp_ver = components
    duration_0 = _core._significant_duration(comp_0, dt, quantile_low, quantile_high)
    duration_90 = _core._significant_duration(comp_90, dt, quantile_low, quantile_high)
    duration_ver = _core._significant_duration(
        comp_ver, dt, quantile_low, quantile_high
    )
    geom = np.sqrt(duration_0 * duration_90)
    out = np.stack([duration_0, duration_90, duration_ver, geom], axis=-1)
    return out.reshape(lead + (len(GEOM_COMPONENTS),))


def significant_duration(
    waveforms: Waveform,
    dt: float,
    percent_low: float,
    percent_high: float,
    name: str = "duration",
) -> xr.Dataset:
    """Compute significant duration based on Arias Intensity accumulation.

    Parameters
    ----------
    waveforms : Waveform
        Acceleration waveforms (g) with shape (n_components, n_stations, nt).
    dt : float
        Timestep resolution (s).
    percent_low : float
        Lower bound percentage (e.g., 5.0 for 5%).
    percent_high : float
        Upper bound percentage (e.g., 95.0 for 95%).
    name : str, optional
        Name of the resulting dataset.

    Returns
    -------
    xr.Dataset
        One data variable per component containing the significant duration
        (s) for ['000', '090', 'ver', 'geom'].
    """
    return _im_dataset(
        _duration_kernel,
        waveforms,
        GEOM_COMPONENTS,
        name=name,
        kwargs={
            "dt": dt,
            "quantile_low": percent_low / 100,
            "quantile_high": percent_high / 100,
        },
    )


def ds575(waveform: Waveform, dt: float) -> xr.Dataset:
    """Compute 5-75% Significant Duration (DS575) in seconds.

    Parameters
    ----------
    waveform : Waveform
        Acceleration waveforms (g) with shape (n_components, n_stations, nt).
    dt : float
        Timestep resolution (s).

    Returns
    -------
    xr.Dataset
        One data variable per component containing duration values (s) for
        ['000', '090', 'ver', 'geom'].
    """
    return significant_duration(waveform, dt, 5, 75, IM.Ds575.value)


def ds595(waveform: Waveform, dt: float) -> xr.Dataset:
    """Compute 5-95% Significant Duration (DS595) in seconds.

    Parameters
    ----------
    waveform : Waveform
        Acceleration waveforms (g) with shape (n_components, n_stations, nt).
    dt : float
        Timestep resolution (s).

    Returns
    -------
    xr.Dataset
        One data variable per component containing duration values (s) for
        ['000', '090', 'ver', 'geom'].
    """
    return significant_duration(waveform, dt, 5, 95, IM.Ds595.value)


def _psa_kernel(
    block: np.ndarray,
    *,
    periods: npt.NDArray[np.float64],
    dt: float,
) -> np.ndarray:
    """Kernel for `pseudo_spectral_acceleration`.

    Parameters
    ----------
    block : np.ndarray
        A block of stations to solve pSA for.
    periods : np.ndarray of float64
        Periods to solve pSA with.
    dt : float
        Shared station timestep.

    Returns
    -------
    np.ndarray
        A chunk of solved pSA values.
    """
    (comp_0, comp_90, comp_ver), lead = _components(block)
    psa = _core._psa(comp_0, comp_90, comp_ver, periods, dt, DAMPING)
    return psa.reshape(lead + (len(periods), len(ROTD_COMPONENTS)))


def pseudo_spectral_acceleration(
    waveforms: Waveform,
    periods: npt.ArrayLike,
    dt: float,
) -> xr.Dataset:
    """Compute pseudo-spectral acceleration (PSA) statistics.

    Calculates PSA for single-degree-of-freedom oscillators across various
    periods using the Newmark-beta method and computes rotated (RotD) statistics.

    Parameters
    ----------
    waveforms : Waveform
        Acceleration waveforms (g) with shape (n_components, n_stations, nt).
    periods : array_like
        Natural periods of the oscillators (s).
    dt : float
        Timestep resolution of the waveforms (s).

    Returns
    -------
    xr.Dataset
        One data variable per component in `ROTD_COMPONENTS`, each with a
        `period` dimension: PSA for
        ['000', '090', 'ver', 'geom', 'rotd0', 'rotd50', 'rotd100'], then
        `rotd0_orientation` and `rotd100_orientation` holding the angle
        (degrees) at which RotD0 and RotD100 occur.
    """
    periods = np.asarray(periods, dtype=np.float64)
    return _im_dataset(
        _psa_kernel,
        waveforms,
        ROTD_COMPONENTS,
        name=IM.pSA.value,
        extra_dims={"period": periods},
        kwargs={"periods": periods, "dt": dt},
    )


def _interpolate(
    smoothed: np.ndarray,
    fa_frequencies: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
) -> np.ndarray:
    """Interpolate a smoothed spectrum onto the requested frequencies.

    Parameters
    ----------
    smoothed : ndarray
        Smoothed spectrum values, shape `(..., len(fa_frequencies))`.
    fa_frequencies : ndarray of float64
        The `rfft` bin frequencies the spectrum is defined on (Hz).
    freqs : ndarray of float64
        Frequencies to interpolate onto (Hz).

    Returns
    -------
    ndarray
        The spectrum at `freqs`, shape `(..., len(freqs))`.
    """
    interpolator = sp.interpolate.make_interp_spline(
        fa_frequencies, smoothed, axis=-1, k=1
    )
    return interpolator(freqs)


def _fas_kernel(
    block: np.ndarray,
    *,
    dt: float,
    n_fft: int,
    freqs: npt.NDArray[np.float64],
    fa_frequencies: npt.NDArray[np.float64],
    bandwidth: float,
    scratch_directory: Path | None,
) -> np.ndarray:
    """Kernel for `fourier_amplitude_spectra`.

    Parameters
    ----------
    block : ndarray
        A `(*lead, n_components, nt)` acceleration block (g).
    dt : float
        Timestep resolution (s).
    n_fft : int
        Length the record is zero-padded to before the real FFT.
    freqs : ndarray of float
        Output frequencies (Hz) the smoothed spectrum is interpolated onto.
    fa_frequencies : ndarray of float
        The `rfft` bin frequencies (Hz) the smoothed spectrum is defined on.
    bandwidth : float
        Bandwidth of the Konno-Ohmachi smoothing window.
    scratch_directory : Path or None
        Where to build a Konno-Ohmachi matrix too large to hold in memory.

    Returns
    -------
    ndarray
        A `(*lead, len(freqs), len(FAS_COMPONENTS))` array of Fourier
        amplitudes, the last component being EAS.
    """
    components, lead = _components(block)
    n_components, rows, _ = components.shape
    n_fa = len(fa_frequencies)

    spectra = np.empty((n_components, rows, n_fa), dtype=np.float64)
    for index in range(n_components):
        spectra[index] = np.abs(fft.rfft(components[index], n=n_fft, axis=-1) * dt)

    # EAS is computed from the *unsmoothed* spectrum to avoid distortion of
    # inter-frequency correlations, then smoothed alongside 000/090/ver in a
    # single pass over the (potentially huge) Konno matrix.
    eas_unsmoothed = np.sqrt(
        0.5
        * (np.square(spectra[Component.COMP_0]) + np.square(spectra[Component.COMP_90]))
    )
    spectra_and_eas = np.concatenate([spectra, eas_unsmoothed[np.newaxis]], axis=0)

    smoothed = _interpolate(
        konno_ohmachi.smooth(spectra_and_eas, bandwidth, scratch_directory),
        fa_frequencies,
        freqs,
    )

    geom = np.sqrt(smoothed[Component.COMP_0] * smoothed[Component.COMP_90])
    out = np.stack([smoothed[0], smoothed[1], smoothed[2], geom, smoothed[3]], axis=-1)
    return out.reshape(lead + (len(freqs), len(FAS_COMPONENTS)))


def fourier_amplitude_spectra(
    waveforms: Waveform,
    dt: float,
    freqs: npt.NDArray[np.float64],
    bandwidth: float = konno_ohmachi.DEFAULT_BANDWIDTH,
    scratch_directory: Path | None = None,
) -> xr.Dataset:
    """Compute Fourier Amplitude Spectrum (FAS) of seismic waveforms.

    Parameters
    ----------
    waveforms : Waveform
        Waveform array (g) with shape `(n_components, n_stations, n_timesteps)`.
    dt : float
        Timestep resolution of the waveforms (s).
    freqs : ndarray of float64
        Frequencies at which to compute FAS (Hz).
    bandwidth : float, optional
        Bandwidth of the Konno-Ohmachi smoothing window. Lower values smooth
        more strongly.
    scratch_directory : Path, optional
        Where to build a Konno-Ohmachi matrix too large to hold in memory.
        Defaults to `$IM_CALCULATION_SCRATCH_DIR`, else the platform temporary
        directory.

    Returns
    -------
    xr.Dataset
        One data variable per component, each with a `frequency` dimension,
        containing FAS values for ['000', '090', 'ver', 'geom', 'eas'].
    """
    waveform = _as_waveform(waveforms)

    nyquist_frequency = 1 / (2 * dt)
    max_frequency = freqs.max()
    if max_frequency > nyquist_frequency:
        warnings.warn(
            RuntimeWarning(
                f"Attempting to compute FAS for frequencies above Nyquist frequency {nyquist_frequency:.2e} Hz. Results only include frequencies at or below Nyquist frequency {nyquist_frequency:.2e} Hz."
            ),
        )
        freqs = freqs[freqs <= nyquist_frequency]

    n_fft = 2 ** int(np.ceil(np.log2(waveform.sizes["time"])))
    fa_frequencies = np.fft.rfftfreq(n_fft, dt)

    return _im_dataset(
        _fas_kernel,
        waveform,
        FAS_COMPONENTS,
        name=IM.FAS.value,
        extra_dims={"frequency": freqs},
        kwargs={
            "dt": dt,
            "n_fft": n_fft,
            "freqs": freqs,
            "fa_frequencies": fa_frequencies,
            "bandwidth": bandwidth,
            "scratch_directory": scratch_directory,
        },
    )
