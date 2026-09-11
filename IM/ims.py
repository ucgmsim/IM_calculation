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
    ko_matrices,
)

# Concrete (n_components, n_stations, nt) block, as seen inside a kernel once
# `_components` has flattened any leading (broadcast) axes into `n_stations`.
ChunkedWaveformArray = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]

# A (component, station, time) waveform. `_as_waveform` wraps a bare ndarray
# eagerly, and returns a dask-backed DataArray still unevaluated.
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
    "rotd50_orientation",
    "rotd100_orientation",
)
GEOM_COMPONENTS = ("000", "090", "ver", "geom")
FAS_COMPONENTS = ("000", "090", "ver", "geom", "eas")

DAMPING = 0.05
G = 981
# Bounds how much of a (possibly multi-gigabyte, float32, memmapped) Konno
# matrix gets promoted to float64 at once by `_konno_smooth`.
KONNO_BLOCK_BYTES = 64 * 2**20


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

    A bare `(n_components, n_stations, nt)` ndarray becomes an eager
    DataArray with dims `("component", "station", "time")`. For a dask-backed
    DataArray this rechunks `component` and `time` (the core dimensions every
    kernel operates on) into one chunk each, and preserves whatever `station`
    chunking the input had.

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
        If the waveform has the wrong number of dimensions, or lacks either
        the `component` and `time` dimensions or a count of 3 components.
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

    `apply_ufunc` moves core dimensions to the end, so `block` arrives as
    `(*lead, n_components, nt)`, where `lead` is whatever loop dimensions the
    input had (normally just `station`, but there may be none or several).
    Moving the component axis to the front and forcing a contiguous float64
    copy collapses `lead` into one row axis, so `components[i]` is a
    contiguous `(n_rows, nt)` matrix, which is what every `_core` kernel
    expects.
    Callers restore the original leading shape with `out.reshape(lead + (...))`.

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

    The kernel receives a `(*lead, n_components, nt)` block (see
    `_components`) and must return a `(*lead, *extra_sizes, len(components))`
    array. Unstacking the trailing component axis gives a `Dataset` with one
    variable per component, lazy if the waveform was lazy.

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
        the only place that records the IM name, since `components` become
        data variables rather than a `component` dimension.
    extra_dims : mapping of str to ndarray, optional
        Extra output dimensions the kernel introduces (`period` for pSA,
        `frequency` for FAS), mapping dimension name to coordinate values.
    kwargs : mapping, optional
        Extra keyword arguments passed through to `kernel`.

    Returns
    -------
    xr.Dataset
        One data variable per component, sharing the input's `station`
        dimension and any non-dimension coordinates (real station names,
        `latitude`, `longitude`).
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
        peaks (integration for PGV/PGD). Must preserve the leading
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
    # (rows, 6) = rotd0, rotd50, rotd100 then their three orientations.
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
        Applied to the acceleration components before taking peaks:
        integration to velocity or displacement for PGV and PGD.

    Returns
    -------
    xr.Dataset
        One data variable per component in `ROTD_COMPONENTS`: peak values for
        `['000', '090', 'ver', 'geom', 'rotd0', 'rotd50', 'rotd100']`, then
        `rotd0_orientation`, `rotd50_orientation` and `rotd100_orientation`
        holding the angle (degrees) at which each RotD statistic occurs.
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
        Acceleration threshold ($cm/s^2$). The kernel zeroes samples below
        it before integrating. `None` or zero integrates the record as-is.

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
        Acceleration threshold ($cm/s^2$), 5 for CAV5. The calculation
        ignores values below it.

    Returns
    -------
    xr.Dataset
        One data variable per component (`attrs["name"]` becomes `CAV5` with
        a `threshold`, else `CAV`) containing CAV values (m/s) for
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
        fraction of the total (0.05 for Ds595).
    quantile_high : float
        Upper bound of that window (0.95 for Ds595).

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
        Lower bound percentage, 5.0 for 5%.
    percent_high : float
        Upper bound percentage, 95.0 for 95%.
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


N_ROTD180_ANGLES = 180
ROTD180_ANGLES = np.arange(N_ROTD180_ANGLES)

# Column layout of an `_core._psa_rotd180` row: the 180 rotated peaks, then
# the exact unrotated 000 and 090 peaks, then the six RotD statistics.
PSA_PEAK_COLUMNS = slice(N_ROTD180_ANGLES, N_ROTD180_ANGLES + 2)
PSA_STATS_COLUMNS = slice(N_ROTD180_ANGLES + 2, N_ROTD180_ANGLES + 8)


def _psa_kernel(
    block: np.ndarray,
    *,
    periods: npt.NDArray[np.float64],
    dt: float,
    full_rotd180: bool,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Kernel for `pseudo_spectral_acceleration`.

    Loops over periods internally (rather than treating `period` as a
    broadcast input) so every IM kernel shares one contract: `(*lead,
    n_components, nt) -> (*lead, *extra, k)`. Station-chunk parallelism is
    already ample, so the period loop here runs at no extra cost.

    With `full_rotd180`, this also returns the full 180-angle RotD curve
    behind the summary statistics rather than discarding it, so the
    Newmark-beta solve never runs twice for the same (period, station chunk).

    Parameters
    ----------
    block : ndarray
        A `(*lead, n_components, nt)` acceleration block (g).
    periods : ndarray of float
        Oscillator periods (s) to solve for.
    dt : float
        Timestep resolution (s).
    full_rotd180 : bool
        Whether to also return the 180-angle RotD curve.

    Returns
    -------
    ndarray
        A `(*lead, len(periods), len(ROTD_COMPONENTS))` array of pseudo
        spectral accelerations (g).
    ndarray
        Only with `full_rotd180`: the `(*lead, len(periods),
        N_ROTD180_ANGLES)` rotated-peak curve those statistics came from.
    """
    components, lead = _components(block)
    comp_0, comp_90, comp_ver = components
    rows = comp_0.shape[0]
    out = np.empty((rows, len(periods), len(ROTD_COMPONENTS)), dtype=np.float64)
    rotd180 = (
        np.empty((rows, len(periods), N_ROTD180_ANGLES), dtype=np.float64)
        if full_rotd180
        else None
    )
    for index, period in enumerate(periods):
        w = 2 * np.pi / period
        # (rows, 188): 180 rotated peaks, the exact 000 and 090 peaks, then
        # the RotD statistics row. The statistics come back from rust rather
        # than off the curve because RotD0 and RotD100 come off the hull
        # geometry, which only exists inside that call.
        psa = _core._psa_rotd180(comp_0, comp_90, dt, w, DAMPING)
        stats = psa[:, PSA_STATS_COLUMNS]
        peak_0, peak_90 = psa[:, PSA_PEAK_COLUMNS].T
        peak_ver = _core._psa_peak(comp_ver, dt, w, DAMPING)
        peaks = np.stack(
            [peak_0, peak_90, peak_ver, np.sqrt(peak_0 * peak_90)], axis=-1
        )
        out[:, index] = np.concatenate([peaks, stats], axis=-1)
        if rotd180 is not None:
            rotd180[:, index] = psa[:, :N_ROTD180_ANGLES]

    out = out.reshape(lead + (len(periods), len(ROTD_COMPONENTS)))
    if rotd180 is None:
        return out
    return out, rotd180.reshape(lead + (len(periods), N_ROTD180_ANGLES))


def pseudo_spectral_acceleration(
    waveforms: Waveform,
    periods: npt.ArrayLike,
    dt: float,
    full_rotd180: bool = False,
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
    full_rotd180 : bool, optional
        If set, also include a `rotd180` data variable with an extra `angle`
        dimension (0..179 degrees), holding pSA (g) at every rotation angle.
        This reuses the same Newmark-beta solve already run for the summary
        statistics.

    Returns
    -------
    xr.Dataset
        One data variable per component, each with a `period` dimension:
        PSA for
        ['000', '090', 'ver', 'geom', 'rotd0', 'rotd50', 'rotd100'], then
        `rotd0_orientation`, `rotd50_orientation` and `rotd100_orientation`
        holding the angle (degrees) at which each RotD statistic occurs. If
        you pass `full_rotd180`, also a `rotd180` variable with dims
        (..., period, angle).
    """
    periods = np.asarray(periods, dtype=np.float64)
    waveform = _as_waveform(waveforms)
    kernel = functools.partial(
        _psa_kernel, periods=periods, dt=dt, full_rotd180=full_rotd180
    )

    output_core_dims = [["period", "im_component"]]
    output_sizes = {"period": len(periods), "im_component": len(ROTD_COMPONENTS)}
    if full_rotd180:
        output_core_dims.append(["period", "angle"])
        output_sizes["angle"] = N_ROTD180_ANGLES

    outputs = xr.apply_ufunc(
        kernel,
        waveform,
        input_core_dims=[["component", "time"]],
        output_core_dims=output_core_dims,
        keep_attrs=False,
        dask="parallelized",
        output_dtypes=[np.float64] * len(output_core_dims),
        dask_gufunc_kwargs={"output_sizes": output_sizes},
    )
    summary, rotd180 = outputs if full_rotd180 else (outputs, None)

    summary = summary.assign_coords(im_component=list(ROTD_COMPONENTS), period=periods)
    dataset = summary.to_dataset("im_component")
    dataset.attrs = {"name": IM.pSA.value}
    if rotd180 is not None:
        dataset["rotd180"] = rotd180.assign_coords(period=periods, angle=ROTD180_ANGLES)
    return dataset


def _konno_smooth(spectrum_data: np.ndarray, konno: np.ndarray) -> np.ndarray:
    """Multiply a spectrum by a Konno-Ohmachi matrix, a block of columns at a time.

    `konno` is a float32 memmap of up to tens of gigabytes. Writing
    `spectrum_data @ konno` directly makes numpy promote the *entire* matrix
    to float64 before the product. Taking a block of output columns at a
    time bounds the promoted array to `KONNO_BLOCK_BYTES` while leaving each
    output element one full-length float64 accumulation, so the result is
    the same product, not an approximation of it. The blocking divides the
    output columns only, never the contraction axis (the matrix's rows).

    Parameters
    ----------
    spectrum_data : ndarray
        Spectrum values, shape `(..., n_fa)`.
    konno : ndarray
        Konno-Ohmachi smoothing matrix, shape `(n_fa, n_fa)`.

    Returns
    -------
    ndarray
        Smoothed spectrum, shape `(..., n_fa)`.
    """
    n_output = konno.shape[1]
    columns = max(1, KONNO_BLOCK_BYTES // (konno.shape[0] * np.float64().itemsize))
    smoothed = np.empty(spectrum_data.shape[:-1] + (n_output,), dtype=np.float64)
    for start in range(0, n_output, columns):
        block = slice(start, start + columns)
        smoothed[..., block] = spectrum_data @ np.asarray(
            konno[:, block], dtype=np.float64
        )
    return smoothed


def smooth_and_interpolate(
    spectrum_data: np.ndarray,
    konno: np.ndarray,
    freqs: npt.NDArray[np.float64],
    fa_frequencies: np.ndarray,
) -> np.ndarray:
    """
    Smooths and interpolates a spectrum.

    Parameters
    ----------
    spectrum_data : ndarray
        The spectrum data to smooth and interpolate.
    konno : ndarray
        The Konno-Ohmachi smoothing matrix to apply to the spectrum data.
    freqs : ndarray of float64
        The frequencies at which to interpolate the smoothed spectrum data.
    fa_frequencies : ndarray
        The original frequencies corresponding to the spectrum data before smoothing.

    Returns
    -------
    ndarray
        The smoothed and interpolated spectrum data at the specified frequencies.
    """
    smoothed = _konno_smooth(spectrum_data, konno)
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
    ko_directory: Path,
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
        Output frequencies (Hz) for the interpolated spectrum.
    fa_frequencies : ndarray of float
        The `rfft` bin frequencies (Hz) that size the Konno-Ohmachi matrix.
    ko_directory : Path
        Directory holding the cached Konno-Ohmachi matrices.

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

    # EAS comes from the *unsmoothed* spectrum, to avoid distorting the
    # inter-frequency correlations, and then smooths alongside 000/090/ver in
    # one pass over the (potentially huge) Konno matrix.
    eas_unsmoothed = np.sqrt(
        0.5
        * (np.square(spectra[Component.COMP_0]) + np.square(spectra[Component.COMP_90]))
    )
    spectra_and_eas = np.concatenate([spectra, eas_unsmoothed[np.newaxis]], axis=0)

    konno = ko_matrices.get_konno_matrix(n_fa, ko_directory)
    smoothed = smooth_and_interpolate(spectra_and_eas, konno, freqs, fa_frequencies)

    geom = np.sqrt(smoothed[Component.COMP_0] * smoothed[Component.COMP_90])
    out = np.stack([smoothed[0], smoothed[1], smoothed[2], geom, smoothed[3]], axis=-1)
    return out.reshape(lead + (len(freqs), len(FAS_COMPONENTS)))


def fourier_amplitude_spectra(
    waveforms: Waveform,
    dt: float,
    freqs: npt.NDArray[np.float64],
    ko_directory: Path,
) -> xr.Dataset:
    """Compute Fourier Amplitude Spectrum (FAS) of seismic waveforms.

    An FFT gives the FAS, which the Konno-Ohmachi algorithm then smooths.

    Parameters
    ----------
    waveforms : Waveform
        Waveform array (g) with shape `(n_components, n_stations, n_timesteps)`.
    dt : float
        Timestep resolution of the waveforms (s).
    freqs : ndarray of float64
        Frequencies at which to compute FAS (Hz).
    ko_directory : Path
        Directory containing precomputed Konno-Ohmachi matrices.

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
            "ko_directory": ko_directory,
        },
    )
