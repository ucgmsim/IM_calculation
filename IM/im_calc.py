"""Intensity Measure Calculation.

Description
-----------
Calculate intensity measures from broadband waveform files.

Inputs
------
A realisation file containing metadata configuration.

Typically, this information comes from a stage like [NSHM To Realisation](#nshm-to-realisation).

Outputs
-------
A CSV containing intensity measure summary statistics.


Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `im-calc` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`im-calc [OPTIONS] REALISATION_FFP BROADBAND_SIMULATION_FFP OUTPUT_PATH`

For More Help
-------------
See the output of `im-calc --help`.
"""

import gc
import multiprocessing
import sys
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Optional

import h5py
import numba
import numexpr as ne
import numpy as np
import numpy.typing as npt
import pandas as pd
import pykooh
import scipy as sp
import tqdm
import typer

from workflow import log_utils, utils
from workflow.realisations import (
    BroadbandParameters,
    IntensityMeasureCalculationParameters,
    RealisationMetadata,
)

app = typer.Typer()


class ComponentWiseOperation(StrEnum):
    """Component-wise numexpr accelerated operations."""

    NONE = "cos(theta) * comp_0 + sin(theta) * comp_90"
    ABS = "abs(cos(theta) * comp_0 + sin(theta) * comp_90)"
    SQUARE = "(cos(theta) * comp_0 + sin(theta) * comp_90)*(cos(theta) * comp_0 + sin(theta) * comp_90)"


@numba.njit
def newmark_estimate_psa(
    waveforms: npt.NDArray[np.float32],
    t: npt.NDArray[np.float32],
    dt: float,
    w: npt.NDArray[np.float32],
    xi: np.float32 = np.float32(0.05),
    gamma: np.float32 = np.float32(1 / 2),
    beta: np.float32 = np.float32(1 / 4),
    m: np.float32 = np.float32(1),
) -> npt.NDArray[np.float32]:
    """Compute pSA with the Newmark-beta method.

    Parameters
    ----------
    waveforms : npt.NDArray[np.float32]
        An array of shape (number_of_stations x number_of_timesteps).
    t : npt.NDArray[np.float32]
        The t-values of waveforms.
    dt : float
        The timestep.
    w : npt.NDArray[np.float32]
        The angular frequency of the SDOF oscillators.
    xi : np.float32
        The damping coefficient.
    gamma : np.float32
        The gamma-parameter of the Newmark method.
    beta : np.float32
        The beta-parameter of the Newmark method.
    m : np.float32
        The m parameter of the Newmark method.

    Returns
    -------
    npt.NDArray[np.float32]
        A response curve for the SDOF oscillators with natural frequencies `w`.

    See Also
    --------
    [0]: https://en.wikipedia.org/wiki/Newmark-beta_method
    [1]: The course notes on newmark-beta method: ENCI335 (Structural Analysis), Chapter 4
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


def rotd_psa_values(
    comp_000: npt.NDArray[np.float32],
    comp_090: npt.NDArray[np.float32],
    w: npt.NDArray[np.float32],
    step: int,
) -> npt.NDArray[np.float32]:
    """Compute the rotated pSA statistics.

    Parameters
    ----------
    comp_000 : npt.NDArray[np.float32]
        The pseudo-spectral acceleration in the 000 component (in cm/s^2).
    comp_090 : npt.NDArray[np.float32]
        The pseudo-spectral acceleration in the 090 component (in cm/s^2).
    w : npt.NDArray[np.float32]
        The natural angular frequency of the oscillators.
    step : int
        The number of stations to process in parallel at a single time.

    Returns
    -------
    npt.NDArray[np.float32]
        The median and maximum pSA values for all rotation between 0
        and 180 degrees, for every station, and every oscillator
        frequency. Has shape station count x period count x 2 (0 =
        rotd50, 1 = rotd100).
    """
    theta = np.linspace(0, np.pi, num=180, dtype=np.float32)
    psa = np.zeros((comp_000.shape[0], comp_000.shape[-1], 2), np.float32)
    out = np.zeros((step, *comp_000.shape[1:], 180), np.float32)
    logger = log_utils.get_logger("im_calc")
    logger.info(
        log_utils.structured_log("psa rotd buffer size", memory=sys.getsizeof(out))
    )
    w2 = np.square(w)

    for i in tqdm.trange(0, comp_000.shape[0], step):
        step_000 = comp_000[i : i + step]
        step_090 = comp_000[i : i + step]
        psa[i : i + step] = np.transpose(
            np.percentile(
                np.max(
                    ne.evaluate(
                        "abs(comp_000 * cos(theta) + comp_090 * sin(theta))",
                        {
                            "comp_000": step_000[..., np.newaxis],
                            "theta": theta[np.newaxis, ...],
                            "comp_090": step_090[..., np.newaxis],
                        },
                        out=out[: len(step_000)],
                    )[: len(step_000)],
                    axis=1,
                ),
                [50, 100],
                axis=-1,
            ),
            [1, 2, 0],
        )

    del out
    gc.collect()  # This is required because Python's GC is too lazy to remove the out array when it should
    return w2[np.newaxis, :, np.newaxis] * psa


def compute_psa(
    stations: pd.Series,
    waveforms: npt.NDArray[np.float32],
    periods: npt.NDArray[np.float32],
    dt: float,
    psa_rotd_maximum_memory_allocation: Optional[float] = None,
) -> pd.DataFrame:
    """Compute pSA statistics for all waveforms.

    Parameters
    ----------
    stations : pd.Series
        The list of stations.
    waveforms : npt.NDArray[np.float32]
        The list of acceleration waveforms.
    periods : npt.NDArray[np.float32]
        The list of periods to compute pSA for. The periods correspond
        to the natural angular frequency of the single degree of
        freedom oscillators (SDOFs) used to compute pSA.
    dt : float
        The timestep resolution of the waveform array.

    Returns
    -------
    pd.DataFrame
        A dataframe containing pSA for each period and station.
    """
    t = np.arange(waveforms.shape[1]) * dt
    w = 2 * np.pi / periods

    logger = log_utils.get_logger("im_calc")
    logger.info(
        log_utils.structured_log(
            "about to compute newmark psa repsonse curves",
            waveforms_shape=waveforms.shape,
            t_shape=t.shape,
            dt=dt,
            w=w,
        )
    )
    comp_0 = newmark_estimate_psa(
        waveforms[:, :, 1],
        t,
        dt,
        w,
    )

    comp_90 = newmark_estimate_psa(
        waveforms[:, :, 0],
        t,
        dt,
        w,
    )
    # Step size is either the CPU count, or the maximum number of
    # steps that fits within the psa_rotd_maximum_memory_allocation.
    if psa_rotd_maximum_memory_allocation:
        step = min(
            utils.get_available_cores(),
            int(psa_rotd_maximum_memory_allocation / (180 * sys.getsizeof(comp_0[0]))),
        )
    else:
        step = multiprocessing.cpu_count()
    if step < 1:
        raise ValueError(
            "PSA rotd memory allocation is too small (cannot even calculate a single station's pSA)."
        )
    logger.info(log_utils.structured_log("calculated", step=step))
    rotd_psa = rotd_psa_values(comp_0, comp_90, w, step=step)

    conversion_factor = np.square(w)[np.newaxis, :]
    comp_0_psa = conversion_factor * np.abs(comp_0).max(axis=1)
    comp_90_psa = conversion_factor * np.abs(comp_90).max(axis=1)
    ver_psa = conversion_factor * np.abs(
        newmark_estimate_psa(waveforms[:, :, 2], t, dt, w)
    ).max(axis=1)
    geom_psa = np.sqrt(comp_0_psa * comp_90_psa)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "station": stations,
                    "intensity_measure": f"pSA_{p:.2f}",
                    "000": comp_0_psa[:, i],
                    "090": comp_90_psa[:, i],
                    "ver": ver_psa[:, i],
                    "geom": geom_psa[:, i],
                    "rotd50": rotd_psa[:, i, 0],
                    "rotd100": rotd_psa[:, i, 1],
                }
            )
            for i, p in enumerate(periods)
        ]
    )


def compute_in_rotations(
    waveforms: npt.NDArray[np.float32],
    function: Callable,
    component_wise_operation: ComponentWiseOperation | str = ComponentWiseOperation.ABS,
) -> pd.DataFrame:
    """Compute a the rotated intensity measure statistics for a list of waveforms.

    Parameters
    ----------
    waveforms : npt.NDArray[np.float32]
        The acceleration waveforms (in cm/s^2).
    function : Callable
        The intensity measure function to evaluate.
    component_wise_operation : ComponentWiseOperation | str
        The component wise operation to evaluate of each waveform. Will be computed in parallel.


    Returns
    -------
    pd.DataFrame
        A dataframe containing the intensity measure statistics.

    """
    (stations, nt, _) = waveforms.shape
    values = np.zeros(shape=(stations, 180), dtype=waveforms.dtype)

    comp_0 = waveforms[:, :, 1]
    comp_90 = waveforms[:, :, 0]
    for i in range(180):
        theta = np.deg2rad(i).astype(np.float32)

        array = ne.evaluate(component_wise_operation)
        values[:, i] = function(array)

    comp_0 = values[:, 0]
    comp_90 = values[:, 90]
    comp_ver = waveforms[:, :, 2]
    match component_wise_operation:
        case ComponentWiseOperation.ABS:
            comp_ver = ne.evaluate("abs(comp_ver)")
        case ComponentWiseOperation.SQUARE:
            comp_ver = ne.evaluate("(comp_ver)**2")
    ver = function(comp_ver)
    rotated_max = np.max(values, axis=1)
    rotated_median = np.median(values, axis=1)
    return pd.DataFrame(
        {
            "000": comp_0,
            "090": comp_90,
            "ver": ver,
            "geom": np.sqrt(comp_0 * comp_90),
            "rotd100": rotated_max,
            "rotd50": rotated_median,
        }
    )


@numba.njit(parallel=True)
def trapz(waveforms: npt.NDArray[np.float32], dt: float) -> npt.NDArray[np.float32]:
    """A parallel equivalent of np.trapz.

    Parameters
    ----------
    waveforms : npt.NDArray[np.float32]
        The waveform accelerations to integrate.
    dt : float
        The timestep resolution of the waveform array.


    Returns
    -------
    npt.NDArray[np.float32]
        The integrated waveform array using the trapezium rule.
    """
    sums = np.zeros((waveforms.shape[0],), np.float32)
    for i in numba.prange(waveforms.shape[0]):
        for j in range(waveforms.shape[1]):
            if j == 0 or j == waveforms.shape[1] - 1:
                sums[i] += waveforms[i, j] / 2
            else:
                sums[i] += waveforms[i, j]
    return sums * dt


def compute_significant_duration(
    waveforms: npt.NDArray[np.float32],
    dt: float,
    percent_low: float,
    percent_high: float,
) -> npt.NDArray[np.float32]:
    """Compute the significant duration of the rupture (the time to accumulate `percent_low` to `percent_high` of Arias Intensity).

    Parameters
    ----------
    waveforms : npt.NDArray[np.float32]
        The waveform accelerations to evaluate.
    dt : float
        The timestep resolution of the waveform array.
    percent_low : float
        The lower bound on the significant duration.
    percent_high : float
        The upper bound on the significant duration.

    Returns
    -------
    npt.NDArray[np.float32]
        The significant duration for each station (in seconds).
    """
    arias_intensity = cumulative_arias_integrate(waveforms, dt)
    arias_intensity /= arias_intensity[:, -1][:, np.newaxis]
    sum_mask = ne.evaluate(
        "(arias_intensity >= percent_low / 100) & (arias_intensity <= percent_high / 100)"
    )
    threshold_values = np.count_nonzero(sum_mask, axis=1) * dt
    return threshold_values.ravel()


def compute_fas(
    stations: pd.Series,
    waveforms: npt.NDArray[np.float32],
    dt: float,
    freqs: npt.NDArray[np.float32],
) -> pd.DataFrame:
    """Compute fourier amplitude spectrum (FAS) of a seismic waveform.

    Parameters
    ----------
    stations : pd.Series
        Station information for each waveform entry.
    waveforms : npt.NDArray[np.float32]
        The waveform array.
    dt : float
        The timestep for the waveforms.
    freqs : npt.NDArray[np.float32]
        The frequencies to compute FAS for.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the FAS calculation for each station at each period.
    """
    n_fft = 2 ** int(np.ceil(np.log2(waveforms.shape[1])))
    fa_frequencies = np.fft.rfftfreq(n_fft, dt)
    fa_spectrum = np.abs(np.fft.rfft(waveforms, n=n_fft, axis=1))
    smoother = pykooh.CachedSmoother(fa_frequencies, freqs, 40)
    with multiprocessing.Pool(utils.get_available_cores()) as pool:
        fas_0 = np.array(
            pool.map(smoother, fa_spectrum[:, :, 1]),
            dtype=np.float32,
        )
        fas_90 = np.array(
            pool.map(smoother, fa_spectrum[:, :, 0]),
            dtype=np.float32,
        )
        fas_ver = np.array(
            pool.map(smoother, fa_spectrum[:, :, 2]),
            dtype=np.float32,
        )
    fas_mean = np.sqrt(np.square(fas_0) + np.square(fas_90))
    fas_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "station": stations,
                    "intensity_measure": f"FAS_{freq:.2f}",
                    "000": fas_0[:, i],
                    "090": fas_90[:, i],
                    "ver": fas_ver[:, i],
                    "geom": fas_mean[:, i],
                }
            )
            for i, freq in enumerate(freqs)
        ]
    )
    fas_df["rotd50"] = np.nan
    fas_df["rotd100"] = np.nan
    return fas_df


# CAV calculation improved by
# Jérôme Richard: https://stackoverflow.com/questions/79164983/numerically-integrating-signals-with-absolute-value/79173972#79173972
@numba.njit(parallel=True)
def cav_integrate(
    waveform: npt.NDArray[np.float32], dt: float
) -> npt.NDArray[np.float32]:
    """Compute the Cumulative Absolute Velocity (CAV) of a waveform."""
    cav = np.zeros((waveform.shape[0],), dtype=np.float32)
    dtf = np.float32(dt)
    half = np.float32(0.5)
    for i in numba.prange(np.int32(waveform.shape[0])):
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
    return cav


@numba.njit(parallel=True)
def arias_integrate(
    waveform: npt.NDArray[np.float32], dt: float
) -> npt.NDArray[np.float32]:
    """Compute the Arias Intensity (AI) of a waveform."""
    ai = np.zeros((waveform.shape[0],), dtype=np.float32)
    dtf = np.float32(dt)
    half = np.float32(0.5)
    for i in numba.prange(np.int32(waveform.shape[0])):
        tmp = np.float32(0)
        for j in range(np.int32(waveform.shape[1] - 1)):
            v1 = waveform[i, j]
            v2 = waveform[i, j + 1]
            if min(v1, v2) >= 0 or max(v1, v2) <= 0:
                tmp += dtf * (np.abs(v1) + np.abs(v2))
            else:
                inv_slope = dtf / (v2 - v1)
                x0 = -v1 * inv_slope
                tmp += x0 * np.square(v1) + (dtf - x0) * np.square(v2)
        ai[i] = tmp * half
    return ai


@numba.njit(parallel=True)
def cumulative_arias_integrate(
    waveform: npt.NDArray[np.float32], dt: float
) -> npt.NDArray[np.float32]:
    """Compute the cumulative AI of a waveform."""
    ai = np.zeros_like(waveform, dtype=np.float32)
    dtf = np.float32(dt)
    half = np.float32(0.5)
    for i in numba.prange(np.int32(waveform.shape[0])):
        tmp = np.float32(0)
        for j in range(np.int32(waveform.shape[1] - 1)):
            v1 = waveform[i, j]
            v2 = waveform[i, j + 1]
            if min(v1, v2) >= 0 or max(v1, v2) <= 0:
                tmp += dtf * (np.abs(v1) + np.abs(v2))
            else:
                inv_slope = dtf / (v2 - v1)
                x0 = -v1 * inv_slope
                tmp += x0 * np.square(v1) + (dtf - x0) * np.square(v2)

            ai[i, j + 1] = tmp
    return ai * half


@app.command(help="Calculate instensity measures for simulation data.")
def calculate_instensity_measures(
    realisation_ffp: Annotated[
        Path,
        typer.Argument(
            help="Realisation filepath", exists=True, dir_okay=False, writable=True
        ),
    ],
    broadband_simulation_ffp: Annotated[
        Path,
        typer.Argument(help="Broadband simulation file.", exists=True, dir_okay=False),
    ],
    output_path: Annotated[
        Path,
        typer.Argument(
            help="Output path for IM calculation summary statistics.",
            dir_okay=False,
            writable=True,
        ),
    ],
    simulated_stations: Annotated[
        bool, typer.Option(help="If passed, calculate for simulated stations.")
    ] = True,
    psa_rotd_maximum_memory_allocation: Annotated[
        Optional[float],
        typer.Option(
            help="Maximum amount of memory allocated for rotated PSA calculation station buffer, in gigabytes.",
            min=0,
        ),
    ] = None,
):
    """Calculate intensity measures for simulation data.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation file.
    broadband_simulation_ffp : Path
        Path to the broadband simulation waveforms.
    output_path : Path
        Output directory for IM calc summary statistics.
    """
    ne.set_num_threads(utils.get_available_cores())

    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    broadband_parameters = BroadbandParameters.read_from_realisation(realisation_ffp)
    intensity_measure_parameters = (
        IntensityMeasureCalculationParameters.read_from_realisation_or_defaults(
            realisation_ffp, metadata.defaults_version
        )
    )

    with h5py.File(broadband_simulation_ffp, mode="r") as broadband_file:
        waveforms = np.array(broadband_file["waveforms"]).astype(np.float32)

    stations = pd.read_hdf(broadband_simulation_ffp, key="stations")
    if not simulated_stations:
        stations = stations.filter(regex=r"^\w{4}$", axis=0)
        waveforms = waveforms[stations["waveform_index"]]

    intensity_measures = intensity_measure_parameters.ims
    intensity_measure_statistics = pd.DataFrame()
    pbar = tqdm.tqdm(intensity_measures)
    g = 981
    for intensity_measure in pbar:
        pbar.set_description(intensity_measure)
        match intensity_measure.lower():
            case "pga":
                individual_intensity_measure_statistics = compute_in_rotations(
                    waveforms, lambda v: v.max(axis=1)
                )
                individual_intensity_measure_statistics["station"] = stations.index
                individual_intensity_measure_statistics["intensity_measure"] = "PGA"
            case "pgv":
                individual_intensity_measure_statistics = compute_in_rotations(
                    np.abs(
                        sp.integrate.cumulative_trapezoid(
                            waveforms, dx=broadband_parameters.dt, axis=1
                        )
                    )
                    * g,
                    lambda v: v.max(axis=1),
                    component_wise_operation=ComponentWiseOperation.NONE,
                )
                individual_intensity_measure_statistics["station"] = stations.index
                individual_intensity_measure_statistics["intensity_measure"] = "PGV"
            case "cav":
                individual_intensity_measure_statistics = compute_in_rotations(
                    waveforms,
                    lambda v: cav_integrate(v, broadband_parameters.dt),
                    component_wise_operation=ComponentWiseOperation.NONE,
                )
                individual_intensity_measure_statistics["station"] = stations.index
                individual_intensity_measure_statistics["intensity_measure"] = "CAV"
            case "ai":
                individual_intensity_measure_statistics = compute_in_rotations(
                    waveforms,
                    lambda v: (np.pi * g)
                    / 2
                    * arias_integrate(v, broadband_parameters.dt),
                    component_wise_operation=ComponentWiseOperation.NONE,
                )
                individual_intensity_measure_statistics["station"] = stations.index
                individual_intensity_measure_statistics["intensity_measure"] = "AI"
            case "ds575":
                individual_intensity_measure_statistics = compute_in_rotations(
                    waveforms,
                    lambda v: compute_significant_duration(
                        v, broadband_parameters.dt, 5, 75
                    ),
                    component_wise_operation=ComponentWiseOperation.NONE,
                )
                individual_intensity_measure_statistics["station"] = stations.index
                individual_intensity_measure_statistics["intensity_measure"] = "Ds575"
            case "ds595":
                individual_intensity_measure_statistics = compute_in_rotations(
                    waveforms,
                    lambda v: compute_significant_duration(
                        v, broadband_parameters.dt, 5, 95
                    ),
                    component_wise_operation=ComponentWiseOperation.NONE,
                )
                individual_intensity_measure_statistics["station"] = stations.index
                individual_intensity_measure_statistics["intensity_measure"] = "Ds595"
            case "psa":
                individual_intensity_measure_statistics = compute_psa(
                    stations.index,
                    waveforms,
                    np.array(
                        intensity_measure_parameters.valid_periods, dtype=np.float32
                    ),
                    broadband_parameters.dt,
                    psa_rotd_maximum_memory_allocation=psa_rotd_maximum_memory_allocation
                    * 1e9
                    if psa_rotd_maximum_memory_allocation
                    else None,
                )
            case "fas":
                individual_intensity_measure_statistics = compute_fas(
                    stations.index,
                    waveforms,
                    broadband_parameters.dt,
                    np.array(
                        intensity_measure_parameters.fas_frequencies, dtype=np.float32
                    ),
                )

        intensity_measure_statistics = pd.concat(
            [intensity_measure_statistics, individual_intensity_measure_statistics]
        )

    intensity_measure_statistics.set_index(["station", "intensity_measure"]).to_parquet(
        output_path
    )
