"""IM calculation script for ascii waveforms"""

import numpy as np
import pandas as pd
import xarray as xr

from IM import ims, konno_ohmachi
from IM.ims import IM

DEFAULT_PERIODS = np.asarray(
    [
        0.010,
        0.020,
        0.022,
        0.025,
        0.029,
        0.030,
        0.032,
        0.035,
        0.036,
        0.040,
        0.042,
        0.044,
        0.045,
        0.046,
        0.048,
        0.050,
        0.055,
        0.060,
        0.065,
        0.067,
        0.070,
        0.075,
        0.080,
        0.085,
        0.090,
        0.095,
        0.100,
        0.110,
        0.120,
        0.130,
        0.133,
        0.140,
        0.150,
        0.160,
        0.170,
        0.180,
        0.190,
        0.200,
        0.220,
        0.240,
        0.250,
        0.260,
        0.280,
        0.290,
        0.300,
        0.320,
        0.340,
        0.350,
        0.360,
        0.380,
        0.400,
        0.420,
        0.440,
        0.450,
        0.460,
        0.480,
        0.500,
        0.550,
        0.600,
        0.650,
        0.667,
        0.700,
        0.750,
        0.800,
        0.850,
        0.900,
        0.950,
        1.000,
        1.100,
        1.200,
        1.300,
        1.400,
        1.500,
        1.600,
        1.700,
        1.800,
        1.900,
        2.000,
        2.200,
        2.400,
        2.500,
        2.600,
        2.800,
        3.000,
        3.200,
        3.400,
        3.500,
        3.600,
        3.800,
        4.000,
        4.200,
        4.400,
        4.600,
        4.800,
        5.000,
        5.500,
        6.000,
        6.500,
        7.000,
        7.500,
        8.000,
        8.500,
        9.000,
        9.500,
        10.000,
        11.000,
        12.000,
        13.000,
        14.000,
        15.000,
        20.000,
    ]
)
DEFAULT_FREQUENCIES = np.logspace(
    np.log10(0.01318257),
    np.log10(100),
    num=389,
)

FREQUENCY_LABEL_SIGNIFICANT_FIGURES = 6
"""Significant figures used when labelling FAS columns (e.g. FAS_0.0131826)."""


def frequency_label(frequency: float) -> str:
    """Format a frequency as it appears in FAS column names.

    Parameters
    ----------
    frequency : float
        Frequency in Hz.

    Returns
    -------
    str
        The frequency rounded to `FREQUENCY_LABEL_SIGNIFICANT_FIGURES`
        significant figures.
    """
    return f"{frequency:.{FREQUENCY_LABEL_SIGNIFICANT_FIGURES}g}"


def _dataset_to_frame(dataset: xr.Dataset, index: list[str]) -> pd.DataFrame:
    """Convert a component-per-variable IM dataset into a wide DataFrame.

    Each component (`000`, `090`, ..., `rotd100`) is already a data variable,
    so the dataset's own columns are the frame's columns; this just drops
    any non-dimension coordinates (e.g. `latitude`/`longitude`, when a real
    DataArray is passed in) and replaces the row index with `index`.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with one data variable per component.
    index : list of str
        Row labels to assign to the resulting DataFrame.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with component names as columns.
    """
    frame = dataset.reset_coords(drop=True).to_dataframe()
    frame.index = index
    return frame


def calculate_ims(
    waveform: np.ndarray,
    dt: float,
    ims_list: list[IM] | None = None,
    periods: np.ndarray = DEFAULT_PERIODS,
    frequencies: np.ndarray = DEFAULT_FREQUENCIES,
    bandwidth: float = konno_ohmachi.DEFAULT_BANDWIDTH,
):
    """
    Calculate intensity measures for a single waveform.

    Parameters
    ----------
    waveform : np.ndarray
        Waveform data as a NumPy array.
    dt : float
        Sampling interval (dt) of the waveform.
    ims_list : list of IM, optional
        List of intensity measures (IMs) to calculate, e.g., [IM.PGA, IM.pSA, IM.CAV].
    periods : np.ndarray, optional
        List of periods required for calculating the pseudo-spectral acceleration (pSA).
    frequencies : np.ndarray, optional
        List of frequencies required for calculating the Fourier amplitude spectrum (FAS).
    bandwidth : float, optional
        Bandwidth of the Konno-Ohmachi window used to smooth the Fourier
        amplitude spectrum. Lower values smooth more strongly.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the calculated intensity measures.
        The columns are the IMs and the rows are the different components.

    Raises
    ------
    ValueError
        If the IM is not recognized or if required environment variables are not set to 1.
    """
    if ims_list is None:
        ims_list = list(IM)
    results = []
    waveform = np.atleast_3d(waveform)
    waveform = np.ascontiguousarray(np.moveaxis(waveform, -1, 0))
    # Iterate through IMs and calculate them
    for im in ims_list:
        if im == IM.PGA:
            dataset = ims.peak_ground_acceleration(waveform)
            result = _dataset_to_frame(dataset, [im.value])
        elif im == IM.PGV:
            dataset = ims.peak_ground_velocity(waveform, dt)
            result = _dataset_to_frame(dataset, [im.value])
        elif im == IM.PGD:
            dataset = ims.peak_ground_displacement(waveform, dt)
            result = _dataset_to_frame(dataset, [im.value])
        elif im == IM.pSA:
            dataset = ims.pseudo_spectral_acceleration(waveform, periods, dt)
            result = _dataset_to_frame(
                dataset,
                [f"{im.value}_{idx}" for idx in dataset.coords["period"].values],
            )
        elif im == IM.CAV:
            dataset = ims.cumulative_absolute_velocity(waveform, dt)
            result = _dataset_to_frame(dataset, [im.value])
        elif im == IM.CAV5:
            dataset = ims.cumulative_absolute_velocity(waveform, dt, threshold=5)
            result = _dataset_to_frame(dataset, [im.value])
        elif im == IM.Ds575:
            dataset = ims.ds575(waveform, dt)
            result = _dataset_to_frame(dataset, [im.value])
        elif im == IM.Ds595:
            dataset = ims.ds595(waveform, dt)
            result = _dataset_to_frame(dataset, [im.value])
        elif im == IM.AI:
            dataset = ims.arias_intensity(waveform, dt)
            result = _dataset_to_frame(dataset, [im.value])
        elif im == IM.FAS:
            dataset = ims.fourier_amplitude_spectra(
                waveform,
                dt,
                frequencies,
                bandwidth=bandwidth,
            )
            result = _dataset_to_frame(
                dataset,
                [
                    f"{im.value}_{frequency_label(idx)}"
                    for idx in dataset.coords["frequency"].values
                ],
            )
        else:
            raise ValueError(
                f"IM {im} not recognized. Available IMs are {IM.__members__.keys()}"
            )
        results.append(result)

    # Combine all results into a single DataFrame
    output_ims = pd.concat(results).T
    return output_ims
