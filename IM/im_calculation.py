import multiprocessing
import os
from pathlib import Path

import numpy as np
import pandas as pd

from IM import ims
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


def calculate_ims(
    waveform: np.ndarray,
    dt: float,
    ims_list: list[IM] = list(IM),
    periods: np.ndarray = DEFAULT_PERIODS,
    frequencies: np.ndarray = DEFAULT_FREQUENCIES,
    cores: int = multiprocessing.cpu_count(),
    ko_directory: Path | None = None,
    use_numexpr: bool = False,
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
    cores : int, optional
        Number of cores to use for parallel processing in pSA and FAS calculations.
    ko_directory : Path, optional
        Path to the directory containing the Konno-Ohmachi matrices.
        Only required if FAS is in the list of IMs.
    use_numexpr : bool, optional
        If True, use numexpr for calculations. (Faster off for single waveform and multiprocessing)
        Default is False.

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
    if cores == 1:
        required_env_vars = [
            "NUMEXPR_NUM_THREADS",
            "NUMBA_MAX_THREADS",
            "NUMBA_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        ]
        unset_vars = [var for var in required_env_vars if os.getenv(var) != "1"]
        if unset_vars:
            raise ValueError(
                f"The following environment variables must be set to 1: {', '.join(unset_vars)}"
            )
    if ko_directory is None and IM.FAS in ims_list:
        raise ValueError(
            "The Konno-Ohmachi directory must be provided if Fourier amplitude spectrum is in the list of IMs."
        )

    results = []

    # Iterate through IMs and calculate them
    for im in ims_list:
        if im == IM.PGA:
            result = ims.peak_ground_acceleration(waveform, use_numexpr=use_numexpr)
            result.index = [im.value]
        elif im == IM.PGV:
            result = ims.peak_ground_velocity(waveform, dt, use_numexpr=use_numexpr)
            result.index = [im.value]
        elif im == IM.pSA:
            data_array = ims.pseudo_spectral_acceleration(
                waveform, periods, np.float32(dt), cores=cores, use_numexpr=use_numexpr
            )
            # Convert the data array to a DataFrame
            result = data_array.to_dataframe().unstack(level="component")
            result.index = [
                f"{im.value}_{idx}" for idx in data_array.coords["period"].values
            ]
            result.columns = result.columns.droplevel(0)
        elif im == IM.CAV:
            result = ims.cumulative_absolute_velocity(waveform, dt)
            result.index = [im.value]
        elif im == IM.CAV5:
            result = ims.cumulative_absolute_velocity(waveform, dt, 5)
            result.index = [im.value]
        elif im == IM.Ds575:
            result = ims.ds575(waveform, dt, use_numexpr=use_numexpr)
            result.index = [im.value]
        elif im == IM.Ds595:
            result = ims.ds595(waveform, dt, use_numexpr=use_numexpr)
            result.index = [im.value]
        elif im == IM.AI:
            result = ims.arias_intensity(waveform, dt)
            result.index = [im.value]
        elif im == IM.FAS:
            assert ko_directory
            data_array = ims.fourier_amplitude_spectra(
                waveform,
                dt,
                frequencies,
                cores=cores,
                # ko_directory must be Path because of the check earlier.
                ko_directory=ko_directory,
            )
            # Convert the data array to a DataFrame
            result = data_array.to_dataframe().unstack(level="component")
            result.index = [
                f"{im.value}_{idx}" for idx in data_array.coords["frequency"].values
            ]
            result.columns = result.columns.droplevel(0)
        else:
            raise ValueError(
                f"IM {im} not recognized. Available IMs are {IM.__members__.keys()}"
            )
        results.append(result)

    # Combine all results into a single DataFrame
    output_ims = pd.concat(results).T
    return output_ims
