from pathlib import Path

import numpy as np
import pandas as pd


def read_ascii(
    file_000: Path, file_090: Path, file_ver: Path
) -> tuple[float, np.ndarray]:
    """
    Read ASCII waveform files (000, 090, vertical) and return the sampling interval (dt) and waveform data.

    Parameters
    ----------
    file_000 : Path
        Path to the 000 component file.
    file_090 : Path
        Path to the 090 component file.
    file_ver : Path
        Path to the vertical component file.

    Returns
    -------
    Tuple[float, np.ndarray]
        Sampling interval (dt) and waveform data as a NumPy array.
    """
    # Load all components
    comp_000 = pd.read_csv(file_000, sep=r"\s+", header=None, skiprows=2).values.ravel()
    comp_090 = pd.read_csv(file_090, sep=r"\s+", header=None, skiprows=2).values.ravel()
    comp_ver = pd.read_csv(file_ver, sep=r"\s+", header=None, skiprows=2).values.ravel()

    # Extract the sampling interval (dt) from the 000 component file
    delta = pd.read_csv(file_000, sep=r"\s+", header=None, nrows=2, skiprows=1).iloc[
        0, 1
    ]

    if np.any(np.isnan(comp_000) | np.isnan(comp_090) | np.isnan(comp_ver)):
        raise ValueError('Components contain NaN values.')

    # Stack components into a single NumPy array
    waveform_data = np.stack((comp_000, comp_090, comp_ver), axis=1)

    # Ensure dtype is float 32
    waveform_data = waveform_data.astype(np.float32)

    # Reshape the waveform to have the correct shape for the IM calculation
    reshaped_waveform = waveform_data[np.newaxis, :, :]

    return delta, reshaped_waveform
