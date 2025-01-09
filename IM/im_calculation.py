from enum import StrEnum

import numpy as np
from typing import List, Optional

from IM import ims

class IM(StrEnum):
    PGA = "PGA"
    PGV = "PGV"
    pSA = "pSA"
    CAV = "CAV"
    CAV5 = "CAV5"
    MMI = "MMI"
    Ds575 = "Ds575"
    Ds595 = "Ds595"
    AI = "AI"
    FAS = "FAS"

def calculate_ims(
    waveform: np.ndarray[np.float32],
    dt: float,
    ims_list: List[IM],
    components: List[str],
    periods: Optional[List[float]] = None,
    frequencies: Optional[List[float]] = None,
):
    """
    Calculate intensity measures for a waveform.

    Parameters
    ----------
    waveform : np.ndarray
        Waveform data as a NumPy array.
    ims_list : list of IM
        List of intensity measures (IMs) to calculate, e.g., ['PGA', 'pSA', 'CAV'].
    components : list of str
        List of waveform components to process, e.g., ['X', 'Y', 'Z'].
    periods : list of float, optional
        List of periods required for calculating the pseudo-spectral acceleration (pSA).
    frequencies : list of float, optional
        List of frequencies required for calculating the Fourier amplitude spectrum (FAS).

    Returns
    -------
    None
        Displays the calculated intensity measures in the terminal or command output.
    """

    # Placeholder for results
    results = {}

    # Iterate through IMs and calculate them
    for im in ims_list:
        if im == IM.PGA:
            result = ims.peak_ground_acceleration(waveform)
            result.index = [im.value]
        elif im == IM.PGV:
            result = ims.peak_ground_velocity(waveform, dt)
            result.index = [im.value]
        elif im == IM.pSA:
            result = ims.pseudo_spectral_acceleration(waveform, periods, dt, cores=1)
            result.index = [f"{im.value}_{idx}" for idx in result.index]
        elif im == IM.CAV:
            result = ims.cumulative_absolute_velocity(waveform, dt)
            result.index = [im.value]
        elif im == IM.CAV5:
            result = ims.cumulative_absolute_velocity(waveform, dt, 5)
            result.index = [im.value]
        elif im == IM.MMI:
            results[im] = ims.calculate_mmi(waveform, components)
        elif im == IM.Ds575:
            results[im] = ims.calculate_ds575(waveform, components)
        elif im == IM.Ds595:
            results[im] = ims.calculate_ds595(waveform, components)
        elif im == IM.AI:
            results[im] = ims.calculate_ai(waveform, components)
        elif im == IM.FAS:
            results[im] = ims.calculate_fas(waveform, components, frequencies)
    return results