import typer
from pathlib import Path
import numpy as np
from enum import StrEnum
from typing import List, Optional

from IM import ims, waveform_reading


app = typer.Typer()


@app.command()
def calculate_ims_mseed(
    mseed_file: Path,
    ims_list: List[IM],
    components: List[str],
    periods: Optional[List[float]] = typer.Option(None, help="Periods required for pSA"),
    frequencies: Optional[List[float]] = typer.Option(None, help="Frequencies required for FAS"),
):
    """
    Calculate intensity measures for a waveform stored in a MiniSEED file.

    Parameters
    ----------
    mseed_file : Path
        Path to the MiniSEED file.
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
    # Read the mseed file
    dt, waveform = waveform_reading.read_mseed(mseed_file)

    # Calculate the intensity measures
    calculate_ims(waveform, dt, ims_list, components, periods, frequencies)


@app.command()
def calculate_ims_ascii(
    file_000: Path,
    file_090: Path,
    file_ver: Path,
    ims_list: List[IM],
    components: List[str],
    periods: Optional[List[float]] = typer.Option(None, help="Periods required for pSA"),
    frequencies: Optional[List[float]] = typer.Option(None, help="Frequencies required for FAS"),
):
    """
    Calculate intensity measures for ASCII waveform files (000, 090, vertical).

    Parameters
    ----------
    file_000 : Path
        Path to the 000 component file.
    file_090 : Path
        Path to the 090 component file.
    file_ver : Path
        Path to the vertical component file.
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
    # Read the ASCII files
    dt, waveform_000 = waveform_reading.read_ascii(file_000, file_090, file_ver)

    # Calculate the intensity measures
    calculate_ims(waveform_000, dt, ims_list, components, periods, frequencies)



if __name__ == "__main__":
    app()