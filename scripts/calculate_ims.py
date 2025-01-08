import typer
from pathlib import Path
import numpy as np
from enum import StrEnum
from typing import List, Optional

from IM import ims, waveform_reading


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

app = typer.Typer()


def calculate_ims(
    waveform: np.ndarray,
    dt: float,
    ims_list: List[IM],
    components: List[str],
    periods: Optional[List[float]] = None,
    frequencies: Optional[List[float]] = typer.Option(None, help="Frequencies required for FAS"),
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
    # Validate components
    if not components:
        typer.echo("Components list cannot be empty.")
        raise typer.Exit()

    # Placeholder for results
    results = {}

    # Iterate through IMs and calculate them
    for im in ims_list:
        typer.echo(f"Calculating {im}...")
        try:
            if im == IM.PGA:
                results[im] = ims.peak_ground_acceleration(waveform)
            elif im == IM.PGV:
                results[im] = ims.calculate_pgv(waveform, components)
            elif im == IM.pSA:
                if not periods:
                    typer.echo("Periods are required for pSA.")
                    raise typer.Exit()
                results[im] = ims.calculate_psa(waveform, components, periods)
            elif im == IM.CAV:
                results[im] = ims.calculate_cav(waveform, components)
            elif im == IM.CAV5:
                results[im] = ims.calculate_cav5(waveform, components)
            elif im == IM.MMI:
                results[im] = ims.calculate_mmi(waveform, components)
            elif im == IM.Ds575:
                results[im] = ims.calculate_ds575(waveform, components)
            elif im == IM.Ds595:
                results[im] = ims.calculate_ds595(waveform, components)
            elif im == IM.AI:
                results[im] = ims.calculate_ai(waveform, components)
            elif im == IM.FAS:
                if not frequencies:
                    typer.echo("Frequencies are required for FAS.")
                    raise typer.Exit()
                results[im] = ims.calculate_fas(waveform, components, frequencies)
            else:
                typer.echo(f"Unsupported IM: {im}")
        except Exception as e:
            typer.echo(f"Error calculating {im}: {e}")
            raise typer.Exit()


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