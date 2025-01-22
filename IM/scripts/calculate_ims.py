import multiprocessing
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from IM import im_calculation, waveform_reading

app = typer.Typer()


@app.command(
    help="Calculate intensity measures for a single ASCII waveform fileset (000, 090, vertical)."
)
def calculate_ims_ascii(
    file_000: Annotated[
        Path,
        typer.Argument(
            help="ASCII file containing 000 component of waveform.",
            exists=True,
            dir_okay=False,
        ),
    ],
    file_090: Annotated[
        Path,
        typer.Argument(
            help="ASCII file containing 090 component of waveform.",
            exists=True,
            dir_okay=False,
        ),
    ],
    file_ver: Annotated[
        Path,
        typer.Argument(
            help="ASCII file containing ver component of waveform.",
            exists=True,
            dir_okay=False,
        ),
    ],
    output_file: Annotated[
        Path, typer.Argument(help="Output file for the calculated IMs.", dir_okay=False)
    ],
    ims_list: Annotated[
        list[im_calculation.IM],
        typer.Argument(
            help="Intensity measures to calculate.",
        ),
    ],
    periods: Annotated[
        list[float], typer.Option(help="Periods required for pSA")
    ] = list(im_calculation.DEFAULT_PERIODS),
    frequencies: Annotated[
        list[float], typer.Option(help="Frequencies required for FAS")
    ] = list(im_calculation.DEFAULT_FREQUENCIES),
    cores: Annotated[
        int,
        typer.Option(
            help="Number of cores to use for parallel processing in pSA and FAS calculations."
        ),
    ] = multiprocessing.cpu_count(),
    ko_bandwidth: Annotated[
        int, typer.Option(help="Bandwidth for Konno-Ohmachi smoothing.")
    ] = 40,
):
    """
    Calculate intensity measures for a single ASCII waveform fileset (000, 090, vertical).

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
    periods : list of float, optional
        List of periods required for calculating the pseudo-spectral acceleration (pSA).
    frequencies : list of float, optional
        List of frequencies required for calculating the Fourier amplitude spectrum (FAS).
    cores : int, optional
        Number of cores to use for parallel processing in pSA and FAS calculations.
    ko_bandwidth : int, optional
        Bandwidth for the Konno-Ohmachi smoothing (Applied to FAS).
    output_file : Path, optional
        Output file for the calculated IMs.
        When provided, the calculated IMs will be saved to this file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the calculated intensity measures.
        The columns are the IMs and the rows are the different components.
    """
    # Read the ASCII files
    dt, waveform = waveform_reading.read_ascii(file_000, file_090, file_ver)

    # Calculate the intensity measures
    result = im_calculation.calculate_ims(
        waveform,
        dt,
        ims_list,
        np.array(periods),
        np.array(frequencies),
        cores,
        ko_bandwidth,
    )

    result.to_csv(output_file)
