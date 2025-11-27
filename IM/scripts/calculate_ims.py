import multiprocessing
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from IM import im_calculation, waveform_reading
from qcore import cli

app = typer.Typer()


@cli.from_docstring(app)
def calculate_ims_ascii(
    file_000: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
        ),
    ],
    file_090: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
        ),
    ],
    file_ver: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
        ),
    ],
    output_file: Annotated[Path, typer.Argument(dir_okay=False)],
    ims_list: Annotated[
        list[im_calculation.IM],
        typer.Argument(),
    ],
    periods: Annotated[list[float], typer.Option()] = list(
        im_calculation.DEFAULT_PERIODS
    ),
    frequencies: Annotated[list[float], typer.Option()] = list(
        im_calculation.DEFAULT_FREQUENCIES
    ),
    cores: Annotated[
        int,
        typer.Option(),
    ] = multiprocessing.cpu_count(),
    ko_directory: Annotated[Path | None, typer.Option()] = None,
) -> None:
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
    output_file : Path
        Output file for the calculated IMs.
    ims_list : list of IM
        List of intensity measures (IMs) to calculate, e.g., ['PGA', 'pSA', 'CAV'].
    periods : list of float, optional
        List of periods required for calculating the pseudo-spectral acceleration (pSA).
    frequencies : list of float, optional
        List of frequencies required for calculating the Fourier amplitude spectrum (FAS).
    cores : int, optional
        Number of cores to use for parallel processing in pSA and FAS calculations.
    ko_directory : Path, optional
        Path to the directory containing the Konno-Ohmachi matrices.
        Only required if FAS is in the list of IMs.
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
        ko_directory,
    )

    result.to_csv(output_file)
