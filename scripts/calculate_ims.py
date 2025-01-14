import multiprocessing
from pathlib import Path

import typer

from IM import im_calculation, waveform_reading

app = typer.Typer()


@app.command()
def calculate_ims_ascii(
    file_000: Path,
    file_090: Path,
    file_ver: Path,
    ims_list: list[im_calculation.IM],
    periods: list[float] = typer.Option(im_calculation.DEFAULT_PERIODS, help="Periods required for pSA"),
    frequencies: list[float] = typer.Option(im_calculation.DEFAULT_FREQUENCIES, help="Frequencies required for FAS"),
    cores: int = typer.Option(multiprocessing.cpu_count(), help="Number of cores to use for parallel processing in pSA and FAS calculations."),
    ko_bandwidth: int = typer.Option(40, help="Bandwidth for the Konno-Ohmachi smoothing."),
    output_file: Path = typer.Option(None, help="Output file for the calculated IMs."),
):
    """
    Calculate intensity measures for ASCII waveform files (000, 090, vertical) for a single waveform.

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
    result = im_calculation.calculate_ims(waveform, dt, ims_list, periods, frequencies, cores, ko_bandwidth)

    # Save the results to a file if provided
    if output_file:
        result.to_csv(output_file)

    return result



if __name__ == "__main__":
    app()