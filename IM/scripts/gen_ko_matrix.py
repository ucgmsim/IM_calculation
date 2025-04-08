"""Script for generating the Konno matrices"""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from obspy.signal.konnoohmachismoothing import calculate_smoothing_matrix

from qcore import cli

app = typer.Typer()


@cli.from_docstring(app)
def main(
    output_dir: Annotated[
        Path,
        typer.Argument(
            exists=False,
            file_okay=False,
        ),
    ],
    num_to_gen: Annotated[int, typer.Option()] = 14,
    bandwidth: Annotated[int, typer.Option()] = 40,
):
    """Generate the Konno matrices for different window sizes.

    Parameters
    ----------
    output_dir : Path
        Directory where the Konno matrices will be saved.
    num_to_gen : int
        Number of KO matrix generations to compute. The largest matrix will have `4 * 2 ** num_to_gen` rows and columns.
    bandwidth : int
        Bandwidth of the Konno-Ohmachi smoothing window.

    """
    for i in range(num_to_gen):
        n = 8 * 2**i
        ft_freq = np.fft.rfftfreq(n * 2).astype(np.float32)
        cur_konno = calculate_smoothing_matrix(
            ft_freq, bandwidth=bandwidth, normalize=True
        )
        np.save(output_dir / f"KO_{n}.npy", cur_konno)


if __name__ == "__main__":
    app()
