"""Script for generating the Konno matrices"""
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from obspy.signal.konnoohmachismoothing import calculate_smoothing_matrix

app = typer.Typer()


@app.command()
def main(
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory where the Konno matrices will be saved",
            exists=False,
            file_okay=False,
        ),
    ],
    num_to_gen: Annotated[
        int, typer.Option(help="Number of Ko matrix generations to compute, starts at 8, 16, 32, 64, etc. Default of 14 (going upto KO matrix of 65536)",)
    ] = 14,
    bandwidth: Annotated[
        int, typer.Option(help="Bandwidth of the Konno-Ohmachi smoothing window",)
    ] = 40,
    dt: Annotated[
        float, typer.Option(help="Sampling interval of the time series",)
    ] = 0.005,
):
    """Generate the Konno matrices for different window sizes."""
    for i in range(num_to_gen):
        n = 8 * 2 ** i
        ft_freq = np.fft.rfftfreq(n * 2, dt).astype(np.float32)
        cur_konno = calculate_smoothing_matrix(ft_freq, bandwidth=bandwidth, normalize=True)
        np.save(output_dir / f"KO_{n}.npy", cur_konno)


if __name__ == "__main__":
    app()