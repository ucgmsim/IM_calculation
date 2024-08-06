import argparse
from pathlib import Path

import numpy as np
from obspy.signal.konnoohmachismoothing import calculate_smoothing_matrix


def createKonnoMatrix_single(ft_len: int, bandwidth: int = 20):
    """
    Creates a single Konno Ohmachi matrix

    Parameters
    ----------
    ft_len : int
        Length of Fourier transform
    bandwidth : int, optional
        Bandwidth of Konno Ohmachi smoothing, by default 20
    """
    dt = 0.005
    ft_freq = np.fft.rfftfreq(ft_len, dt)
    konno = calculate_smoothing_matrix(ft_freq, bandwidth=bandwidth, normalize=True)
    return konno


def createKonnoMatrices(
    install_directory: Path, num_to_gen: int = 8, bandwidth: int = 20
):
    """
    Creates several Konno Ohmachi matrices

    Parameters
    ----------
    install_directory : Path
        Directory to install matrices
    num_to_gen : int, optional
        Number of matrices to generate, by default 8
    bandwidth : int, optional
        Bandwidth of Konno Ohmachi smoothing, by default 20
    """
    install_directory.mkdir(exist_ok=True)
    for i in range(num_to_gen):
        # n = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        n = 256 * 2**i
        file_name = install_directory / f"KO_{n}.npy"
        matrix = createKonnoMatrix_single(n * 2, bandwidth)
        np.save(file_name, matrix)
        del matrix
        print(f"Generated Konno {n}")


def load_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "-o",
        "--output_dir",
        help="path to output directory Default (.../IM_calculation/IM/KO_matrices)",
        type=Path,
        default=Path(__file__).parent.parent / "IM" / "KO_matrices",
    )
    parser.add_argument(
        "-n",
        "--num_to_gen",
        help="Number of Konno matrices to generate",
        type=int,
        default=8,
    )
    parser.add_argument(
        "-b",
        "--bandwidth",
        help="Bandwidth of Konno matrices",
        type=int,
        default=20,
    )

    args = parser.parse_args()
    return args


def main():
    args = load_args()
    createKonnoMatrices(args.output_dir, args.num_to_gen, args.bandwidth)


if __name__ == "__main__":
    main()
