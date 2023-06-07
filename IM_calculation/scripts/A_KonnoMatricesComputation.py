from pathlib import Path
import numpy as np
import os

from obspy.signal.konnoohmachismoothing import calculate_smoothing_matrix


def createKonnoMatrix_single(ft_len):
    # creates several Konno Ohmachi matrices
    dt = 0.005
    ft_freq = np.fft.rfftfreq(ft_len, dt)
    konno = calculate_smoothing_matrix(ft_freq, bandwidth=20, normalize=True)
    return konno


def createKonnoMatrices(install_directory, num_to_gen: int = 7):
    root = os.path.abspath(
        os.path.join(install_directory, "IM_calculation", "IM", "KO_matrices")
    )
    os.makedirs(root, exist_ok=True)

    for i in range(num_to_gen):
        # n = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        n = 512 * 2 ** i
        file_name = os.path.join(root, f"KO_{n}.npy")
        matrix = createKonnoMatrix_single(n * 2)
        np.save(file_name, matrix)
        del matrix
        print(f"Generated Konno {n}")


def main():
    output_dir = Path(__file__).parent.parent / "IM" / "KO_matrices"
    output_dir.mkdir(exist_ok=True)

    for i in range(7):
        n = 512 * 2 ** i
        cur_ffp = output_dir / f"KO_{n}.npy"
        print(cur_ffp)
        if not cur_ffp.exists():
            print(f"Generating Konno {n}")
            np.save(str(cur_ffp), createKonnoMatrix_single(n * 2))
        else:
            print(f"Skipping Konno {n} as it already exists")


if __name__ == "__main__":
    main()
