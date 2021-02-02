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
    curr_dir = os.path.dirname(__file__)
    os.makedirs(os.path.join(curr_dir, "KO_matrices"), exist_ok=True)

    for i in range(7):
        # n = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        n = 512 * 2 ** i
        file_name = os.path.join(curr_dir, "KO_matrices", f"KO_{n}.npy")
        print(file_name)
        if not os.path.exists(file_name):
            print(f"Generating Konno {n}")
            np.save(file_name, createKonnoMatrix_single(n * 2))
        else:
            print(f"Skipping Konno {n}")


if __name__ == "__main__":
    main()
