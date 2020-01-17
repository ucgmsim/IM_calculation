import numpy as np
import os

from obspy.signal.konnoohmachismoothing import calculate_smoothing_matrix

def konno_ohmachi_smoothing_window(
    frequencies, center_frequency, bandwidth=40.0, normalize=False
):
    """
    Returns the Konno & Ohmachi Smoothing window for every frequency in
    frequencies.

    Returns the smoothing window around the center frequency with one value per
    input frequency defined as follows (see [Konno1998]_)::

        [sin(b * log_10(f/f_c)) / (b * log_10(f/f_c)]^4
            b   = bandwidth
            f   = frequency
            f_c = center frequency

    The bandwidth of the smoothing function is constant on a logarithmic scale.
    A small value will lead to a strong smoothing, while a large value of will
    lead to a low smoothing of the Fourier spectra.
    The default (and generally used) value for the bandwidth is 40. (From the
    `Geopsy documentation <http://www.geopsy.org>`_)

    All parameters need to be positive. This is not checked due to performance
    reasons and therefore any negative parameters might have unexpected
    results.

    :type frequencies: :class:`numpy.ndarray` (float32 or float64)
    :param frequencies:
        All frequencies for which the smoothing window will be returned.
    :type center_frequency: float
    :param center_frequency:
        The frequency around which the smoothing is performed. Must be greater
        or equal to 0.
    :type bandwidth: float
    :param bandwidth:
        Determines the width of the smoothing peak. Lower values result in a
        broader peak. Must be greater than 0. Defaults to 40.
    :type normalize: bool, optional
    :param normalize:
        The Konno-Ohmachi smoothing window is normalized on a logarithmic
        scale. Set this parameter to True to normalize it on a normal scale.
        Default to False.
    """
    if frequencies.dtype != np.float32 and frequencies.dtype != np.float64:
        msg = "frequencies needs to have a dtype of float32/64."
        raise ValueError(msg)
    # If the center_frequency is 0 return an array with zero everywhere except
    # at zero.
    if center_frequency == 0:
        smoothing_window = np.zeros(len(frequencies), dtype=frequencies.dtype)
        smoothing_window[frequencies == 0.0] = 1.0
        return smoothing_window
    # Disable div by zero errors and return zero instead
    with np.errstate(divide="ignore", invalid="ignore"):
        # Calculate the bandwidth*log10(f/f_c)
        smoothing_window = bandwidth * np.log10(frequencies / center_frequency)
        # Just the Konno-Ohmachi formulae.
        smoothing_window[:] = (np.sin(smoothing_window) / smoothing_window) ** 4
    # Check if the center frequency is exactly part of the provided
    # frequencies. This will result in a division by 0. The limit of f->f_c is
    # one.
    smoothing_window[frequencies == center_frequency] = 1.0
    # Also a frequency of zero will result in a logarithm of -inf. The limit of
    # f->0 with f_c!=0 is zero.
    smoothing_window[frequencies == 0.0] = 0.0
    # Normalize to one if wished.
    if normalize:
        smoothing_window /= smoothing_window.sum()
    return smoothing_window


def calculate_smoothing_matrix_ours(
    frequencies, bandwidth=40.0, normalize=False, fn: str = None
):
    """
    Calculates a len(frequencies) x len(frequencies) matrix with the Konno &
    Ohmachi window for each frequency as the center frequency.

    Any spectrum with the same frequency bins as this matrix can later be
    smoothed by using
    :func:`~obspy.signal.konnoohmachismoothing.apply_smoothing_matrix`.

    This also works for many spectra stored in one large matrix and is even
    more efficient.

    This makes it very efficient for smoothing the same spectra again and again
    but it comes with a high memory consumption for larger frequency arrays!

    :type frequencies: :class:`numpy.ndarray` (float32 or float64)
    :param frequencies:
        The input frequencies.
    :type bandwidth: float
    :param bandwidth:
        Determines the width of the smoothing peak. Lower values result in a
        broader peak. Must be greater than 0. Defaults to 40.
    :type normalize: bool, optional
    :param normalize:
        The Konno-Ohmachi smoothing window is normalized on a logarithmic
        scale. Set this parameter to True to normalize it on a normal scale.
        Default to False.
    :param fn: The name of the file to save the matrix to. Cached in ram otherwise. Useful for systems with low ram
    """
    # Create matrix to be filled with smoothing entries.

    sm_matrix = np.empty((len(frequencies), len(frequencies)), frequencies.dtype)
    if fn:
        np.save(fn, sm_matrix)
        sm_matrix = np.load(fn, mmap_mode="r+")
        # sm_matrix = np.memmap(fn, dtype=frequencies.dtype, mode='w+', shape=(len(frequencies), len(frequencies)))
        print("Generated")

    for _i, freq in enumerate(frequencies):
        sm_matrix[_i, :] = konno_ohmachi_smoothing_window(
            frequencies, freq, bandwidth, normalize=normalize
        )
        # print(sm_matrix[_i])
    return sm_matrix


def getSmoothMatrix(ft_freq, fn: str = None):
    # creates Konno Ohmachi matrix
    smooth_matrix = calculate_smoothing_matrix(ft_freq, bandwidth=20, normalize=True)
    # smooth_matrix = calculate_smoothing_matrix(
    #     ft_freq, bandwidth=20, normalize=True, fn=fn
    # )
    return smooth_matrix


def createKonnoMatrix_single(ft_len, fn: str = None):
    # creates several Konno Ohmachi matrices
    dt = 0.005
    ft_freq = np.arange(0, ft_len / 2 + 1) * (1.0 / (ft_len * dt))
    konno = getSmoothMatrix(ft_freq, fn)
    return konno


def createKonnoMatrices(
    install_directory, generate_on_disk: bool = False, num_to_gen: int = 5
):
    root = os.path.abspath(
        os.path.join(install_directory, "IM_calculation", "IM", "KO_matrices")
    )
    os.makedirs(root, exist_ok=True)

    for i in range(num_to_gen):
        # n = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        n = 512 * 2 ** i
        file_name = os.path.join(root, f"KO_{n}.npy")
        if generate_on_disk:
            matrix: np.memmap = createKonnoMatrix_single(n * 2, file_name)
            matrix.flush()
        else:
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
