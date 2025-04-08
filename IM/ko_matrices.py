from pathlib import Path

import numpy as np


def get_konno_matrix(size: int, directory: Path) -> np.memmap:
    """
    Retrieves the precomputed Konno matrix from a file.

    Parameters
    ----------
    size : int
        The size of the matrix to load.
    directory : Path
        Directory containing precomputed KO matrices.

    Returns
    -------
    np.memmap
        The loaded Konno matrix.

    Raises
    ------
    FileNotFoundError
        If the required matrix file does not exist.
    """

    # File path for the matrix
    ko_matrix_file = directory / f"KO_{size - 1}.npy"

    if not ko_matrix_file.exists():
        try:
            num_to_gen = max(0, int(np.log2(size - 1)) - 2)
        except OverflowError:
            raise FileNotFoundError(
                f"Matrix size {size} is too small and most likely an issue with the waveform."
            )
        raise FileNotFoundError(
            f"KO matrix file '{ko_matrix_file}' not found.\n"
            f"Run 'gen_ko_matrix.py' with at least {num_to_gen} generations."
        )

    # Load matrix with memory mapping for efficiency
    return np.load(ko_matrix_file, mmap_mode="r")
