from pathlib import Path
import numpy as np
import threading

# Thread-safe dictionary
matrices = {}
matrix_lock = threading.Lock()

def get_konno_matrix(size: int, directory: Path = None):
    """
    Retrieves the precomputed Konno matrix from a file.

    Parameters
    ----------
    size : int
        The size of the matrix to load.
    directory : Path, optional
        Directory containing precomputed KO matrices. If None, defaults to the script's "KO_matrices" folder.

    Returns
    -------
    np.ndarray
        The loaded Konno matrix.

    Raises
    ------
    FileNotFoundError
        If the required matrix file does not exist.
    """
    # Default directory handling
    directory = Path(directory or Path(__file__).parent / "KO_matrices").resolve()

    with matrix_lock:
        # Return cached matrix if already loaded
        if size in matrices:
            return matrices[size]

        # File path for the matrix
        ko_matrix_file = directory / f"KO_{size - 1}.npy"

        if not ko_matrix_file.exists():
            num_to_gen = max(0, int(np.log2(size - 1)) - 9)
            raise FileNotFoundError(
                f"KO matrix file '{ko_matrix_file}' not found.\n"
                f"Run 'A_KonnoMatricesComputation.py' with at least {num_to_gen} generations."
            )

        # Load matrix with memory mapping for efficiency
        matrices[size] = np.load(ko_matrix_file, mmap_mode="r")

    return matrices[size]
