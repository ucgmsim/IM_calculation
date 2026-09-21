"""Konno-Ohmachi spectral smoothing.

Smoothing a spectrum of `n` real-FFT bins is a product with an `n x n` matrix.
That matrix grows quadratically -- a 22 minute record at 100 Hz needs 17 GB --
so this module decides, per size, where to put it:

- small enough to hold, which is every observed record: build it in memory and
  keep it for the life of the process;
- too large for memory: build it into a scratch file that is unlinked while
  still open, so it costs address space rather than resident memory and cannot
  outlive the process;
- no room even for that: fall back to the matrix-free kernel, which is correct
  but re-derives every window on every call.

Nothing is cached between runs. Building the matrix costs about a sixth of what
applying it costs even at the largest sizes, so persisting it would buy under a
percent in exchange for stale-cache invalidation, bandwidth metadata and a
cross-process locking protocol.
"""

import atexit
import os
import shutil
import tempfile
import threading
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt

from IM import (
    _core,  # ty: ignore[unresolved-import]
)

DEFAULT_BANDWIDTH = 40.0
"""Bandwidth of the Konno-Ohmachi window. Lower values smooth more strongly."""

DEFAULT_MEMORY_BUDGET = 2 * 2**30
"""Largest matrix held in memory before spilling to scratch (bytes)."""

KONNO_BLOCK_BYTES = 64 * 2**20
"""Working-set target when building or applying a spilled matrix (bytes)."""

MEMORY_BUDGET_VARIABLE = "IM_CALCULATION_KO_MEMORY_BUDGET"
SCRATCH_DIRECTORY_VARIABLE = "IM_CALCULATION_SCRATCH_DIR"

_MATRICES: dict[tuple[int, float], np.ndarray] = {}
_RESIDENT_BYTES = 0
_LOCK = threading.RLock()


def memory_budget() -> int:
    """Largest matrix, in bytes, that is held in memory rather than spilled.

    Returns
    -------
    int
        The value of `$IM_CALCULATION_KO_MEMORY_BUDGET` if set, otherwise
        `DEFAULT_MEMORY_BUDGET`.
    """
    return int(os.environ.get(MEMORY_BUDGET_VARIABLE, DEFAULT_MEMORY_BUDGET))


def resolve_scratch_directory(directory: Path | None = None) -> Path:
    """Directory spilled matrices are built in.

    Parameters
    ----------
    directory : Path, optional
        An explicit directory, which takes precedence over the environment.

    Returns
    -------
    Path
        `directory` if given, else `$IM_CALCULATION_SCRATCH_DIR` if set, else
        the platform temporary directory.
    """
    if directory is not None:
        return Path(directory)
    return Path(os.environ.get(SCRATCH_DIRECTORY_VARIABLE, tempfile.gettempdir()))


def _row_blocks(n_bins: int) -> list[tuple[int, int]]:
    """Split `n_bins` rows into blocks of roughly `KONNO_BLOCK_BYTES`.

    Parameters
    ----------
    n_bins : int
        Number of rows in the matrix.

    Returns
    -------
    list of tuple of int
        `(start, stop)` row ranges covering `0..n_bins`.
    """
    rows = max(1, KONNO_BLOCK_BYTES // (n_bins * np.float32().itemsize))
    return [(start, min(start + rows, n_bins)) for start in range(0, n_bins, rows)]


def smoothing_matrix(
    n_bins: int, bandwidth: float = DEFAULT_BANDWIDTH
) -> npt.NDArray[np.float32]:
    """Build the Konno-Ohmachi smoothing matrix in memory.

    Row `c` holds the window centred on bin `c`, normalised to sum to one, so a
    spectrum is smoothed with `spectra @ matrix`.

    Parameters
    ----------
    n_bins : int
        Number of real-FFT bins the matrix smooths over.
    bandwidth : float, optional
        Bandwidth of the Konno-Ohmachi window.

    Returns
    -------
    ndarray of float32
        An `(n_bins, n_bins)` row-major matrix.
    """
    return _core._konno_ohmachi_matrix_rows(n_bins, bandwidth, 0, n_bins)


def _spilled_matrix(
    n_bins: int, bandwidth: float, directory: Path | None = None
) -> np.memmap | None:
    """Build the matrix into a scratch file and return it memory-mapped.

    The file is unlinked as soon as it is mapped, so the space is reclaimed when
    the process exits and no run can inherit a half-written matrix from a
    crashed one.

    Parameters
    ----------
    n_bins : int
        Number of real-FFT bins the matrix smooths over.
    bandwidth : float
        Bandwidth of the Konno-Ohmachi window.
    directory : Path, optional
        Where to build the file. Defaults to `resolve_scratch_directory()`.

    Returns
    -------
    np.memmap or None
        The matrix, or None if the scratch directory cannot hold it.
    """
    directory = resolve_scratch_directory(directory)
    needed = n_bins * n_bins * np.float32().itemsize
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(directory).free < needed:
            return None
        handle, name = tempfile.mkstemp(dir=directory, prefix="ko_", suffix=".npy")
        os.close(handle)
    except OSError:
        return None

    path = Path(name)
    try:
        matrix = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(n_bins, n_bins)
        )
        for start, stop in _row_blocks(n_bins):
            matrix[start:stop] = _core._konno_ohmachi_matrix_rows(
                n_bins, bandwidth, start, stop
            )
        matrix.flush()
    except OSError:
        path.unlink(missing_ok=True)
        return None

    try:
        # Unlinking an open file is a POSIX guarantee, not a Windows one.
        path.unlink()
    except OSError:  # pragma: no cover
        atexit.register(path.unlink, missing_ok=True)
    return matrix


def _matrix(
    n_bins: int, bandwidth: float, scratch_directory: Path | None = None
) -> np.ndarray | None:
    """Return the smoothing matrix for this size, building it if needed.

    Parameters
    ----------
    n_bins : int
        Number of real-FFT bins the matrix smooths over.
    bandwidth : float
        Bandwidth of the Konno-Ohmachi window.
    scratch_directory : Path, optional
        Where to spill a matrix too large to hold in memory.

    Returns
    -------
    ndarray or None
        The matrix, in memory or memory-mapped, or None if it will not fit
        anywhere and the caller should smooth matrix-free.
    """
    global _RESIDENT_BYTES

    key = (n_bins, bandwidth)
    with _LOCK:
        if key in _MATRICES:
            return _MATRICES[key]

        budget = memory_budget()
        nbytes = n_bins * n_bins * np.float32().itemsize
        if nbytes <= budget:
            # Evict oldest-first until the new matrix fits alongside what is
            # already resident. Spilled matrices cost address space, not memory,
            # so they are not counted and not evicted.
            while _MATRICES and _RESIDENT_BYTES + nbytes > budget:
                evicted = _MATRICES.pop(next(iter(_MATRICES)))
                if not isinstance(evicted, np.memmap):
                    _RESIDENT_BYTES -= evicted.nbytes
            matrix = smoothing_matrix(n_bins, bandwidth)
            _RESIDENT_BYTES += matrix.nbytes
        else:
            matrix = _spilled_matrix(n_bins, bandwidth, scratch_directory)
            if matrix is None:
                return None

        _MATRICES[key] = matrix
        return matrix


def clear_matrix_cache() -> None:
    """Drop every matrix held by this process.

    Spilled matrices are already unlinked, so their scratch space is returned as
    soon as the mapping is released.
    """
    global _RESIDENT_BYTES

    with _LOCK:
        _MATRICES.clear()
        _RESIDENT_BYTES = 0


def _apply(spectra: np.ndarray, matrix: np.ndarray) -> npt.NDArray[np.float64]:
    """Multiply a stack of spectra by a smoothing matrix.

    Parameters
    ----------
    spectra : ndarray
        Spectrum values with shape `(n_spectra, n_bins)`.
    matrix : ndarray
        An `(n_bins, n_bins)` smoothing matrix.

    Returns
    -------
    ndarray of float64
        The smoothed spectra, shape `(n_spectra, n_bins)`.
    """
    # Single precision throughout: the matrix is stored as float32, so promoting
    # it to run dgemm would cost a full copy of it (~23 s at 17 GB) to buy
    # accuracy the smoothing does not have. The weights are positive and sum to
    # one, so the accumulation cannot cancel; the error is ~1e-6 relative.
    spectra = np.ascontiguousarray(spectra, dtype=np.float32)

    if not isinstance(matrix, np.memmap):
        return np.asarray(spectra @ matrix, dtype=np.float64)

    # Contract over row blocks rather than column blocks: a row block of a
    # row-major memmap is one contiguous read, and each block is a complete
    # contribution to every output bin.
    smoothed = np.zeros(spectra.shape, dtype=np.float32)
    for start, stop in _row_blocks(matrix.shape[0]):
        smoothed += spectra[:, start:stop] @ matrix[start:stop, :]
    return np.asarray(smoothed, dtype=np.float64)


def smooth(
    spectra: np.ndarray,
    bandwidth: float = DEFAULT_BANDWIDTH,
    scratch_directory: Path | None = None,
) -> npt.NDArray[np.float64]:
    """Apply Konno-Ohmachi smoothing along the last axis.

    Parameters
    ----------
    spectra : ndarray
        Spectrum values with shape `(..., n_bins)`, where `n_bins` is the number
        of real-FFT bins.
    bandwidth : float, optional
        Bandwidth of the Konno-Ohmachi window. Lower values smooth more
        strongly.
    scratch_directory : Path, optional
        Where to build a matrix too large to hold in memory. Defaults to
        `$IM_CALCULATION_SCRATCH_DIR`, else the platform temporary directory.

    Returns
    -------
    ndarray of float64
        The smoothed spectra, with the same shape as `spectra`.
    """
    n_bins = spectra.shape[-1]
    flat = np.asarray(spectra).reshape(-1, n_bins)

    matrix = _matrix(n_bins, bandwidth, scratch_directory)
    if matrix is None:
        warnings.warn(
            RuntimeWarning(
                f"A {n_bins} bin Konno-Ohmachi matrix needs "
                f"{n_bins * n_bins * 4 / 2**30:.1f} GiB and fits neither the "
                f"{memory_budget() / 2**30:.1f} GiB memory budget nor the scratch "
                f"directory {resolve_scratch_directory(scratch_directory)}. Smoothing "
                f"matrix-free, which is "
                f"far slower. Raise ${MEMORY_BUDGET_VARIABLE} or point "
                f"${SCRATCH_DIRECTORY_VARIABLE} somewhere with more room."
            ),
        )
        smoothed = _core._konno_ohmachi_smooth(
            np.ascontiguousarray(flat, dtype=np.float64), bandwidth
        )
    else:
        smoothed = _apply(flat, matrix)

    return smoothed.reshape(spectra.shape)
