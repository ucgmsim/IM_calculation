"""Konno-Ohmachi spectral smoothing.

Smoothing a spectrum of `n` real-FFT bins is a product with an `n x n` matrix.
That matrix grows quadratically -- a 22 minute record at 100 Hz needs 17 GB --
so this module decides, per size, where to put it:

- small enough to hold, which is every observed record: build it in memory and
  keep it for the life of the process;
- too large for memory: build it into an unnamed scratch file, so it costs
  address space rather than resident memory and cannot outlive the process;
- no room even for that: fall back to the matrix-free kernel, which is correct
  but re-derives every window on every call.

Each output bin is a weighted average of the spectrum whose weights sum to one,
so a flat spectrum is returned unchanged. This is obspy's direct smoothing path;
its matrix path contracts over the other index and attenuates a flat spectrum by
up to 25% near the band edges.

Nothing is cached between runs. Once the matrix is reused across enough spectra
to be worth having at all, building it is a small fraction of applying it -- for
the largest records, under a minute against hours -- so persisting it would buy
about a percent in exchange for stale-cache invalidation, bandwidth metadata and
a cross-process locking protocol.
"""

import atexit
import contextlib
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

_CLEANUP = contextlib.ExitStack()
"""Holds every scratch handle open for the life of the process."""
atexit.register(_CLEANUP.close)

_RESIDENT: dict[tuple[int, float], np.ndarray] = {}
"""Matrices held in memory, oldest first. Their total is capped by the budget."""

_SPILLED: dict[tuple[int, float], np.memmap] = {}
"""Matrices mapped from scratch files. They cost address space, not memory."""

_LOCK = threading.Lock()


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


def _cached(key: tuple[int, float]) -> np.ndarray | None:
    """Look up a matrix across both tiers.

    Parameters
    ----------
    key : tuple of int and float
        The `(n_bins, bandwidth)` the matrix was built for.

    Returns
    -------
    ndarray or None
        The matrix if this process already holds one, else None. Two lookups
        rather than `a or b`, because an array has no truth value.
    """
    matrix = _RESIDENT.get(key)
    return matrix if matrix is not None else _SPILLED.get(key)


def _resident_bytes() -> int:
    """Total size of the matrices currently held in memory.

    Returns
    -------
    int
        Bytes across `_RESIDENT`. Spilled matrices are not counted: they are
        pages of a scratch file, not resident memory.
    """
    return sum(matrix.nbytes for matrix in _RESIDENT.values())


def smoothing_matrix(
    n_bins: int, bandwidth: float = DEFAULT_BANDWIDTH
) -> npt.NDArray[np.float32]:
    """Build the Konno-Ohmachi smoothing matrix in memory.

    Row `c` holds the window centred on bin `c`, normalised to sum to one, so it
    is the set of weights that produce output bin `c` and a spectrum is smoothed
    with `spectra @ matrix.T`.

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

    The file is never given a name: `tempfile.TemporaryFile` hands back a handle
    to an already-unlinked file, so reclaiming the space is the operating
    system's job. No run can inherit a half-written matrix from a crashed one,
    and there is nothing to clean up even when the process is killed outright --
    which an exit hook would not have survived.

    Parameters
    ----------
    n_bins : int
        Number of real-FFT bins the matrix smooths over.
    bandwidth : float
        Bandwidth of the Konno-Ohmachi window.
    directory : Path, optional
        Which filesystem to build on. Defaults to `resolve_scratch_directory()`.

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
        # The handle outlives this call: the mapping is read until the matrix is
        # dropped, and on Windows the file lives only as long as its handles.
        # `_CLEANUP` is the context manager -- one that closes at process exit.
        handle = _CLEANUP.enter_context(tempfile.TemporaryFile(dir=directory))  # noqa: SIM115
        matrix = np.memmap(handle, dtype=np.float32, mode="w+", shape=(n_bins, n_bins))
        for start, stop in _row_blocks(n_bins):
            matrix[start:stop] = _core._konno_ohmachi_matrix_rows(
                n_bins, bandwidth, start, stop
            )
        matrix.flush()
    except OSError:
        return None
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
    key = (n_bins, bandwidth)
    # Hit without taking the lock: a build holds it for the whole of an O(n^2)
    # matrix, and every other worker thread needs only the dict lookup.
    if (matrix := _cached(key)) is not None:
        return matrix

    with _LOCK:
        # Another thread may have finished the build while we waited for it.
        if (matrix := _cached(key)) is not None:
            return matrix

        budget = memory_budget()
        nbytes = n_bins * n_bins * np.float32().itemsize
        if nbytes > budget:
            spilled = _spilled_matrix(n_bins, bandwidth, scratch_directory)
            if spilled is not None:
                _SPILLED[key] = spilled
            return spilled

        # Drop the oldest resident matrices until this one fits beside them.
        while _RESIDENT and _resident_bytes() + nbytes > budget:
            del _RESIDENT[next(iter(_RESIDENT))]
        _RESIDENT[key] = smoothing_matrix(n_bins, bandwidth)
        return _RESIDENT[key]


def clear_matrix_cache() -> None:
    """Drop every matrix held by this process.

    Spilled matrices are already unlinked, so their scratch space is returned as
    soon as the mapping is released.
    """
    with _LOCK:
        _RESIDENT.clear()
        _SPILLED.clear()


def _apply(spectra: np.ndarray, matrix: np.ndarray) -> npt.NDArray[np.float64]:
    """Smooth a stack of spectra with a smoothing matrix.

    Row `c` of `matrix` holds the weights for output bin `c`, so this computes
    `spectra @ matrix.T`.

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

    # Block over the output bins. A row block of a row-major matrix is one
    # contiguous read and carries every weight its output bins need, so each
    # block of the result is written once rather than accumulated into. Anything
    # within `KONNO_BLOCK_BYTES` is a single block, which is the plain product --
    # so there is no separate in-memory path to keep in step.
    smoothed = np.empty(spectra.shape, dtype=np.float32)
    for start, stop in _row_blocks(matrix.shape[0]):
        smoothed[:, start:stop] = spectra @ matrix[start:stop, :].T
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
