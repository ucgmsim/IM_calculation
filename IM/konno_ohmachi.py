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

Matrices live in a `MatrixStore`, one per process, and are not cached between
runs. Under `fork` a child inherits the parent's store and shares it
copy-on-write, so a pool should warm the store before forking -- see
`MatrixStore.warm`. Under `spawn` nothing is shared and every worker rebuilds.
"""

import contextlib
import os
import shutil
import tempfile
import threading
import warnings
from collections.abc import Iterable
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


def bins_for_samples(n_samples: int) -> int:
    """Number of real-FFT bins a record of `n_samples` is smoothed over.

    The matrix a record needs is fixed by this, so it is what `MatrixStore.warm`
    takes. Mirrors the zero-padding `fourier_amplitude_spectra` applies.

    Parameters
    ----------
    n_samples : int
        Number of timesteps in the record.

    Returns
    -------
    int
        Bin count of the real FFT, after padding up to a power of two.
    """
    return 2 ** int(np.ceil(np.log2(n_samples))) // 2 + 1


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


class MatrixStore:
    """Konno-Ohmachi matrices held for the life of a process.

    A matrix is fixed by `(n_bins, bandwidth)` alone, so the store is keyed on
    that and never needs invalidating. Where each one is kept depends on its
    size, measured against `memory_budget`:

    - within budget: built in memory and kept, evicting the oldest as needed;
    - over budget: built into an unnamed scratch file and memory-mapped, so it
      costs address space instead;
    - too large for even that: not built at all, and `get` returns None so the
      caller can fall back to the matrix-free kernel.

    The store is per-process. Under `fork` the children inherit whatever the
    parent had built, sharing it copy-on-write, so warming before forking is
    worth doing -- see `warm`. Under `spawn` nothing is inherited and every
    worker builds its own.

    Parameters
    ----------
    memory_budget : int, optional
        Largest matrix, in bytes, to hold in memory. Defaults to
        `$IM_CALCULATION_KO_MEMORY_BUDGET`, else `DEFAULT_MEMORY_BUDGET`.
    scratch_directory : Path, optional
        Which filesystem to spill onto. Defaults to
        `$IM_CALCULATION_SCRATCH_DIR`, else the platform temporary directory.
    """

    def __init__(
        self,
        memory_budget: int | None = None,
        scratch_directory: Path | None = None,
    ) -> None:
        """Create an empty store.

        Parameters
        ----------
        memory_budget : int, optional
            Largest matrix, in bytes, to hold in memory.
        scratch_directory : Path, optional
            Which filesystem to spill onto.
        """
        self._memory_budget = memory_budget
        self._scratch_directory = scratch_directory
        self._resident: dict[tuple[int, float], np.ndarray] = {}
        self._spilled: dict[tuple[int, float], np.memmap] = {}
        self._handles = contextlib.ExitStack()
        self._lock = threading.Lock()

    @property
    def memory_budget(self) -> int:
        """Largest matrix held in memory rather than spilled.

        Returns
        -------
        int
            The constructor argument if given, else
            `$IM_CALCULATION_KO_MEMORY_BUDGET`, else `DEFAULT_MEMORY_BUDGET`.
        """
        if self._memory_budget is not None:
            return self._memory_budget
        return int(os.environ.get(MEMORY_BUDGET_VARIABLE, DEFAULT_MEMORY_BUDGET))

    @property
    def scratch_directory(self) -> Path:
        """Directory spilled matrices are built in.

        Returns
        -------
        Path
            The constructor argument if given, else
            `$IM_CALCULATION_SCRATCH_DIR`, else the platform temporary directory.
        """
        if self._scratch_directory is not None:
            return Path(self._scratch_directory)
        return Path(os.environ.get(SCRATCH_DIRECTORY_VARIABLE, tempfile.gettempdir()))

    def __len__(self) -> int:
        """Number of matrices currently held, across both tiers.

        Returns
        -------
        int
            Count of resident plus spilled matrices.
        """
        return len(self._resident) + len(self._spilled)

    def get(
        self,
        n_bins: int,
        bandwidth: float = DEFAULT_BANDWIDTH,
        scratch_directory: Path | None = None,
    ) -> np.ndarray | None:
        """Return the matrix for this size, building it if the store lacks one.

        Parameters
        ----------
        n_bins : int
            Number of real-FFT bins the matrix smooths over.
        bandwidth : float, optional
            Bandwidth of the Konno-Ohmachi window.
        scratch_directory : Path, optional
            Overrides `self.scratch_directory` for this call.

        Returns
        -------
        ndarray or None
            The matrix, in memory or memory-mapped, or None if it fits nowhere
            and the caller should smooth matrix-free.
        """
        key = (n_bins, bandwidth)
        # Hit without taking the lock: a build holds it for the whole of an
        # O(n^2) matrix, and every other worker thread needs only the lookup.
        if (matrix := self._lookup(key)) is not None:
            return matrix

        with self._lock:
            # Another thread may have finished the build while we waited.
            if (matrix := self._lookup(key)) is not None:
                return matrix

            budget = self.memory_budget
            nbytes = n_bins * n_bins * np.float32().itemsize
            if nbytes > budget:
                spilled = self._spill(n_bins, bandwidth, scratch_directory)
                if spilled is not None:
                    self._spilled[key] = spilled
                return spilled

            # Drop the oldest resident matrices until this one fits beside them.
            while self._resident and self._resident_bytes() + nbytes > budget:
                del self._resident[next(iter(self._resident))]
            self._resident[key] = smoothing_matrix(n_bins, bandwidth)
            return self._resident[key]

    def warm(
        self, n_bins: int | Iterable[int], bandwidth: float = DEFAULT_BANDWIDTH
    ) -> None:
        """Build matrices ahead of time.

        The reason to call this is `fork`: a child inherits whatever the parent
        has already built and shares it copy-on-write, so warming in the parent
        before starting a pool leaves one physical copy instead of one per
        worker. Warming after the fork does nothing for the other children.

        Use `bins_for_samples` to turn record lengths into bin counts.

        Parameters
        ----------
        n_bins : int or iterable of int
            Bin counts to build for.
        bandwidth : float, optional
            Bandwidth of the Konno-Ohmachi window.
        """
        sizes = [n_bins] if isinstance(n_bins, int) else n_bins
        for size in sizes:
            self.get(size, bandwidth)

    def clear(self) -> None:
        """Drop every matrix held, and close the scratch files behind them.

        Those files are already unlinked, so their space returns to the
        filesystem as soon as the handles close.
        """
        with self._lock:
            self._resident.clear()
            self._spilled.clear()
            self._handles.close()

    def _lookup(self, key: tuple[int, float]) -> np.ndarray | None:
        """Look up a matrix across both tiers.

        Parameters
        ----------
        key : tuple of int and float
            The `(n_bins, bandwidth)` the matrix was built for.

        Returns
        -------
        ndarray or None
            The matrix if the store holds one. Two lookups rather than `a or b`,
            because an array has no truth value.
        """
        matrix = self._resident.get(key)
        return matrix if matrix is not None else self._spilled.get(key)

    def _resident_bytes(self) -> int:
        """Total size of the matrices currently held in memory.

        Returns
        -------
        int
            Bytes across the resident tier. Spilled matrices are not counted:
            they are pages of a scratch file, not resident memory.
        """
        return sum(matrix.nbytes for matrix in self._resident.values())

    def _spill(
        self, n_bins: int, bandwidth: float, directory: Path | None = None
    ) -> np.memmap | None:
        """Build the matrix into a scratch file and return it memory-mapped.

        The file is never given a name: `tempfile.TemporaryFile` hands back a
        handle to an already-unlinked file, so reclaiming the space is the
        operating system's job. No run can inherit a half-written matrix from a
        crashed one, and there is nothing to clean up even when the process is
        killed outright -- which an exit hook would not have survived.

        Parameters
        ----------
        n_bins : int
            Number of real-FFT bins the matrix smooths over.
        bandwidth : float
            Bandwidth of the Konno-Ohmachi window.
        directory : Path, optional
            Overrides `self.scratch_directory`.

        Returns
        -------
        np.memmap or None
            The matrix, or None if the scratch directory cannot hold it.
        """
        directory = Path(directory) if directory is not None else self.scratch_directory
        needed = n_bins * n_bins * np.float32().itemsize
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(directory).free < needed:
                return None
            # The handle outlives this call: the mapping is read until the
            # matrix is dropped, and on Windows the file lives only as long as
            # its handles. `self._handles` is the context manager holding them.
            scratch = tempfile.TemporaryFile(dir=directory)  # noqa: SIM115
            handle = self._handles.enter_context(scratch)
            matrix = np.memmap(
                handle, dtype=np.float32, mode="w+", shape=(n_bins, n_bins)
            )
            for start, stop in _row_blocks(n_bins):
                matrix[start:stop] = _core._konno_ohmachi_matrix_rows(
                    n_bins, bandwidth, start, stop
                )
            matrix.flush()
        except OSError:
            return None
        return matrix


MATRICES = MatrixStore()
"""The store `smooth` uses when no other is given."""


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


def clear_matrix_cache() -> None:
    """Drop every matrix held by the default store."""
    MATRICES.clear()


def smooth(
    spectra: np.ndarray,
    bandwidth: float = DEFAULT_BANDWIDTH,
    scratch_directory: Path | None = None,
    store: MatrixStore | None = None,
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
    store : MatrixStore, optional
        Where matrices are kept. Defaults to the module-level `MATRICES`.

    Returns
    -------
    ndarray of float64
        The smoothed spectra, with the same shape as `spectra`.
    """
    store = store if store is not None else MATRICES
    n_bins = spectra.shape[-1]
    flat = np.asarray(spectra).reshape(-1, n_bins)

    matrix = store.get(n_bins, bandwidth, scratch_directory)
    if matrix is None:
        warnings.warn(
            RuntimeWarning(
                f"A {n_bins} bin Konno-Ohmachi matrix needs "
                f"{n_bins * n_bins * 4 / 2**30:.1f} GiB and fits neither the "
                f"{store.memory_budget / 2**30:.1f} GiB memory budget nor the "
                f"scratch directory {store.scratch_directory}. Smoothing "
                f"matrix-free, which is far slower. Raise "
                f"${MEMORY_BUDGET_VARIABLE} or point ${SCRATCH_DIRECTORY_VARIABLE} "
                f"somewhere with more room."
            ),
        )
        smoothed = _core._konno_ohmachi_smooth(
            np.ascontiguousarray(flat, dtype=np.float64), bandwidth
        )
    else:
        smoothed = _apply(flat, matrix)

    return smoothed.reshape(spectra.shape)
