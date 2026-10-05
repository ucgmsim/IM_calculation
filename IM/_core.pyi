"""Type stub for the compiled Rust extension (`src-rust/lib.rs`).

Keep these signatures in step with the `#[pyfunction]`s in `src-rust/lib.rs`.
Every array argument must be a C-contiguous array of the stated dtype.
"""

import numpy as np

_Array1F64 = np.ndarray[tuple[int], np.dtype[np.float64]]
_Array2F64 = np.ndarray[tuple[int, int], np.dtype[np.float64]]
_Array2F32 = np.ndarray[tuple[int, int], np.dtype[np.float32]]
_Array3F64 = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]

def _arias_intensity(waveforms_py: np.ndarray, dt: float) -> _Array1F64: ...
def _cav(waveforms_py: np.ndarray, dt: float) -> _Array1F64: ...
def _konno_ohmachi_matrix_rows(
    n_bins: int, bandwidth: float, start: int, stop: int
) -> _Array2F32: ...
def _konno_ohmachi_smooth(spectra_py: np.ndarray, bandwidth: float) -> _Array2F64: ...
def _psa(
    comp_0_py: np.ndarray,
    comp_90_py: np.ndarray,
    comp_ver_py: np.ndarray,
    coefficients_py: np.ndarray,
) -> _Array3F64: ...
def _rotd(comp_0_py: np.ndarray, comp_90_py: np.ndarray) -> _Array2F64: ...
def _significant_duration(
    waveforms_py: np.ndarray, dt: float, low: float, high: float
) -> _Array1F64: ...
