use _core::{arias_intensity, significant_duration};
use ndarray::{Array1, Array2, ArrayView2};

/// Generate synthetic waveform data for benchmarking (shared with tests, to
/// keep the benchmark's data generation in sync with its regression test).
pub fn generate_waveforms(stations: usize, samples: usize) -> Array2<f64> {
    // Using a simple sine wave with some noise for realistic computation
    Array2::from_shape_fn((stations, samples), |(i, j)| {
        0.5 * ((j as f64 * 0.01 + i as f64).sin() + 0.1 * (j as f64 * 0.1).cos())
    })
}

/// The full Ds pipeline as used by the Python binding (`_significant_duration`
/// in `src-rust/lib.rs`): integrate to cumulative Arias intensity first, then
/// binary-search it for the threshold crossings. `significant_duration`
/// expects a non-decreasing cumulative intensity array, not raw waveforms.
pub fn significant_duration_from_waveforms(
    waveforms: ArrayView2<f64>,
    dt: f64,
    low: f64,
    high: f64,
) -> Array1<f64> {
    let arias = arias_intensity::cumulative_arias_intensity(waveforms, dt);
    significant_duration::significant_duration(arias.view(), dt, low, high)
}
