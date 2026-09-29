//! Konno-Ohmachi spectral smoothing.
//!
//! The smoothing window centred on frequency $f_c$ is
//!
//! $$W(f; f_c) = \left[\frac{\sin(b \log_{10}(f / f_c))}{b \log_{10}(f / f_c)}\right]^4$$
//!
//! where $b$ is the bandwidth, with the removable singularity $W(f_c; f_c) = 1$
//! and the limit $W(0; f_c) = 0$.

use ndarray::azip;
use ndarray::prelude::*;

/// Base-10 logarithm of every bin index.
fn log_bins(n_bins: usize) -> Array1<f64> {
    Array1::from_iter((0..n_bins).map(|bin| (bin as f64).log10()))
}

/// Writes the normalised window centred on bin `centre` into `out`.
///
/// `out` must have the same length as `logs`.
fn smoothing_window(
    centre: usize,
    logs: ArrayView1<f64>,
    bandwidth: f64,
    mut out: ArrayViewMut1<f64>,
) {
    if centre == 0 {
        out.fill(0.0);
        out[0] = 1.0;
        return;
    }

    let log_centre = logs[centre];
    azip!((value in &mut out, &log_bin in &logs) {
        let x = bandwidth * (log_bin - log_centre);
        *value = (x.sin() / x).powi(4);
    });

    // sin(0)/0 at the centre; log10(0) is -inf at bin zero. Both come out NaN
    // above and are replaced here by the limits of the window.
    out[centre] = 1.0;
    out[0] = 0.0;

    let sum = out.sum();
    out /= sum;
}

/// Rows `start..stop` of the Konno-Ohmachi smoothing matrix.
///
/// Row `c` is the set of weights that produce output bin `c`, so the result has
/// shape `(stop - start, n_bins)`, is row-major, and each row sums to one.
///
/// # Panics
///
/// Panics if `start > stop` or `stop > n_bins`.
pub fn matrix_rows(n_bins: usize, bandwidth: f64, start: usize, stop: usize) -> Array2<f32> {
    assert!(start <= stop, "start {start} is past stop {stop}");
    assert!(stop <= n_bins, "stop {stop} is past n_bins {n_bins}");

    let logs = log_bins(n_bins);
    let mut rows = Array2::<f32>::zeros((stop - start, n_bins));
    let mut window = Array1::<f64>::zeros(n_bins);

    for (row, centre) in (start..stop).enumerate() {
        smoothing_window(centre, logs.view(), bandwidth, window.view_mut());
        azip!((row_value in rows.row_mut(row), &value in &window) *row_value = value as f32);
    }

    rows
}

/// Konno-Ohmachi smoothing without materialising the matrix.
pub fn smooth(spectra: ArrayView2<f64>, bandwidth: f64) -> Array2<f64> {
    let (n_spectra, n_bins) = spectra.dim();
    let logs = log_bins(n_bins);
    let mut smoothed = Array2::<f64>::zeros((n_spectra, n_bins));
    let mut window = Array1::<f64>::zeros(n_bins);

    for centre in 0..n_bins {
        smoothing_window(centre, logs.view(), bandwidth, window.view_mut());
        smoothed.column_mut(centre).assign(&spectra.dot(&window));
    }

    smoothed
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    const BANDWIDTH: f64 = 40.0;

    /// The whole matrix, as `f64`, for tests that want the unrounded values.
    fn full_matrix(n_bins: usize, bandwidth: f64) -> Array2<f64> {
        let logs = log_bins(n_bins);
        let mut matrix = Array2::<f64>::zeros((n_bins, n_bins));
        for centre in 0..n_bins {
            smoothing_window(centre, logs.view(), bandwidth, matrix.row_mut(centre));
        }
        matrix
    }

    /// The window before normalisation, which is what the symmetry test needs.
    fn raw_window(centre: usize, bin: usize, bandwidth: f64) -> f64 {
        let x = bandwidth * ((bin as f64).log10() - (centre as f64).log10());
        (x.sin() / x).powi(4)
    }

    #[test]
    fn test_window_peaks_at_centre() {
        let matrix = full_matrix(129, BANDWIDTH);
        for centre in 1..129 {
            let row = matrix.row(centre);
            let peak = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            assert_abs_diff_eq!(row[centre], peak, epsilon = 0.0);
        }
    }

    #[test]
    fn test_window_is_zero_at_dc() {
        let matrix = full_matrix(129, BANDWIDTH);
        for centre in 1..129 {
            assert_abs_diff_eq!(matrix[[centre, 0]], 0.0, epsilon = 0.0);
        }
    }

    #[test]
    fn test_centre_zero_row_is_unit_impulse() {
        let matrix = full_matrix(129, BANDWIDTH);
        assert_abs_diff_eq!(matrix[[0, 0]], 1.0, epsilon = 0.0);
        for bin in 1..129 {
            assert_abs_diff_eq!(matrix[[0, bin]], 0.0, epsilon = 0.0);
        }
    }

    /// Each row is the weight set behind one output bin, so each sums to one.
    #[test]
    fn test_rows_sum_to_one() {
        for &n_bins in &[65usize, 129, 257] {
            let matrix = full_matrix(n_bins, BANDWIDTH);
            for centre in 0..n_bins {
                assert_abs_diff_eq!(matrix.row(centre).sum(), 1.0, epsilon = 1e-12);
            }
        }
    }

    /// The asymmetry of the stored matrix comes only from normalisation: the
    /// window itself is even in `log10(f / f_c)`.
    #[test]
    fn test_raw_window_is_symmetric() {
        for centre in 1..40 {
            for bin in 1..40 {
                if bin == centre {
                    continue;
                }
                assert_abs_diff_eq!(
                    raw_window(centre, bin, BANDWIDTH),
                    raw_window(bin, centre, BANDWIDTH),
                    epsilon = 1e-15
                );
            }
        }
    }

    /// The matrix-free path must agree with the matrix it replaces.
    #[test]
    fn test_matrix_free_matches_matrix() {
        let n_bins = 129;
        let spectra = Array2::<f64>::from_shape_fn((3, n_bins), |(row, bin)| {
            ((row * 7 + bin) as f64).sin().abs() + 0.5
        });

        let expected = spectra.dot(&full_matrix(n_bins, BANDWIDTH).t());
        let actual = smooth(spectra.view(), BANDWIDTH);

        assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
    }

    /// Row blocks tile the full matrix, which is what lets Python build a matrix
    /// far larger than memory a block at a time.
    #[test]
    fn test_row_blocks_tile_the_matrix() {
        let n_bins = 65;
        let whole = matrix_rows(n_bins, BANDWIDTH, 0, n_bins);
        for split in [0, 1, 17, 64, 65] {
            let head = matrix_rows(n_bins, BANDWIDTH, 0, split);
            let tail = matrix_rows(n_bins, BANDWIDTH, split, n_bins);
            assert_eq!(head.slice(s![.., ..]), whole.slice(s![..split, ..]));
            assert_eq!(tail.slice(s![.., ..]), whole.slice(s![split.., ..]));
        }
    }

    /// The `f64` helper above must agree with the `f32` matrix that ships, or
    /// the tests built on it would pass while `matrix_rows` drifted.
    #[test]
    fn test_full_matrix_helper_matches_matrix_rows() {
        let n_bins = 65;
        let shipped = matrix_rows(n_bins, BANDWIDTH, 0, n_bins);
        let helper = full_matrix(n_bins, BANDWIDTH).mapv(|value| value as f32);
        assert_eq!(shipped, helper);
    }

    #[test]
    fn test_degenerate_sizes() {
        let one = matrix_rows(1, BANDWIDTH, 0, 1);
        assert_abs_diff_eq!(one[[0, 0]], 1.0, epsilon = 0.0);

        let two = matrix_rows(2, BANDWIDTH, 0, 2);
        assert_abs_diff_eq!(two[[0, 0]], 1.0, epsilon = 0.0);
        assert_abs_diff_eq!(two[[0, 1]], 0.0, epsilon = 0.0);
        assert_abs_diff_eq!(two[[1, 0]], 0.0, epsilon = 0.0);
        assert_abs_diff_eq!(two[[1, 1]], 1.0, epsilon = 0.0);
    }

    proptest! {
        /// A larger bandwidth is a narrower window, so it puts less weight on
        /// bins away from the centre.
        #[test]
        fn prop_larger_bandwidth_is_narrower(centre in 4usize..60, bin in 4usize..60) {
            prop_assume!((bin as f64 / centre as f64).log10().abs() > 0.05);
            let narrow = raw_window(centre, bin, 80.0);
            let wide = raw_window(centre, bin, 20.0);
            prop_assert!(narrow <= wide + 1e-12, "narrow {narrow} exceeded wide {wide}");
        }

        /// Smoothing a flat spectrum returns it unchanged. This is the
        /// sharpest statement of the convention: it holds only because the
        /// contraction runs over the same index the weights are normalised
        /// over. Contracting over the centre index instead attenuates a flat
        /// spectrum by up to 25% near the band edges.
        #[test]
        fn prop_flat_spectrum_is_preserved(n_bins in 8usize..48) {
            let spectra = Array2::<f64>::ones((1, n_bins));
            let smoothed = smooth(spectra.view(), BANDWIDTH);
            for bin in 0..n_bins {
                prop_assert!((smoothed[[0, bin]] - 1.0).abs() < 1e-12);
            }
        }
    }
}
