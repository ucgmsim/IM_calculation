//! Konno-Ohmachi spectral smoothing.
//!
//! The smoothing window centred on frequency $f_c$ is
//!
//! $$W(f; f_c) = \left[\frac{\sin(b \log_{10}(f / f_c))}{b \log_{10}(f / f_c)}\right]^4$$
//!
//! where $b$ is the bandwidth, with the removable singularity $W(f_c; f_c) = 1$
//! and the limit $W(0; f_c) = 0$. The window is constant-width on a logarithmic
//! frequency axis, so a small bandwidth smooths strongly.
//!
//! Only the ratio $f / f_c$ appears. For the real-FFT bin frequencies
//! $f_k = k / (2 (n - 1) \Delta t)$ that ratio is exactly $j / k$, so the window
//! depends on the number of bins alone -- not on $\Delta t$, and not on the
//! frequency values themselves. Everything here is therefore indexed by bin.
//!
//! Row $c$ of the matrix holds the window centred on bin $c$, normalised to sum
//! to one, so a spectrum is smoothed with `spectra.dot(&matrix)` and the
//! contraction runs over the *centre* index. This matches obspy's
//! `calculate_smoothing_matrix` / `apply_smoothing_matrix` pair, which is what
//! this package's Fourier amplitude spectra are defined against.

use ndarray::prelude::*;

/// Base-10 logarithm of every bin index.
///
/// Element zero is `-inf` and is never read: bin zero is special-cased both as a
/// centre and as an evaluation point.
fn log_bins(n_bins: usize) -> Array1<f64> {
    Array1::from_iter((0..n_bins).map(|bin| (bin as f64).log10()))
}

/// Writes the normalised window centred on bin `centre` into `out`.
///
/// `out` must have the same length as `logs`. The steps below are ordered to
/// match obspy exactly, because the fix-ups at bins `centre` and `0` overwrite
/// the `0/0` and `-inf` the formula produces there.
fn smoothing_window(centre: usize, logs: ArrayView1<f64>, bandwidth: f64, out: &mut [f64]) {
    let n_bins = logs.len();

    // A centre of zero has no ratio to take. obspy returns the unit impulse
    // here *before* normalising; it already sums to one, so the result is the
    // same either way.
    if centre == 0 {
        out.fill(0.0);
        out[0] = 1.0;
        return;
    }

    let log_centre = logs[centre];
    let mut sum = 0.0;
    for bin in 1..n_bins {
        // Skipped rather than computed and overwritten: sin(0)/0 is NaN, and a
        // NaN added to `sum` would poison the whole row.
        if bin == centre {
            continue;
        }
        let x = bandwidth * (logs[bin] - log_centre);
        let value = (x.sin() / x).powi(4);
        out[bin] = value;
        sum += value;
    }

    // The limit as f -> f_c is one.
    out[centre] = 1.0;
    sum += 1.0;
    // The limit as f -> 0 is zero, and `logs[0]` is -inf. Bin zero is outside
    // the loop above, so `out` may still hold the previous call's value here.
    out[0] = 0.0;

    let scale = 1.0 / sum;
    for value in out.iter_mut() {
        *value *= scale;
    }
}

/// Rows `start..stop` of the Konno-Ohmachi smoothing matrix.
///
/// The result has shape `(stop - start, n_bins)` and is row-major, so a caller
/// building the whole matrix can fill it a block of rows at a time without ever
/// holding all of it. Each row sums to one.
///
/// # Panics
///
/// Panics if `start > stop` or `stop > n_bins`.
pub fn matrix_rows(n_bins: usize, bandwidth: f64, start: usize, stop: usize) -> Array2<f32> {
    assert!(start <= stop, "start {start} is past stop {stop}");
    assert!(stop <= n_bins, "stop {stop} is past n_bins {n_bins}");

    let logs = log_bins(n_bins);
    let mut rows = Array2::<f32>::zeros((stop - start, n_bins));
    let mut window = vec![0.0f64; n_bins];

    for (row, centre) in (start..stop).enumerate() {
        smoothing_window(centre, logs.view(), bandwidth, &mut window);
        let mut out_row = rows.row_mut(row);
        for (out, &value) in out_row.iter_mut().zip(window.iter()) {
            *out = value as f32;
        }
    }

    rows
}

/// Konno-Ohmachi smoothing without materialising the matrix.
///
/// `spectra` has shape `(n_spectra, n_bins)` and the result has the same shape.
/// Equals `spectra.dot(&matrix_rows(n_bins, bandwidth, 0, n_bins))` up to the
/// `f32` rounding of the matrix.
///
/// Because the contraction runs over the centre index, this accumulates one
/// outer product per centre rather than taking one inner product per output
/// bin. It allocates a single window buffer and no matrix, which is the only
/// way to smooth a spectrum whose matrix would not fit on the machine -- but it
/// re-evaluates every window on every call, so prefer the matrix whenever one
/// can be held.
pub fn smooth(spectra: ArrayView2<f64>, bandwidth: f64) -> Array2<f64> {
    let (n_spectra, n_bins) = spectra.dim();
    let logs = log_bins(n_bins);
    let mut smoothed = Array2::<f64>::zeros((n_spectra, n_bins));
    let mut window = vec![0.0f64; n_bins];

    for centre in 0..n_bins {
        smoothing_window(centre, logs.view(), bandwidth, &mut window);
        for (mut out_row, spectrum) in smoothed.rows_mut().into_iter().zip(spectra.rows()) {
            let weight = spectrum[centre];
            if weight == 0.0 {
                continue;
            }
            for (out, &value) in out_row.iter_mut().zip(window.iter()) {
                *out += weight * value;
            }
        }
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
        let mut window = vec![0.0f64; n_bins];
        for centre in 0..n_bins {
            smoothing_window(centre, logs.view(), bandwidth, &mut window);
            matrix
                .row_mut(centre)
                .assign(&ArrayView1::from(&window[..]));
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

    /// Rows sum to one, not columns. If anyone ever "fixes" the normalisation to
    /// run over the evaluation index this fails, and every FAS value moves.
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

        let expected = spectra.dot(&full_matrix(n_bins, BANDWIDTH));
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

        /// Smoothing a flat spectrum gives the column sums. It is *not* the
        /// constant back: the weights are normalised over the evaluation index
        /// but contracted over the centre index, so they need not sum to one
        /// along the contraction. This is the sharpest statement of the
        /// convention, and the first thing to break if it ever changes.
        #[test]
        fn prop_flat_spectrum_gives_column_sums(n_bins in 8usize..48) {
            let spectra = Array2::<f64>::ones((1, n_bins));
            let smoothed = smooth(spectra.view(), BANDWIDTH);
            let column_sums = full_matrix(n_bins, BANDWIDTH).sum_axis(Axis(0));
            for bin in 0..n_bins {
                prop_assert!((smoothed[[0, bin]] - column_sums[bin]).abs() < 1e-12);
            }
        }
    }
}
