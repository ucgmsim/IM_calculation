//! Arias Intensity ($I_A$) calculation module.
//!
//! This module provides functions to calculate the intensity of earthquake ground motion
//! based on the integral of the square of acceleration.
//!
//! The formula used is:
//! $$I_A = \frac{\pi}{2g} \int_{0}^{T} a(t)^2 \, dt$$
//!
//! Where $g$ is the acceleration due to gravity (set here to $981.0 \text{ cm/s}^2$).

use crate::trapz::{
    cumulative_trapz_with_fun, parallel_cumulative_trapz_with_fun, parallel_trapz_with_fun,
    trapz_with_fun,
};

use crate::constants::G;
use ndarray::prelude::*;
use std::f64::consts::PI;

/// Precomputed scaling factor: $\frac{\pi}{2g}$
const ARIAS_CONSTANT: f64 = PI / (2.0 * G);

/// Computes the total Arias Intensity ($I_A$) for each row in parallel.
///
/// # Arguments
/// * `waveforms` - A 2D array view where each row is an acceleration time-series.
/// * `dt` - The time step (sampling interval) of the waveforms.
///
/// # Returns
/// An `Array1<f64>` containing the final $I_A$ value for each station.
pub fn parallel_arias_intensity(waveforms: ArrayView2<f64>, dt: f64) -> Array1<f64> {
    ARIAS_CONSTANT * parallel_trapz_with_fun(waveforms, dt, |x| x * x)
}

/// Computes the total Arias Intensity ($I_A$) for each row using a single thread.
///
/// # Arguments
/// * `waveforms` - A 2D array view where each row is an acceleration time-series.
/// * `dt` - The time step (sampling interval) of the waveforms.
pub fn arias_intensity(waveforms: ArrayView2<f64>, dt: f64) -> Array1<f64> {
    ARIAS_CONSTANT * trapz_with_fun(waveforms, dt, |x| x * x)
}

/// Computes the cumulative Arias Intensity time-history for each row in parallel.
///
/// This returns the "Husid plot" data, showing how the intensity builds over time.
///
/// # Arguments
/// * `waveforms` - A 2D array view where each row is an acceleration time-series.
/// * `dt` - The time step (sampling interval) of the waveforms.
///
/// # Returns
/// An `Array2<f64>` of the same shape as `waveforms`, representing the intensity accumulated at each timestep.
pub fn parallel_cumulative_arias_intensity(waveforms: ArrayView2<f64>, dt: f64) -> Array2<f64> {
    ARIAS_CONSTANT * parallel_cumulative_trapz_with_fun(waveforms, dt, |x| x * x)
}

/// Computes the cumulative Arias Intensity time-history for each row using a single thread.
///
/// # Arguments
/// * `waveforms` - A 2D array view where each row is an acceleration time-series.
/// * `dt` - The time step (sampling interval) of the waveforms.
pub fn cumulative_arias_intensity(waveforms: ArrayView2<f64>, dt: f64) -> Array2<f64> {
    ARIAS_CONSTANT * cumulative_trapz_with_fun(waveforms, dt, |x| x * x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_arias_physical_constant() {
        // If a = 1.0 cm/s^2 (constant) for 1 second with dt=1
        // Integral of a^2 dt from 0 to 1 is 1.0.
        // Result should be PI / (2.0 * 981.0)
        let waveforms = array![[1.0, 1.0]];
        let dt = 1.0;
        let result = arias_intensity(waveforms.view(), dt);

        let expected = ARIAS_CONSTANT;
        assert_abs_diff_eq!(result[0], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_husid_plot_monotonicity() {
        // Even with negative acceleration, the intensity must increase
        let waveforms = array![[1.0, -2.0, 3.0, -4.0]];
        let dt = 0.1;
        let husid = cumulative_arias_intensity(waveforms.view(), dt);

        // Check each step is >= previous step
        for window in husid.windows((1, 2)).into_iter() {
            assert!(
                window[[0, 1]] >= window[[0, 0]],
                "Husid plot must be monotonic"
            );
        }
    }

    #[test]
    fn test_output_shapes() {
        let waveforms = Array2::<f64>::zeros((3, 100));
        let dt = 0.01;

        let total = arias_intensity(waveforms.view(), dt);
        let cumulative = cumulative_arias_intensity(waveforms.view(), dt);

        assert_eq!(total.shape(), &[3]);
        assert_eq!(cumulative.shape(), &[3, 100]);
    }

    #[test]
    fn test_parallel_equals_sequential_total_intensity() {
        let waveforms = array![[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]];
        let dt = 0.02;

        let seq_res = arias_intensity(waveforms.view(), dt);
        let par_res = parallel_arias_intensity(waveforms.view(), dt);

        assert_abs_diff_eq!(seq_res, par_res, epsilon = 1e-10);
    }

    #[test]
    fn test_parallel_equals_sequential_cumulative_intensity() {
        let waveforms = array![[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]];
        let dt = 0.02;

        let seq_res = cumulative_arias_intensity(waveforms.view(), dt);
        let par_res = parallel_cumulative_arias_intensity(waveforms.view(), dt);

        assert_abs_diff_eq!(seq_res, par_res, epsilon = 1e-10);
    }
}
