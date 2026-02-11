//! Cumulative absolute velocity (CAV) calculation module.
//!
//! This module provides functions to calculate the intensity of earthquake ground motion
//! based on the integral of the absolute value of acceleration.
//!
//! The formula used is:
//! $$CAV = \int_{0}^{T} |a(t)| \, dt$$
//!

use crate::constants::G;
use crate::trapz::{parallel_trapz_with_fun, trapz_with_fun};

use ndarray::prelude::*;

/// Computes the total Cumulative Absolute Velocity ($CAV$) for each row in parallel.
///
/// # Arguments
/// * `waveforms` - A 2D array view where each row is an acceleration time-series.
/// * `dt` - The time step (sampling interval) of the waveforms.
///
/// # Returns
/// An `Array1<f64>` containing the final CAV value for each station.
pub fn parallel_cav(waveforms: ArrayView2<f64>, dt: f64) -> Array1<f64> {
    G * parallel_trapz_with_fun(waveforms, dt, |x| x.abs())
}

/// Computes the total Cumulative Absolute Velocity ($CAV$) for each row using a single thread.
///
/// # Arguments
/// * `waveforms` - A 2D array view where each row is an acceleration time-series.
/// * `dt` - The time step (sampling interval) of the waveforms.
pub fn cav(waveforms: ArrayView2<f64>, dt: f64) -> Array1<f64> {
    G * trapz_with_fun(waveforms, dt, |x| x.abs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array};
    use std::f64::consts::PI;

    #[test]
    fn test_cav_constant() {
        // If a = 1.0 cm/s^2 (constant) for 1 second with dt = 1
        // Integral of |a| dt from 0 to 1 is 1.0.
        // Result should be G (9.81 cm/s)
        let waveforms = array![[1.0, 1.0]];
        let dt = 1.0;
        let result = cav(waveforms.view(), dt);

        let expected = G;
        assert_abs_diff_eq!(result[0], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_cav_negative() {
        // CAV should not care about negative values
        let waveforms = array![[-1.0, -1.0]];
        let dt = 1.0;
        let result = cav(waveforms.view(), dt);

        let expected = G;
        assert_abs_diff_eq!(result[0], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_cav_quadratic() {
        let waveform = Array::linspace(0.0, 1.0, 100).powi(2);

        let mut waveforms: Array2<f64> = Array2::zeros((1, 100));
        waveforms.assign(&waveform);

        let dt = 1.0 / 99.0;
        let result = cav(waveforms.view(), dt);
        let expected = G / 3.0;
        assert_abs_diff_eq!(result[0], expected, epsilon = 0.1);
    }

    #[test]
    fn test_cav_sin() {
        let waveform = 2.0 * Array::linspace(0.0, 2.0 * PI, 1000).sin() - 1.0;
        let mut waveforms: Array2<f64> = Array2::zeros((1, 1000));
        waveforms.assign(&waveform);

        let dt = (2.0 * PI) / 999.0;
        let result = cav(waveforms.view(), dt);
        let expected = 9.81 * (2.0 / 3.0) * (6.0 * 3.0f64.sqrt() + PI);
        assert_abs_diff_eq!(result[0], expected, epsilon = 0.1);
    }

    #[test]
    fn test_parallel_equals_sequential() {
        // This test prevents "implementation drift" where the parallel version
        // gets updated but the sequential one is forgotten.

        // Shape (2 rows, 3 cols)
        let waveforms = array![[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]];
        let dt = 1.0;

        let seq_res = cav(waveforms.view(), dt);
        let par_res = parallel_cav(waveforms.view(), dt);

        // Check if shapes match
        assert_eq!(
            seq_res.dim(),
            par_res.dim(),
            "Sequential and Parallel output shapes mismatch"
        );

        // Check if values match
        assert_abs_diff_eq!(seq_res, par_res, epsilon = 1e-10);
    }
}
