//! Numerical integration module using the Trapezium Rule with zero-crossing correction.
//!
//! This module provides utilities for both total and cumulative integration of 1D and 2D arrays,
//! with support for mapping a function $f(x)$ over the values before integration.
//!
//! A key feature of this implementation is the handling of zero-crossings, which ensures that
//! rectified functions (like $f(x) = |x|$) are integrated with geometric precision.

use crate::utils::{parallel_map_rows, parallel_reduce_rows};
use ndarray::prelude::*;

/// Calculates the contribution of a single step to the trapezium integral.
///
/// If a zero-crossing is detected between `v1` and `v2`, the interval is split
/// at the estimated root to improve precision for non-linear functions like absolute value.
///
/// # Arguments
/// * `v1` - Value at the start of the interval.
/// * `v2` - Value at the end of the interval.
/// * `dt` - The time step between points.
/// * `f` - Function to apply to the values before integration.
///
/// # Returns
/// The area of the trapezium (scaled by 2.0).
fn trapz_step<F>(v1: f64, v2: f64, dt: f64, f: &F) -> f64
where
    F: Fn(f64) -> f64,
{
    if v1.min(v2) >= 0.0 || v1.max(v2) <= 0.0 {
        // Standard case: no zero crossing
        dt * (f(v1) + f(v2))
    } else {
        // Zero-crossing correction: split the interval at x0
        let inv_slope = dt / (v2 - v1);
        let x0 = -v1 * inv_slope;
        x0 * f(v1) + (dt - x0) * f(v2)
    }
}

/// Integrates a 1D waveform using the trapezium rule.
///
/// The result is calculated as:
/// $$\text{Area} = \frac{1}{2} \sum_{i=0}^{n-1} \text{trapz\_step}(v_i, v_{i+1}, dt, f)$$
fn trapz_one_with_fun<F>(waveform: ArrayView1<f64>, dt: f64, f: &F) -> f64
where
    F: Fn(f64) -> f64,
{
    let sum = waveform.windows(2).into_iter().fold(0.0, |sum, window| {
        let v1 = window[0];
        let v2 = window[1];
        sum + trapz_step(v1, v2, dt, f)
    });

    0.5 * sum
}

/// Computes the cumulative integral of a 1D waveform.
///
/// # Returns
/// An `Array1` of the same length as the input, where the first element is $0.0$
/// and each subsequent element $j$ is the integral from index $0$ to $j$.
fn cumulative_trapz_one_with_fun<F>(waveform: ArrayView1<f64>, dt: f64, f: F) -> Array1<f64>
where
    F: Fn(f64) -> f64,
{
    let nt = waveform.dim();
    let mut cumulative_sum = Array1::zeros(nt);
    let mut sum = 0.0;
    for (window, v) in waveform
        .windows(2)
        .into_iter()
        .zip(cumulative_sum.iter_mut().skip(1))
    {
        let v1 = window[0];
        let v2 = window[1];
        sum += trapz_step(v1, v2, dt, &f);
        *v = sum;
    }
    0.5 * cumulative_sum
}

/// Computes the total integral for each row of a 2D array in parallel.
///
/// # Arguments
/// * `waveforms` - 2D array where each row is a separate signal.
/// * `dt` - The timestep.
/// * `f` - Function to apply to values (e.g., `|x| x * x` for arias intensity).
pub fn parallel_trapz_with_fun<F>(waveforms: ArrayView2<f64>, dt: f64, f: F) -> Array1<f64>
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    parallel_reduce_rows(waveforms, |waveform| trapz_one_with_fun(waveform, dt, &f))
}

/// Computes the total integral for each row of a 2D array.
///
/// This is the sequential version of [`parallel_trapz_with_fun`].
pub fn trapz_with_fun<F>(waveforms: ArrayView2<f64>, dt: f64, f: F) -> Array1<f64>
where
    F: Fn(f64) -> f64,
{
    waveforms.map_axis(Axis(1), |waveform| trapz_one_with_fun(waveform, dt, &f))
}

/// Computes the cumulative integral for each row of a 2D array in parallel.
///
/// # Returns
/// An `Array2` of the same shape as `waveforms`.
pub fn parallel_cumulative_trapz_with_fun<F>(
    waveforms: ArrayView2<f64>,
    dt: f64,
    f: F,
) -> Array2<f64>
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    parallel_map_rows(waveforms, |waveform| {
        cumulative_trapz_one_with_fun(waveform, dt, &f)
    })
}

/// Computes the cumulative integral for each row of a 2D array.
///
/// This is the sequential version of [`parallel_cumulative_trapz_with_fun`].
pub fn cumulative_trapz_with_fun<F>(waveforms: ArrayView2<f64>, dt: f64, f: F) -> Array2<f64>
where
    F: Fn(f64) -> f64,
{
    let mut out = Array2::zeros(waveforms.dim());
    out.axis_iter_mut(Axis(0))
        .zip(waveforms.axis_iter(Axis(0)))
        .for_each(|(mut out_row, in_row)| {
            let result_row = cumulative_trapz_one_with_fun(in_row, dt, &f);
            out_row.assign(&result_row);
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // A helper to define a linear function for testing
    fn identity(x: f64) -> f64 {
        x
    }
    fn rectified(x: f64) -> f64 {
        x.abs()
    }

    #[test]
    fn test_trapz_step_zero_crossing_precision() {
        //
        // Scenario: A signal goes from -1.0 to 1.0 over dt=2.0.
        // It crosses zero exactly in the middle.
        // We integrate |x| (rectified).

        // STANDARD TRAPEZIUM MATH:
        // Area = 0.5 * dt * (|v1| + |v2|)
        //      = 0.5 * 2.0 * (1.0 + 1.0) = 2.0
        // This is WRONG for |x|. The actual area of two triangles is 1.0.

        // `trapz_step` detects the crossing and splits the interval.
        // It should return exactly 1.0 * 2 (since the 0.5 factor is applied later).

        let v1 = -1.0;
        let v2 = 1.0;
        let dt = 2.0;

        // Note: trapz_step returns 2x the area (the 0.5 is in trapz_one)
        let result_2x = trapz_step(v1, v2, dt, &rectified);

        // We expect the area to be 1.0, so the function returns 2.0
        assert_abs_diff_eq!(result_2x, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trapz_matches_analytical_result() {
        // Integrate f(x) = x over [0, 5]
        // Analytical integral of x is x^2/2.
        // From 0 to 5, Area = 12.5.

        let waveform = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let dt = 1.0;

        let area = trapz_one_with_fun(waveform.view(), dt, &identity);

        assert_abs_diff_eq!(area, 12.5, epsilon = 1e-10);
    }

    #[test]
    fn test_cumulative_matches_total() {
        // Guarantee: The last point of a cumulative integral MUST
        // equal the result of the total integral.
        let waveform = array![0.5, -0.5, 1.0, -2.0, 3.0];
        let dt = 0.1;

        let total = trapz_one_with_fun(waveform.view(), dt, &rectified);
        let cumulative = cumulative_trapz_one_with_fun(waveform.view(), dt, &rectified);

        assert_abs_diff_eq!(*cumulative.last().unwrap(), total, epsilon = 1e-10);
    }

    #[test]
    fn test_linearity_scaling() {
        // Guarantee: If we double dt, the area should double.
        // This ensures variables aren't hardcoded.
        let waveform = array![1.0, 2.0, 3.0];

        let area_dt1 = trapz_one_with_fun(waveform.view(), 1.0, &identity);
        let area_dt2 = trapz_one_with_fun(waveform.view(), 2.0, &identity);

        assert_abs_diff_eq!(area_dt2, area_dt1 * 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_parallel_equals_sequential() {
        // This test prevents "implementation drift" where the parallel version
        // gets updated but the sequential one is forgotten.

        // Shape (2 rows, 3 cols)
        let waveforms = array![[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]];
        let dt = 1.0;

        let seq_res = trapz_with_fun(waveforms.view(), dt, identity);
        let par_res = parallel_trapz_with_fun(waveforms.view(), dt, identity);

        // Check if shapes match
        assert_eq!(
            seq_res.dim(),
            par_res.dim(),
            "Sequential and Parallel output shapes mismatch"
        );

        // Check if values match
        assert_abs_diff_eq!(seq_res, par_res, epsilon = 1e-10);
    }

    #[test]
    fn test_orientation_consistency() {
        // Checks code integrates along rows (Axis 1).
        let waveforms = array![
            [1.0, 1.0, 1.0], // Constant 1, Length 3, Area should be 2.0
            [2.0, 2.0, 2.0]  // Constant 2, Length 3, Area should be 4.0
        ];
        let dt = 1.0;

        let result = trapz_with_fun(waveforms.view(), dt, identity);

        // If the code integrates correctly (Batch, Time), result should have 2 elements.
        // If it integrates incorrectly (Time, Batch), it will have 3 elements.
        assert_eq!(
            result.len(),
            2,
            "Orientation Bug: Output should have 2 elements (one per row), found {}",
            result.len()
        );

        assert_abs_diff_eq!(result[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 4.0, epsilon = 1e-10);
    }
}
