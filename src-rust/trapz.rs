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


}
