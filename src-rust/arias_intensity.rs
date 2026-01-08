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
