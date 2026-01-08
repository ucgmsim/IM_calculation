//! Cumulative absolute velocity (CAV) calculation module.
//!
//! This module provides functions to calculate the intensity of earthquake ground motion
//! based on the integral of the absolute value of accelaration.
//!
//! The formula used is:
//! $$CAV = \int_{0}^{T} |a(t)| \, dt$$
//!

use crate::constants::G;
use crate::trapz::{parallel_trapz_with_fun, trapz_with_fun};

use ndarray::prelude::*;

/// Computes the total Cumulative Absolute Velocity ($I_A$) for each row in parallel.
///
/// # Arguments
/// * `waveforms` - A 2D array view where each row is an acceleration time-series.
/// * `dt` - The time step (sampling interval) of the waveforms.
///
/// # Returns
/// An `Array1<f64>` containing the final $I_A$ value for each station.
pub fn parallel_cav(waveforms: ArrayView2<f64>, dt: f64) -> Array1<f64> {
    G * parallel_trapz_with_fun(waveforms, dt, |x| x.abs())
}

/// Computes the total Cumulative Absolute Velocity ($I_A$) for each row using a single thread.
///
/// # Arguments
/// * `waveforms` - A 2D array view where each row is an acceleration time-series.
/// * `dt` - The time step (sampling interval) of the waveforms.
pub fn cav(waveforms: ArrayView2<f64>, dt: f64) -> Array1<f64> {
    G * trapz_with_fun(waveforms, dt, |x| x.abs())
}
