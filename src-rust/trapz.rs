use crate::utils::{parallel_map_rows, parallel_reduce_rows};
use ndarray::prelude::*;

fn trapz_step<F>(v1: f64, v2: f64, dt: f64, f: &F) -> f64
where
    F: Fn(f64) -> f64,
{
    if v1.min(v2) >= 0.0 || v1.max(v2) <= 0.0 {
        dt * (f(v1) + f(v2))
    } else {
        let inv_slope = dt / (v2 - v1);
        let x0 = -v1 * inv_slope;
        x0 * f(v1) + (dt - x0) * f(v2)
    }
}

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

pub fn parallel_trapz_with_fun<F>(waveforms: ArrayView2<f64>, dt: f64, f: F) -> Array1<f64>
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    parallel_reduce_rows(waveforms, |waveform| trapz_one_with_fun(waveform, dt, &f))
}

pub fn trapz_with_fun<F>(waveforms: ArrayView2<f64>, dt: f64, f: F) -> Array1<f64>
where
    F: Fn(f64) -> f64,
{
    waveforms.map_axis(Axis(1), |waveform| trapz_one_with_fun(waveform, dt, &f))
}

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

pub fn parallel_trapz(waveforms: ArrayView2<f64>, dt: f64) -> Array1<f64> {
    parallel_reduce_rows(waveforms, |waveform| trapz_one(waveform, dt))
}

pub fn trapz(waveforms: ArrayView2<f64>, dt: f64) -> Array1<f64> {
    waveforms.map_axis(Axis(0), |waveform| trapz_one(waveform, dt))
}
