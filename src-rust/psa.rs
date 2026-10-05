use ndarray::prelude::*;

use crate::rotd::{self, Hull, N_ROTD_STATS};

/// Number of coefficients describing one oscillator.
///
/// A row of coefficients is the exact discrete-time update of a unit-mass
/// SDOF oscillator driven by a ground acceleration `ag` that varies linearly
/// between samples (Nigam and Jennings 1969, equivalently a first-order-hold
/// discretisation). The state is the pseudo-acceleration `A = ω² u` and the
/// relative velocity `v`, and the row `[a11, a12, a21, a22, b1, b2, c1, c2]`
/// steps it as
///
/// ```text
/// A_{n+1} = a11 A_n + a12 v_n + b1 ag_n + c1 ag_{n+1}
/// v_{n+1} = a21 A_n + a22 v_n + b2 ag_n + c2 ag_{n+1}
/// ```
///
/// The sign of the inertial load `-ag` is folded into the `b` and `c`
/// coefficients.
pub const N_COEFFICIENTS: usize = 8;

type Coefficients = [f64; N_COEFFICIENTS];

/// Peak pseudo-acceleration of each component's oscillator, starting from
/// rest.
///
/// The 000 and 090 responses are also written to `response_0` and
/// `response_90`, which the RotD sweep needs in full.
fn respond(
    components: [ArrayView1<f64>; 3],
    coefficients: &Coefficients,
    response_0: &mut Array1<f64>,
    response_90: &mut Array1<f64>,
) -> [f64; 3] {
    let [a11, a12, a21, a22, b1, b2, c1, c2] = *coefficients;
    let step = |(a, v): (f64, f64), window: ArrayView1<f64>| {
        (
            a11 * a + a12 * v + b1 * window[0] + c1 * window[1],
            a21 * a + a22 * v + b2 * window[0] + c2 * window[1],
        )
    };
    let [comp_0, comp_90, comp_ver] = components;
    response_0[0] = 0.0;
    response_90[0] = 0.0;
    let mut states = [(0.0, 0.0); 3];
    let mut peaks = [0.0f64; 3];
    // The three components are independent recurrences, so stepping them
    // together lets their latency-bound updates overlap.
    azip!((
        window_0 in comp_0.windows(2),
        window_90 in comp_90.windows(2),
        window_ver in comp_ver.windows(2),
        next_0 in response_0.slice_mut(s![1..]),
        next_90 in response_90.slice_mut(s![1..]),
    ) {
        states = [
            step(states[0], window_0),
            step(states[1], window_90),
            step(states[2], window_ver),
        ];
        *next_0 = states[0].0;
        *next_90 = states[1].0;
        for (peak, state) in peaks.iter_mut().zip(states) {
            *peak = peak.max(state.0.abs());
        }
    });
    peaks
}

/// Columns of a pSA row: the 000, 090, vertical and geometric mean peaks,
/// then the five RotD statistics of [`rotd::rotd_stats`].
pub const N_PSA_COMPONENTS: usize = 4 + N_ROTD_STATS;

/// Pseudo-spectral acceleration statistics for every station and period.
///
/// Row `i` of `coefficients` describes the oscillator for period `i` at the
/// sampling interval of the components, as documented on [`N_COEFFICIENTS`].
/// Output shape: (stations, periods, components = 000, 090, VER, GEOM, rotd0, rotd50, rotd100, theta0, theta100).
pub fn psa(
    comp_0: &ArrayView2<f64>,
    comp_90: &ArrayView2<f64>,
    comp_ver: &ArrayView2<f64>,
    coefficients: &ArrayView2<f64>,
) -> Array3<f64> {
    assert!(
        comp_0.dim() == comp_90.dim() && comp_0.dim() == comp_ver.dim(),
        "components must have matching shapes"
    );
    assert_eq!(
        coefficients.ncols(),
        N_COEFFICIENTS,
        "coefficients must have {N_COEFFICIENTS} columns"
    );
    let (ns, nt) = comp_0.dim();
    let mut out = Array3::zeros((ns, coefficients.nrows(), N_PSA_COMPONENTS));
    if nt == 0 {
        return out;
    }
    // A non-finite input sample poisons the oscillator recursion from that
    // point on, but the peak folds would silently ignore the resulting NaN
    // tail. Short-circuit those stations instead.
    let finite: Vec<bool> = (0..ns)
        .map(|s| {
            [comp_0, comp_90, comp_ver]
                .iter()
                .all(|comp| comp.row(s).iter().all(|v| v.is_finite()))
        })
        .collect();
    let mut hull = Hull::with_capacity(nt);
    let mut response_0 = Array1::zeros(nt);
    let mut response_90 = Array1::zeros(nt);
    for (index, row) in coefficients.rows().into_iter().enumerate() {
        let oscillator: Coefficients = std::array::from_fn(|k| row[k]);
        for (s, &is_finite) in finite.iter().enumerate() {
            let mut out_row = out.slice_mut(s![s, index, ..]);
            if !is_finite {
                out_row.fill(f64::NAN);
                continue;
            }
            let [peak_0, peak_90, peak_ver] = respond(
                [comp_0.row(s), comp_90.row(s), comp_ver.row(s)],
                &oscillator,
                &mut response_0,
                &mut response_90,
            );
            let sweep = hull.peaks(response_0.view(), response_90.view());
            out_row[0] = peak_0;
            out_row[1] = peak_90;
            out_row[2] = peak_ver;
            out_row[3] = (peak_0 * peak_90).sqrt();
            out_row
                .slice_mut(s![4..])
                .assign(&ArrayView1::from(&rotd::rotd_stats(sweep)));
        }
    }
    out
}

#[cfg(test)]
mod tests {

    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;
    const XI: f64 = 0.05;

    /// Coefficients for angular frequency `w` and damping ratio `xi` at
    /// sampling interval `dt`, from a scaled-and-squared Taylor series of the
    /// matrix exponential the Python layer takes with `scipy.linalg.expm`.
    fn coefficients(dt: f64, w: f64, xi: f64) -> Coefficients {
        // State [u, v, p, p'] with p = -ag varying linearly over the step.
        let mut m = Array2::<f64>::zeros((4, 4));
        m[[0, 1]] = dt;
        m[[1, 0]] = -w * w * dt;
        m[[1, 1]] = -2.0 * xi * w * dt;
        m[[1, 2]] = dt;
        m[[2, 3]] = 1.0;
        let norm = m
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
            .fold(0.0, f64::max);
        let squarings = norm.log2().ceil().max(0.0) as i32 + 1;
        let scaled = m * 0.5f64.powi(squarings);
        let mut term = Array2::<f64>::eye(4);
        let mut e = term.clone();
        for k in 1..=20 {
            term = term.dot(&scaled) / k as f64;
            e += &term;
        }
        for _ in 0..squarings {
            e = e.dot(&e);
        }
        let w2 = w * w;
        [
            e[[0, 0]],
            e[[0, 1]] * w2,
            e[[1, 0]] / w2,
            e[[1, 1]],
            -(e[[0, 2]] - e[[0, 3]]) * w2,
            -(e[[1, 2]] - e[[1, 3]]),
            -e[[0, 3]] * w2,
            -e[[1, 3]],
        ]
    }

    /// Pseudo-acceleration response to a single component.
    fn response(waveform: &Array1<f64>, coefficients: &Coefficients) -> Array1<f64> {
        let mut response = Array1::zeros(waveform.len());
        let mut unused = Array1::zeros(waveform.len());
        let view = waveform.view();
        respond([view, view, view], coefficients, &mut response, &mut unused);
        response
    }

    #[test]
    fn test_zeros() {
        let waveform = Array1::<f64>::zeros(100);
        let a = response(&waveform, &coefficients(0.01, 1.0, XI));
        assert_eq!(a, Array1::<f64>::zeros(100));
    }

    #[test]
    fn test_solves_constant() {
        let dt = 0.001;
        let waveform = Array1::<f64>::ones(100_000);
        let a = response(&waveform, &coefficients(dt, 2.0 * PI, XI));
        // The long-run steady state, u = -1 / w^2, also checks the recursion
        // does not accumulate floating point error.
        assert_abs_diff_eq!(a[waveform.len() - 1], -1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_is_exact_for_a_ramp() {
        // A ramp is piecewise linear, so the recurrence should reproduce the
        // analytical response to rounding error even at a coarse step.
        let dt = 0.02;
        let w = 2.0 * PI / 0.3;
        let t = Array1::from_shape_fn(2000, |i| i as f64 * dt);
        for xi in [0.0, XI, 0.7] {
            let a = response(&t, &coefficients(dt, w, xi));

            // u'' + 2 xi w u' + w^2 u = -t from rest.
            let wd = w * (1.0 - xi * xi).sqrt();
            let c = -2.0 * xi / w.powi(3);
            let d = (1.0 / (w * w) + xi * w * c) / wd;
            let u = t.map(|&x| {
                -(x / (w * w) - 2.0 * xi / w.powi(3))
                    + (-xi * w * x).exp() * (c * (wd * x).cos() + d * (wd * x).sin())
            });
            assert_abs_diff_eq!(a, u * w * w, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_solves_critically_damped_harmonic_oscillation() {
        let t = Array1::<f64>::linspace(0.0, 10.0, 10000);
        let dt = t[1] - t[0];
        let waveform = t.sin();

        let a = response(&waveform, &coefficients(dt, 1.0, 1.0));
        // A sine is not piecewise linear, so expect an O(dt^2) error.
        let analytical = t.map(|&x| -0.5 * (-x).exp() * (x - x.exp() * x.cos() + 1.0));
        assert_abs_diff_eq!(a, analytical, epsilon = 1e-7);
    }

    fn coefficient_table(periods: &[f64], dt: f64) -> Array2<f64> {
        Array2::from(
            periods
                .iter()
                .map(|&period| coefficients(dt, 2.0 * PI / period, XI))
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn test_psa_matches_the_shared_rotd_path() {
        // pSA is the RotD reduction applied to oscillator responses rather
        // than to the waveforms themselves, so it must agree exactly with the
        // peak ground motion path run on those responses.
        let t = Array1::<f64>::linspace(0.0, 2.0, 512);
        let dt = t[1] - t[0];
        let comp_0 = t.map(|&x| (3.0 * x).sin());
        let comp_90 = t.map(|&x| 0.7 * (5.0 * x).cos());
        let comp_ver = t.map(|&x| 0.2 * (7.0 * x).sin());

        let result = psa(
            &comp_0.view().insert_axis(Axis(0)),
            &comp_90.view().insert_axis(Axis(0)),
            &comp_ver.view().insert_axis(Axis(0)),
            &coefficient_table(&[1.0], dt).view(),
        );

        let oscillator = coefficients(dt, 2.0 * PI, XI);
        let responses: Vec<Array1<f64>> = [&comp_0, &comp_90, &comp_ver]
            .iter()
            .map(|comp| response(comp, &oscillator))
            .collect();
        let peak = |response: &Array1<f64>| response.iter().fold(0.0f64, |m, &a| m.max(a.abs()));

        for (column, response) in responses.iter().enumerate() {
            assert_abs_diff_eq!(result[[0, 0, column]], peak(response), epsilon = 1e-12);
        }
        assert_abs_diff_eq!(
            result[[0, 0, 3]],
            (peak(&responses[0]) * peak(&responses[1])).sqrt(),
            epsilon = 1e-12
        );
        // The statistics of the responses, straight off the peak ground
        // motion entry point.
        let stats = crate::rotd::rotd(
            responses[0].view().insert_axis(Axis(0)),
            responses[1].view().insert_axis(Axis(0)),
        );
        for column in 0..N_ROTD_STATS {
            assert_abs_diff_eq!(
                result[[0, 0, 4 + column]],
                stats[[0, column]],
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn test_psa_is_nan_when_a_sample_is_non_finite() {
        // A NaN sample poisons the oscillator recursion from that point on,
        // but the peak fold used to ignore the resulting NaN tail and return
        // the (finite, wrong) peak of the record up to that point instead.
        let t = Array1::<f64>::linspace(0.0, 2.0, 512);
        let dt = t[1] - t[0];
        let mut comp_0 = t.map(|&x| (3.0 * x).sin());
        comp_0[100] = f64::NAN;
        let comp_0 = comp_0.insert_axis(Axis(0));
        let comp_90 = t.map(|&x| 0.7 * (5.0 * x).cos()).insert_axis(Axis(0));
        let comp_ver = t.map(|&x| 0.2 * (7.0 * x).sin()).insert_axis(Axis(0));

        let result = psa(
            &comp_0.view(),
            &comp_90.view(),
            &comp_ver.view(),
            &coefficient_table(&[1.0], dt).view(),
        );

        for column in 0..N_PSA_COMPONENTS {
            assert!(
                result[[0, 0, column]].is_nan(),
                "column {column} should be NaN, found {}",
                result[[0, 0, column]]
            );
        }
    }
}
