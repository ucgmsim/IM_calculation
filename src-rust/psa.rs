use ndarray::prelude::*;
use ndarray::{Array1, Ix1, Ix2};

#[allow(clippy::too_many_arguments)]
fn newmark_beta_solver(
    waveform: ArrayView<f64, Ix1>,
    dt: f64,
    w: f64,
    xi: f64,
    gamma: f64,
    beta: f64,
    u0: f64,
    dudt0: f64,
) -> Array1<f64> {
    let nt = waveform.dim();
    let mut u = Array1::zeros(nt);
    let one_over_beta_dt_sq = 1.0 / (beta * dt * dt);
    let one_over_beta_dt = 1.0 / (beta * dt);
    let k = w * w;
    let c = 2.0 * xi * w;
    let c_gamma_over_beta_dt = (gamma * c) / (beta * dt);
    let one_over_two_beta = 1.0 / (2.0 * beta);
    // Constants to solve for u_n+1:
    let kbar = one_over_beta_dt_sq + k + c_gamma_over_beta_dt; // u_n+1
    let a1 = c_gamma_over_beta_dt + one_over_beta_dt_sq; // u_n
    let b1 = one_over_beta_dt + c * (gamma / beta - 1.0); // udot_n
    let c1 = c * dt * (gamma / (2.0 * beta) - 1.0) + one_over_two_beta - 1.0; // uddot_n

    // Constants to solve for uddot_n+1
    let a2 = one_over_beta_dt_sq; // u_n+1 - u_n
    let b2 = -one_over_beta_dt; // udot_n
    let c2 = -c1; // uddot_n
    // Constants to solve for udot_n+1
    // a'3 = 1 for uddot_n
    let a3 = 1.0 - gamma; // uddot_n
    let b3 = gamma; // uddot_n+1

    u[0] = u0;
    let mut udot = dudt0;
    let mut uddot = -waveform[0] - (c * udot + k * u[0]); // negated because ground motion

    // ENCI335 notes solve for u''_n+1, but numerical stability argues for solving for displacement directly and then deriving velocity and acceleration from that.
    // Basically the formulations in the ENCI notes solve for acceleration and then integrate to get displacement, but this involves the calculation of
    // d_pti = waveform[i + 1] - waveform[i]
    // a floating-point operation that introduces noise. Integrating that noise twice amplifies it into the rest of the calculations.
    // Instead: implicitly solve for displacement and differentiate. The u at each time step is the smoothed response, so it's less sensitive to signal noise from the waveform.
    // See https://collab.dvb.bayern/spaces/TUMmodsim/pages/71122788/Newmark-%CE%B2+method for the derivation when beta = 1/4, gamma = 1/2.
    for i in 0..(nt - 1) {
        let u_n = u[i];
        let f_next = -waveform[i + 1]; // Negated because ground motion.
        let pbar = f_next + a1 * u_n + b1 * udot + c1 * uddot;
        let u_next = pbar / kbar;
        let uddot_next = a2 * (u_next - u_n) + b2 * udot + c2 * uddot;
        let udot_next = udot + dt * (a3 * uddot + b3 * uddot_next);
        u[i + 1] = u_next;
        udot = udot_next;
        uddot = uddot_next;
    }
    u
}

fn choose_gamma_beta(dt: f64, w: f64) -> (f64, f64) {
    let gamma = 0.5;

    let stability_constant = 0.551328895421792;
    // Whilst the linear solver is theoretically stable for ratios
    // dt/T up to 0.551-ish, the switch from the constant solver to the
    // linear one uses a wider margin than that. Testing settled on 80%.
    // The cost is a little accuracy, and it falls only on short period
    // pSA with large timesteps.
    let stability_fraction = 0.8;
    let effective_stability_constant = stability_fraction * stability_constant;
    let pi = std::f64::consts::PI;
    let beta = if dt < effective_stability_constant * (2.0 * pi) / w {
        1.0 / 6.0
    } else {
        1.0 / 4.0
    };

    (gamma, beta)
}

pub fn newmark_beta_method(
    waveform: ArrayView<f64, Ix1>,
    dt: f64,
    w: f64,
    xi: f64,
    u0: f64,
    dudt0: f64,
) -> Array1<f64> {
    let (gamma, beta) = choose_gamma_beta(dt, w);
    newmark_beta_solver(waveform, dt, w, xi, gamma, beta, u0, dudt0)
}

/// Solve the SDOF oscillator equation for every row of `waveforms`, serially.
///
/// The `waveforms` array has shape `(ns, nt)`, where `ns` is the number of
/// stations and `nt` the number of timesteps. This solves each row with the
/// Newmark-Beta method for an oscillator of angular frequency `w` and damping
/// coefficient `xi`, and writes the resulting displacement response to the
/// matching row of the `(ns, nt)` output.
pub fn newmark_beta_method_batch(
    waveforms: &ArrayView2<f64>,
    dt: f64,
    w: f64,
    xi: f64,
) -> Array<f64, Ix2> {
    let mut out = Array::<f64, Ix2>::zeros(waveforms.dim());
    out.axis_iter_mut(Axis(0))
        .zip(waveforms.axis_iter(Axis(0)))
        .for_each(|(mut out_row, in_row)| {
            out_row.assign(&newmark_beta_method(in_row, dt, w, xi, 0.0, 0.0));
        });
    out
}

/// Pseudo-spectral acceleration at every integer rotation angle 0..=179
/// degrees, for one oscillator period, computed **serially**.
///
/// For each station this runs the two horizontal components through the
/// Newmark-beta SDOF solver (f64 throughout, for accuracy over long records),
/// then [`crate::rotd::Hull::peaks`] reduces the displacement responses to
/// their peak rotated amplitude at every angle. Multiplying by `w^2`
/// converts the peak relative displacement of the unit-mass oscillator to a
/// pseudo-spectral acceleration.
///
/// The loop over stations is deliberately serial: this runs one period per
/// call inside a Dask worker that already has a core to itself, so spawning a
/// Rayon pool here would oversubscribe the machine and fight the outer
/// scheduler. The RotD hull allocates its work buffers once and reuses them
/// for every station.
///
/// `comp_0` and `comp_90` are the 000 and 090 acceleration waveforms with
/// shape `(ns, nt)`. The result has shape `(ns, 182)`: columns 0..=179 are
/// the rotated peaks at each integer angle, and columns 180 and 181 are the
/// exact peaks of the unrotated 000 and 090 responses (`w^2 * max|response|`),
/// so a caller who only needs those two components doesn't have to re-derive
/// them from angle 0 / angle 90, which are off by `cos(90 deg) ~= 6.12e-17`
/// rather than being exactly zero.
pub fn psa_rotd180(
    comp_0: &ArrayView2<f64>,
    comp_90: &ArrayView2<f64>,
    dt: f64,
    w: f64,
    xi: f64,
) -> Array2<f64> {
    assert_eq!(
        comp_0.dim(),
        comp_90.dim(),
        "components must have matching shapes"
    );
    let ns = comp_0.nrows();
    let conversion_factor = w * w;
    let mut out = Array2::<f64>::zeros((ns, 182));
    let mut hull = crate::rotd::Hull::with_capacity(comp_0.ncols());
    for s in 0..ns {
        let response_0 = newmark_beta_method(comp_0.row(s), dt, w, xi, 0.0, 0.0);
        let response_90 = newmark_beta_method(comp_90.row(s), dt, w, xi, 0.0, 0.0);
        let peaks = hull.peaks(response_0.view(), response_90.view());
        let mut row = out.row_mut(s);
        for (angle, &peak) in peaks.iter().enumerate() {
            row[angle] = conversion_factor * peak;
        }
        row[180] = conversion_factor * response_0.iter().fold(0.0f64, |m, &u| m.max(u.abs()));
        row[181] = conversion_factor * response_90.iter().fold(0.0f64, |m, &u| m.max(u.abs()));
    }
    out
}

/// Pseudo-spectral acceleration peak for one component, one period.
///
/// `waveforms` has shape `(ns, nt)`. This returns only the peak response
/// (shape `(ns,)`), so a caller that needs just one component -- the
/// vertical, say, which never participates in RotD -- doesn't have to move a
/// full `(ns, nt)` displacement response back into Python.
pub fn psa_peak(waveforms: &ArrayView2<f64>, dt: f64, w: f64, xi: f64) -> Array1<f64> {
    let conversion_factor = w * w;
    Array1::from_shape_fn(waveforms.nrows(), |s| {
        let response = newmark_beta_method(waveforms.row(s), dt, w, xi, 0.0, 0.0);
        conversion_factor * response.iter().fold(0.0f64, |m, &u| m.max(u.abs()))
    })
}

#[cfg(test)]
mod tests {

    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;
    const XI: f64 = 0.05;
    // Linear gamma and beta settings
    const GAMMA: f64 = 0.5;
    const BETA: f64 = 1.0 / 6.0;
    const M: f64 = 1.0;

    #[test]
    fn test_newmark_linear_parameters() {
        let dt = 0.01f64;
        let w = 2.0 * PI; // T_n = 1
        let (gamma, beta) = choose_gamma_beta(dt, w);
        assert_eq!((gamma, beta), (0.5, 1.0 / 6.0));
    }

    #[test]
    fn test_newmark_const_parameters() {
        let dt = 0.01f64;
        let w = 100.0 * 2.0 * PI; // T_n = 0.01
        let (gamma, beta) = choose_gamma_beta(dt, w);
        assert_eq!((gamma, beta), (0.5, 0.25));
    }

    fn differentiate(waveform: &Array<f64, Ix1>, dt: f64) -> Array<f64, Ix1> {
        let nt = waveform.dim();
        Array1::from_shape_fn(nt, |i| {
            if i == 0 {
                (waveform[i + 1] - waveform[i]) / dt
            } else if i == nt - 1 {
                (waveform[i] - waveform[i - 1]) / dt
            } else {
                (waveform[i + 1] - waveform[i - 1]) / (2.0 * dt)
            }
        })
    }

    fn ddifferentiate(waveform: &Array<f64, Ix1>, dt: f64) -> Array<f64, Ix1> {
        let nt = waveform.dim();
        Array1::from_shape_fn(nt, |i| {
            if i == 0 {
                (waveform[i + 2] - 2.0 * waveform[i + 1] + waveform[i]) / (dt * dt)
            } else if i == nt - 1 {
                (waveform[i] - 2.0 * waveform[i - 1] + waveform[i - 2]) / (dt * dt)
            } else {
                (waveform[i + 1] - 2.0 * waveform[i] + waveform[i - 1]) / (dt * dt)
            }
        })
    }

    #[test]
    fn test_newmark_beta_single_zeros() {
        let waveform = Array1::<f64>::zeros(100);
        let dt = 0.01;
        let w = 1.0;

        let u = newmark_beta_solver(waveform.view(), dt, w, XI, GAMMA, BETA, 0.0, 0.0);
        let expected = Array1::<f64>::zeros(100);
        assert_eq!(u, expected);
    }

    #[test]
    fn test_newmark_beta_solves_constant() {
        let dt = 0.001;
        let waveform = Array1::<f64>::ones(100_000);
        let w = 2.0 * PI;

        let u = newmark_beta_solver(waveform.view(), dt, w, XI, GAMMA, BETA, 0.0, 0.0);

        // The exact solution would serve here too. This test measures the
        // long-term behaviour of the solver accumulating floating point error.
        let uss = -M / (M * w * w);
        let uss_est = u[waveform.dim() - 1];
        let err = (uss_est - uss).abs();
        assert!(
            err < 1e-4,
            "NB method did not solve for steady-state solution (uss = {}): |{} - {}| = {} > 1e-4",
            uss,
            uss,
            uss_est,
            err
        );
    }

    #[test]
    fn test_newmark_beta_solves_equation() {
        // Test a more complicated sum of frequencies
        let t = Array1::<f64>::linspace(0.0, 10.0, 100_000);
        let dt = t[1] - t[0];
        // W(t) = sum_n 1/n * sin(pi * n * t)
        let waveform = t.map(|&t| {
            (1..100)
                .map(|freq| {
                    let freq_f = freq as f64;
                    1.0 / freq_f * (freq_f * PI * t).sin()
                })
                .sum()
        });
        let w = 2.0 * PI;

        let u = newmark_beta_solver(waveform.view(), dt, w, XI, GAMMA, BETA, 0.0, 0.0);

        let du = differentiate(&u, dt);
        let du2 = ddifferentiate(&u, dt);
        let c = 2.0 * XI * w;
        let k = M * w * w;
        let sdof_invariant = M * du2 + c * du + k * u + M * waveform;
        // Skipping the first and last value of u because the differentiation is less accurate at the boundary.

        let max_deviation = sdof_invariant
            .slice(s![1..sdof_invariant.len() - 1])
            .map(|&x| x.abs())
            .fold(0.0, |x: f64, &y| x.max(y));
        assert_abs_diff_eq!(max_deviation, 0.0, epsilon = 5e-4);
    }

    #[test]
    fn test_newmark_solves_undamped_free_vibration() {
        let t = Array1::<f64>::linspace(0.0, 10.0, 10000);
        let dt = t[1] - t[0];
        let waveform = Array1::zeros(t.dim());
        let w = 2.0 * PI;

        let u = newmark_beta_solver(waveform.view(), dt, w, 0.0, GAMMA, BETA, 1.0, 0.0);
        let analytical = (t * w).cos();
        assert_abs_diff_eq!(u, analytical, epsilon = 5e-4);
    }

    #[test]
    fn test_newmark_solves_damped_free_vibration() {
        let t = Array1::<f64>::linspace(0.0, 10.0, 10000);
        let dt = t[1] - t[0];
        let waveform = Array1::zeros(t.dim());
        let w = 1.0;
        let xi = 1.0;
        let u = newmark_beta_solver(waveform.view(), dt, w, xi, GAMMA, BETA, 1.0, 0.0);
        let analytical = t.map(|&x| (-x).exp() * (x + 1.0));
        assert_abs_diff_eq!(u, analytical, epsilon = 5e-4);
    }

    #[test]
    fn test_psa_rotd180_columns_180_181_match_component_peaks() {
        // Columns 180/181 must agree with an independent per-component
        // psa_peak computation (which is also what column 0 / column 90
        // approximate, up to cos(90 deg) != 0 exactly in f64).
        let t = Array1::<f64>::linspace(0.0, 2.0, 512);
        let dt = t[1] - t[0];
        let comp_0 = t.map(|&x| (3.0 * x).sin());
        let comp_90 = t.map(|&x| 0.7 * (5.0 * x).cos());
        let comp_0_2d = comp_0.clone().insert_axis(Axis(0));
        let comp_90_2d = comp_90.clone().insert_axis(Axis(0));
        let w = 2.0 * PI;

        let combined = psa_rotd180(&comp_0_2d.view(), &comp_90_2d.view(), dt, w, XI);
        let peak_0 = psa_peak(&comp_0_2d.view(), dt, w, XI);
        let peak_90 = psa_peak(&comp_90_2d.view(), dt, w, XI);

        assert_abs_diff_eq!(combined[[0, 180]], peak_0[0], epsilon = 1e-12);
        assert_abs_diff_eq!(combined[[0, 181]], peak_90[0], epsilon = 1e-12);
    }

    #[test]
    fn test_newmark_solves_damped_harmonic_oscillation() {
        let t = Array1::<f64>::linspace(0.0, 10.0, 10000);
        let dt = t[1] - t[0];
        let waveform = t.sin();
        let w = 1.0;
        let xi = 1.0;
        let u = newmark_beta_solver(waveform.view(), dt, w, xi, GAMMA, BETA, 0.0, 0.0);

        let analytical = t.map(|&x| -0.5 * (-x).exp() * (x - x.exp() * x.cos() + 1.0));
        assert_abs_diff_eq!(u, analytical, epsilon = 5e-4);
    }
}
