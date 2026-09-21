use ndarray::prelude::*;
use ndarray::{Array1, Ix1};

use crate::rotd::{self, Hull, N_ROTD_STATS};

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

    // ENCI335 notes solve for u''_n+1, but for numerical stability reasons we really want to solve for displacement directly and then derive velocity and acceleration from that.
    // Basically the formulations in the ENCI notes solve for acceleration and then integrate to get displacement, but this involves the calculation of
    // d_pti = waveform[i + 1] - waveform[i]
    // which is a noisy floating-point operation. We then numerically integrate that noise twice which amplifies the noise carried into the rest of the calculations.
    // Instead: implicitly solve for displacement and differentiate. The u at each time step is the smoothed response, so it is more robust to signal noise from the waveform.
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
    // dt/T up to 0.551-ish, we want to be a bit more conservative
    // about when we choose the linear solver instead of the constant
    // solver. During testing, we pick a conservative 80%. This means we leave
    // some result accuracy on the table, but I can live with this
    // because it only affects very short period pSA with large
    // timesteps.
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

/// Columns of a pSA row: the 000, 090, vertical and geometric mean peaks,
/// then the five RotD statistics of [`rotd::rotd_stats`].
pub const N_PSA_COMPONENTS: usize = 4 + N_ROTD_STATS;

/// Peak pseudo-spectral acceleration of a displacement response.
///
/// Multiplying by `w_squared` converts the peak relative displacement of the
/// unit-mass oscillator to a pseudo-spectral acceleration.
fn peak(response: &Array1<f64>, w_squared: f64) -> f64 {
    w_squared * response.iter().fold(0.0f64, |m, &u| m.max(u.abs()))
}

/// Pseudo-spectral acceleration statistics for every station and period.
/// Output shape: (stations, periods, components = 000, 090, VER, GEOM, rotd0, rotd50, rotd100, theta0, theta100).
pub fn psa(
    comp_0: &ArrayView2<f64>,
    comp_90: &ArrayView2<f64>,
    comp_ver: &ArrayView2<f64>,
    periods: &ArrayView1<f64>,
    dt: f64,
    xi: f64,
) -> Array3<f64> {
    assert!(
        comp_0.dim() == comp_90.dim() && comp_0.dim() == comp_ver.dim(),
        "components must have matching shapes"
    );
    let ns = comp_0.nrows();
    let mut out = Array3::zeros((ns, periods.len(), N_PSA_COMPONENTS));
    let mut hull = Hull::with_capacity(comp_0.ncols());
    for (index, &period) in periods.iter().enumerate() {
        let w = std::f64::consts::TAU / period;
        let w_squared = w * w;
        for s in 0..ns {
            let response_0 = newmark_beta_method(comp_0.row(s), dt, w, xi, 0.0, 0.0);
            let response_90 = newmark_beta_method(comp_90.row(s), dt, w, xi, 0.0, 0.0);
            let response_ver = newmark_beta_method(comp_ver.row(s), dt, w, xi, 0.0, 0.0);
            let mut sweep = hull.peaks(response_0.view(), response_90.view());
            sweep.iter_mut().for_each(|peak| *peak *= w_squared);
            let peak_0 = peak(&response_0, w_squared);
            let peak_90 = peak(&response_90, w_squared);
            let mut row = out.slice_mut(s![s, index, ..]);
            row[0] = peak_0;
            row[1] = peak_90;
            row[2] = peak(&response_ver, w_squared);
            row[3] = (peak_0 * peak_90).sqrt();
            row.slice_mut(s![4..])
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

        // Of course we could use the exact solution here. This test is to determine
        // the long-term behaviour of the solver accumulating floating point error.
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
        // W(t) = sum_i 1/i * sin(pi * i * t)
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
    fn test_psa_matches_the_shared_rotd_path() {
        // pSA is the RotD reduction applied to oscillator responses rather
        // than to the waveforms themselves, so it must agree exactly with the
        // peak ground motion path run on those responses.
        let t = Array1::<f64>::linspace(0.0, 2.0, 512);
        let dt = t[1] - t[0];
        let comp_0 = t.map(|&x| (3.0 * x).sin()).insert_axis(Axis(0));
        let comp_90 = t.map(|&x| 0.7 * (5.0 * x).cos()).insert_axis(Axis(0));
        let comp_ver = t.map(|&x| 0.2 * (7.0 * x).sin()).insert_axis(Axis(0));
        let period = 1.0;
        let w_squared = (std::f64::consts::TAU / period).powi(2);

        let result = psa(
            &comp_0.view(),
            &comp_90.view(),
            &comp_ver.view(),
            &array![period].view(),
            dt,
            XI,
        );

        let responses: Vec<Array2<f64>> = [&comp_0, &comp_90, &comp_ver]
            .iter()
            .map(|comp| {
                newmark_beta_method(comp.row(0), dt, w_squared.sqrt(), XI, 0.0, 0.0)
                    .insert_axis(Axis(0))
            })
            .collect();
        let peak = |response: &Array2<f64>| {
            w_squared * response.iter().fold(0.0f64, |m, &u| m.max(u.abs()))
        };

        for (column, response) in responses.iter().enumerate() {
            assert_abs_diff_eq!(result[[0, 0, column]], peak(response), epsilon = 1e-12);
        }
        assert_abs_diff_eq!(
            result[[0, 0, 3]],
            (peak(&responses[0]) * peak(&responses[1])).sqrt(),
            epsilon = 1e-12
        );
        // The statistics of the scaled responses, straight off the peak
        // ground motion entry point.
        let stats = crate::rotd::rotd(
            (&responses[0] * w_squared).view(),
            (&responses[1] * w_squared).view(),
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
