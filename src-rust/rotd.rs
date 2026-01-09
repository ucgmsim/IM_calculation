use std::f64::consts::PI;

use ndarray::prelude::*;
use ndarray::Zip;

const DEGREES: f64 = PI / 180.0;

/// RotD180 calculations for a single pair of components assuming the absmax reduction function.
fn rotd_calculation(comp_0: ArrayView1<f64>, comp_90: ArrayView1<f64>) -> (f64, f64, f64) {
    let mut rotd_values: [f64; 180] = [0.0; 180];

    for (theta, result) in rotd_values.iter_mut().enumerate() {
        let theta_f: f64 = theta as f64;
        let theta_rad = theta_f * DEGREES;
        let sin_theta = theta_rad.sin();
        let cos_theta = theta_rad.cos();
        *result = Zip::from(comp_0).and(comp_90).fold(0.0, |acc, &x, &y| {
            acc.max((cos_theta * x + sin_theta * y).abs())
        });
    }
    let (small, middle, large) = rotd_values.select_nth_unstable_by(90, f64::total_cmp);
    let min = small.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let lower_middle = small.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let max = large.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let median = (lower_middle + *middle) / 2.0;
    (*min, median, *max)
}

pub fn rotd_parallel(comp_0: ArrayView2<f64>, comp_90: ArrayView2<f64>) -> Array<f64, Ix2> {
    let (ns, _) = comp_0.dim();
    let (ns2, _) = comp_90.dim();
    assert_eq!(ns, ns2);
    let mut out = Array::<f64, Ix2>::zeros((ns, 3));
    Zip::from(out.rows_mut())
        .and(comp_0.rows())
        .and(comp_90.rows())
        .par_for_each(|mut stats, comp_0, comp_90| {
            let (min, median, max) = rotd_calculation(comp_0, comp_90);
            stats.assign(&array![min, median, max]);
        });
    out
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::rotd::rotd_calculation;
    #[test]
    fn test_rotd_calculation() {
        let comp_0 = array![1.0f64, 0.0f64];
        let comp_90 = array![0.0f64, 1.0f64];
        let (min, median, max) = rotd_calculation(comp_0.view(), comp_90.view());
        let expected_min = 2.0.sqrt() / 2.0; // e.g. at pi / 4 degrees
        let expected_max = 1.0; // e.g. at 0 degrees
        let expected_median = 0.9238443540096138; // at 23 degrees, derived independently with numpy
        assert!(
            (min - expected_min).abs() < 1e-6,
            "Minimum calculation failed: expected sqrt(2) +/- 1e-6 found: {}",
            min
        );
        assert!(
            (median - expected_median).abs() < 1e-6,
            "Median calculation failed: expected ~0.9238444 +/- 1e-6 found: {}",
            median
        );
        assert!(
            (max - expected_max).abs() < 1e-6,
            "Maximum calculation failed: expected 1.0 +/- 1e-6 found: {}",
            max
        );
    }
}
