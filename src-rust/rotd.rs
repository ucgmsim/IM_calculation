use std::f64::consts::PI;

use ndarray::prelude::*;
use ndarray::Zip;

const DEGREES: f64 = PI / 180.0;

/// RotD180 calculations for a single pair of components assuming the absmax
/// reduction function.
///
/// Returns the (min, median, max) rotated peak amplitude, i.e. RotD00, RotD50
/// and RotD100.
fn rotd_calculation(comp_0: ArrayView1<f64>, comp_90: ArrayView1<f64>) -> [f64; 3] {
    let mut rotd_values: [f64; 180] = std::array::from_fn(|theta| {
        let (sin_theta, cos_theta) = (theta as f64 * DEGREES).sin_cos();

        Zip::from(comp_0).and(comp_90).fold(0.0f64, |peak, &x, &y| {
            peak.max((cos_theta * x + sin_theta * y).abs())
        })
    });
    rotd_values.sort_unstable_by(f64::total_cmp);
    [
        rotd_values[0],
        (rotd_values[89] + rotd_values[90]) / 2.0,
        rotd_values[179],
    ]
}

/// Fill an (ns, 3) array with the RotD statistics of each waveform pair,
/// optionally distributing the rows across rayon's thread pool.
fn rotd_rows(comp_0: ArrayView2<f64>, comp_90: ArrayView2<f64>, parallel: bool) -> Array2<f64> {
    assert_eq!(
        comp_0.nrows(),
        comp_90.nrows(),
        "Components must have the same number of waveforms"
    );
    let mut out = Array2::zeros((comp_0.nrows(), 3));

    let zip = Zip::from(out.rows_mut())
        .and(comp_0.rows())
        .and(comp_90.rows());
    let stats = |mut out: ArrayViewMut1<f64>, comp_0, comp_90| {
        out.assign(&ArrayView1::from(&rotd_calculation(comp_0, comp_90)));
    };
    if parallel {
        zip.par_for_each(stats);
    } else {
        zip.for_each(stats);
    }
    out
}

pub fn rotd_parallel(comp_0: ArrayView2<f64>, comp_90: ArrayView2<f64>) -> Array2<f64> {
    rotd_rows(comp_0, comp_90, true)
}

pub fn rotd(comp_0: ArrayView2<f64>, comp_90: ArrayView2<f64>) -> Array2<f64> {
    rotd_rows(comp_0, comp_90, false)
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{SQRT_2, TAU};

    use ndarray::prelude::*;
    use proptest::prelude::*;

    use crate::rotd::{rotd, rotd_calculation, rotd_parallel};

    /// Slack allowed on the sqrt(2) bound. The bound is attained exactly by
    /// linearly polarised records, so only floating point error is tolerated.
    const RATIO_TOL: f64 = 1e-12;

    /// Longest generated waveform. RotD is O(180 * nt), so this keeps a full
    /// proptest run to well under a second.
    const MAX_NT: usize = 128;

    /// Assert RotD100 <= sqrt(2) * RotD50 for one pair of components, returning
    /// the ratio.
    ///
    /// The bound holds because the rotated traces at angles theta and
    /// theta + 90 degrees, sampled at the time of the RotD100 peak, have squared
    /// amplitudes summing to RotD100^2 at least. So at least one angle of every
    /// such pair -- 90 of the 180 angles -- peaks at or above RotD100 / sqrt(2),
    /// which puts the median there too.
    fn assert_ratio_bounded(comp_0: ArrayView1<f64>, comp_90: ArrayView1<f64>, case: &str) -> f64 {
        let [rotd00, rotd50, rotd100] = rotd_calculation(comp_0, comp_90);
        assert!(
            rotd00 <= rotd50 && rotd50 <= rotd100,
            "{case}: RotD00 <= RotD50 <= RotD100 violated: {rotd00}, {rotd50}, {rotd100}"
        );
        if rotd100 == 0.0 {
            // A record that never moves has no ratio to speak of.
            assert_eq!(rotd50, 0.0, "{case}: RotD50 non-zero for a zero record");
            return 1.0;
        }
        assert!(
            rotd50 > 0.0,
            "{case}: RotD50 is zero for a non-zero record (RotD100 = {rotd100})"
        );
        let ratio = rotd100 / rotd50;
        assert!(
            ratio <= SQRT_2 + RATIO_TOL,
            "{case}: RotD100 / RotD50 = {ratio} exceeds sqrt(2) = {SQRT_2}"
        );
        ratio
    }

    /// Linearly polarised record: `waveform` projected onto the direction
    /// `angle` (radians), for which the sqrt(2) bound is tight.
    fn polarised(waveform: &Array1<f64>, angle: f64) -> (Array1<f64>, Array1<f64>) {
        let (sin_angle, cos_angle) = angle.sin_cos();
        (waveform * cos_angle, waveform * sin_angle)
    }

    /// A non-zero waveform, sampled in [-1, 1). RotD ratios are scale
    /// invariant, so amplitude is not worth exploring here -- the fixed tests
    /// cover the extremes of the floating point range instead. The all-zero
    /// record is excluded because it has no polarisation direction; it is
    /// covered by `test_ratio_bound_degenerate_records`.
    fn arb_waveform() -> impl Strategy<Value = Array1<f64>> {
        prop::collection::vec(-1.0f64..1.0, 1..=MAX_NT)
            .prop_filter("waveform is identically zero", |samples| {
                samples.iter().any(|&sample| sample != 0.0)
            })
            .prop_map(Array1::from_vec)
    }

    /// A linearly polarised record at an arbitrary angle: the family that
    /// attains the bound.
    fn arb_polarised_components() -> impl Strategy<Value = (Array1<f64>, Array1<f64>)> {
        (arb_waveform(), 0.0f64..TAU).prop_map(|(waveform, angle)| polarised(&waveform, angle))
    }

    /// A record anywhere on the continuum between the two extremes: a polarised
    /// record at `angle` with independent noise mixed into each component. At
    /// `polarisation` 1 the components are perfectly correlated (the sqrt(2)
    /// worst case), at 0 the noise dominates, and the interesting near-polarised
    /// records are the whole stretch in between.
    ///
    /// Sweeping one parameter rather than picking between separate strategies
    /// keeps shrinking effective: a counterexample reduces along `polarisation`
    /// and the waveform together instead of stalling on a `prop_oneof` branch.
    fn arb_record() -> impl Strategy<Value = (Array1<f64>, Array1<f64>)> {
        let samples = prop::collection::vec((-1.0f64..1.0, -1.0f64..1.0, -1.0f64..1.0), 1..=MAX_NT);
        (samples, 0.0f64..TAU, 0.0f64..=1.0).prop_map(|(samples, angle, polarisation)| {
            let waveform: Array1<f64> = samples.iter().map(|&(w, _, _)| w).collect();
            let noise_0: Array1<f64> = samples.iter().map(|&(_, n, _)| n).collect();
            let noise_90: Array1<f64> = samples.iter().map(|&(_, _, n)| n).collect();
            let (comp_0, comp_90) = polarised(&waveform, angle);
            let noise_scale = 1.0 - polarisation;
            (
                comp_0 + noise_0 * noise_scale,
                comp_90 + noise_90 * noise_scale,
            )
        })
    }

    /// A batch of records as the (ns, nt) arrays the public entry points take.
    fn arb_batch() -> impl Strategy<Value = (Array2<f64>, Array2<f64>)> {
        (1usize..8, 1usize..MAX_NT).prop_flat_map(|(ns, nt)| {
            prop::collection::vec((-1.0f64..1.0, -1.0f64..1.0), ns * nt).prop_map(move |samples| {
                let split = |pick: fn(&(f64, f64)) -> f64| {
                    Array2::from_shape_vec((ns, nt), samples.iter().map(pick).collect()).unwrap()
                };
                (split(|&(x, _)| x), split(|&(_, y)| y))
            })
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

        #[test]
        fn prop_rotd100_never_exceeds_sqrt_2_rotd50((comp_0, comp_90) in arb_record()) {
            assert_ratio_bounded(comp_0.view(), comp_90.view(), "generated record");
        }

        #[test]
        fn prop_polarised_records_attain_the_bound((comp_0, comp_90) in arb_polarised_components()) {
            // The bound is not merely respected by linearly polarised records,
            // it is met: their peak in every direction is |cos(theta - angle)|
            // of the RotD100 peak, so the ratio is a property of the 1 degree
            // sampling alone and comes out at sqrt(2) whatever the waveform.
            let ratio = assert_ratio_bounded(comp_0.view(), comp_90.view(), "polarised record");
            prop_assert!(
                (ratio - SQRT_2).abs() <= RATIO_TOL,
                "expected a polarised record to attain sqrt(2), found {ratio}"
            );
        }

        #[test]
        fn prop_batch_api_agrees_and_is_bounded((comp_0, comp_90) in arb_batch()) {
            let serial = rotd(comp_0.view(), comp_90.view());
            let parallel = rotd_parallel(comp_0.view(), comp_90.view());
            prop_assert_eq!(&serial, &parallel, "rotd and rotd_parallel disagree");

            for (i, stats) in serial.rows().into_iter().enumerate() {
                let expected = rotd_calculation(comp_0.row(i), comp_90.row(i));
                prop_assert_eq!(
                    stats,
                    ArrayView1::from(&expected),
                    "row {} disagrees with rotd_calculation", i
                );
                assert_ratio_bounded(comp_0.row(i), comp_90.row(i), &format!("row {i}"));
            }
        }
    }

    #[test]
    fn test_rotd_calculation() {
        let comp_0 = array![1.0f64, 0.0f64];
        let comp_90 = array![0.0f64, 1.0f64];
        let [min, median, max] = rotd_calculation(comp_0.view(), comp_90.view());
        let expected_min = 2.0f64.sqrt() / 2.0; // e.g. at pi / 4 degrees
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

    #[test]
    fn test_ratio_bound_degenerate_records() {
        // Cases proptest will not reach on its own: exact zeros, single
        // samples, and the ends of the floating point range.
        let zeros = Array1::zeros(32);
        assert_ratio_bounded(zeros.view(), zeros.view(), "zero record");

        let single = array![-3.5];
        assert_ratio_bounded(single.view(), array![0.0].view(), "single sample");
        assert_ratio_bounded(
            single.view(),
            single.view(),
            "single sample, both components",
        );

        let constant = Array1::from_elem(32, 2.0);
        assert_ratio_bounded(constant.view(), zeros.view(), "constant offset");

        let mut impulse = Array1::zeros(32);
        impulse[7] = 9.81;
        assert_ratio_bounded(impulse.view(), zeros.view(), "impulse, one component");
        let flipped = &impulse * -1.0;
        assert_ratio_bounded(impulse.view(), flipped.view(), "impulse, anti-correlated");

        for scale in [1e-300, 1e-12, 1.0, 1e12, 1e300] {
            let comp_0 = &impulse * scale;
            let comp_90 = &constant * scale;
            assert_ratio_bounded(comp_0.view(), comp_90.view(), &format!("scale {scale:e}"));
        }
    }

    #[test]
    fn test_ratio_bound_circular_records() {
        // A circularly polarised record peaks identically in every direction,
        // so it sits at the opposite extreme from the bound, with a ratio of 1.
        let nt = 3600;
        let comp_0 = Array1::from_shape_fn(nt, |i| (TAU * i as f64 / nt as f64).cos());
        let comp_90 = Array1::from_shape_fn(nt, |i| (TAU * i as f64 / nt as f64).sin());
        let ratio = assert_ratio_bounded(comp_0.view(), comp_90.view(), "circular");
        assert!(
            (ratio - 1.0).abs() < 1e-3,
            "circular: expected a ratio of ~1, found {ratio}"
        );
    }
}
