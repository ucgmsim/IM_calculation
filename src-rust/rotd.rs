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

/// Fill an (ns, 3) array with the RotD statistics of each waveform pair.
pub fn rotd(comp_0: ArrayView2<f64>, comp_90: ArrayView2<f64>) -> Array2<f64> {
    assert_eq!(
        comp_0.nrows(),
        comp_90.nrows(),
        "Components must have the same number of waveforms"
    );
    let mut out = Array2::zeros((comp_0.nrows(), 3));
    Zip::from(out.rows_mut())
        .and(comp_0.rows())
        .and(comp_90.rows())
        .for_each(|mut out, comp_0, comp_90| {
            out.assign(&ArrayView1::from(&rotd_calculation(comp_0, comp_90)));
        });
    out
}

const fn cross(o: [f64; 2], u: [f64; 2], v: [f64; 2]) -> f64 {
    (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
}

/// Peak rotated amplitude at every integer angle 0..=179 degrees for one pair
/// of components, writing its work into caller-provided buffers so nothing is
/// allocated per station.
///
/// The peak at angle theta is the support function of the response trajectory
/// `(comp_0, comp_90)` along the rotated axis, which is maximised at a vertex
/// of the trajectory's convex hull. The hull is found with Akl-Toussaint
/// culling followed by a monotone chain: a first O(n) pass takes the four
/// axis-extreme points, and any point strictly inside the quadrilateral they
/// span cannot be a hull vertex and is dropped, so the sort that follows sees
/// only a few hundred of the tens of thousands of timesteps. The 180
/// evaluations over the resulting handful of hull vertices are then exact and
/// cheap; `rotd180_matches_brute` pins the result against the direct scan.
pub fn rotd180_peaks(
    x: ArrayView1<f64>,
    y: ArrayView1<f64>,
    survivors: &mut Vec<[f64; 2]>,
    hull: &mut Vec<[f64; 2]>,
) -> [f64; 180] {
    let n = x.len();
    // Axis extremes: a = min x, c = max x, b = max y, d = min y.
    let p0 = [x[0], y[0]];
    let (mut a, mut b, mut c, mut d) = (p0, p0, p0, p0);
    for i in 1..n {
        let p = [x[i], y[i]];
        if p[0] < a[0] {
            a = p;
        }
        if p[0] > c[0] {
            c = p;
        }
        if p[1] > b[1] {
            b = p;
        }
        if p[1] < d[1] {
            d = p;
        }
    }
    // Keep points that are not strictly inside quad a-b-c-d. Sign-agnostic so
    // it holds whatever the quad's winding, and a degenerate (collinear)
    // extreme set simply keeps everything.
    survivors.clear();
    for i in 0..n {
        let p = [x[i], y[i]];
        let (e0, e1, e2, e3) = (cross(a, b, p), cross(b, c, p), cross(c, d, p), cross(d, a, p));
        let inside = (e0 > 0.0 && e1 > 0.0 && e2 > 0.0 && e3 > 0.0)
            || (e0 < 0.0 && e1 < 0.0 && e2 < 0.0 && e3 < 0.0);
        if !inside {
            survivors.push(p);
        }
    }
    survivors.sort_unstable_by(|p, q| p[0].total_cmp(&q[0]).then(p[1].total_cmp(&q[1])));
    survivors.dedup();

    hull.clear();
    if survivors.len() < 3 {
        hull.extend_from_slice(survivors);
    } else {
        for &p in survivors.iter() {
            while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
                hull.pop();
            }
            hull.push(p);
        }
        let lower = hull.len() + 1;
        for &p in survivors.iter().rev() {
            while hull.len() >= lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    std::array::from_fn(|theta| {
        let (sin_theta, cos_theta) = (theta as f64 * DEGREES).sin_cos();
        hull.iter().fold(0.0f64, |peak, &[hx, hy]| {
            peak.max((cos_theta * hx + sin_theta * hy).abs())
        })
    })
}

/// Fill an `(ns, 180)` array with the per-angle peaks of each response pair,
/// serially, allocating the two [`rotd180_peaks`] work buffers once.
pub fn rotd180_rows(comp_0: ArrayView2<f64>, comp_90: ArrayView2<f64>) -> Array2<f64> {
    let ns = comp_0.nrows();
    let mut out = Array2::zeros((ns, 180));
    let mut survivors: Vec<[f64; 2]> = Vec::with_capacity(comp_0.ncols());
    let mut hull: Vec<[f64; 2]> = Vec::with_capacity(256);
    for s in 0..ns {
        let peaks = rotd180_peaks(comp_0.row(s), comp_90.row(s), &mut survivors, &mut hull);
        out.row_mut(s).assign(&ArrayView1::from(&peaks));
    }
    out
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{SQRT_2, TAU};

    use ndarray::prelude::*;
    use ndarray::Zip;
    use proptest::prelude::*;

    use crate::rotd::{rotd, rotd180_peaks, rotd180_rows, rotd_calculation, DEGREES};

    /// Slack allowed on the sqrt(2) bound. The bound is attained exactly by
    /// linearly polarised records, so only floating point error is tolerated.
    const RATIO_TOL: f64 = 1e-12;

    /// Longest generated waveform. RotD is O(180 * nt), so this keeps a full
    /// proptest run to well under a second.
    const MAX_NT: usize = 128;

    /// The direct scan the culled `rotd180_peaks` must reproduce: every angle
    /// against every timestep, no hull reduction.
    fn brute_peaks(x: ArrayView1<f64>, y: ArrayView1<f64>) -> [f64; 180] {
        std::array::from_fn(|theta| {
            let (sin_theta, cos_theta) = (theta as f64 * DEGREES).sin_cos();
            Zip::from(x)
                .and(y)
                .fold(0.0f64, |m: f64, &a, &b| m.max((cos_theta * a + sin_theta * b).abs()))
        })
    }

    /// `rotd180_peaks` with freshly allocated buffers, for tests that do not
    /// exercise buffer reuse themselves.
    fn peaks(x: ArrayView1<f64>, y: ArrayView1<f64>) -> [f64; 180] {
        let mut survivors = Vec::new();
        let mut hull = Vec::new();
        rotd180_peaks(x, y, &mut survivors, &mut hull)
    }

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
            let stats = rotd(comp_0.view(), comp_90.view());
            let peaks = rotd180_rows(comp_0.view(), comp_90.view());

            for (i, row) in stats.rows().into_iter().enumerate() {
                let expected = rotd_calculation(comp_0.row(i), comp_90.row(i));
                prop_assert_eq!(
                    row,
                    ArrayView1::from(&expected),
                    "row {} disagrees with rotd_calculation", i
                );
                // The RotD180 batch's min/median/max must match the 3-stat
                // reducer, tying the two public entry points together.
                let mut sorted: Vec<f64> = peaks.row(i).to_vec();
                sorted.sort_unstable_by(f64::total_cmp);
                prop_assert!((sorted[0] - expected[0]).abs() <= 1e-9 * expected[0].max(1.0));
                prop_assert!(
                    ((sorted[89] + sorted[90]) / 2.0 - expected[1]).abs()
                        <= 1e-9 * expected[1].max(1.0)
                );
                prop_assert!((sorted[179] - expected[2]).abs() <= 1e-9 * expected[2].max(1.0));
                assert_ratio_bounded(comp_0.row(i), comp_90.row(i), &format!("row {i}"));
            }
        }

        #[test]
        fn prop_culled_matches_brute((comp_0, comp_90) in arb_record()) {
            // The interior culling must never change a single per-angle peak,
            // across the full spread from polarised to near-circular records.
            let got = peaks(comp_0.view(), comp_90.view());
            let want = brute_peaks(comp_0.view(), comp_90.view());
            for theta in 0..180 {
                prop_assert!(
                    (got[theta] - want[theta]).abs() <= 1e-9 * want[theta].max(1.0),
                    "angle {}: culled {} != brute {}", theta, got[theta], want[theta]
                );
            }
        }
    }

    #[test]
    fn rotd180_matches_brute() {
        // The per-angle peaks must match a direct scan at every angle, and once
        // sorted must reproduce the RotD00/50/100 statistics of the reference.
        let nt = 733;
        let comp_0 = Array1::from_shape_fn(nt, |i| {
            let t = i as f64;
            (0.31 * t).sin() * (0.007 * t).cos() - 0.4 * (0.13 * t).sin()
        });
        let comp_90 = Array1::from_shape_fn(nt, |i| {
            let t = i as f64;
            (0.17 * t).cos() + 0.6 * (0.05 * t).sin() * (0.002 * t).cos()
        });

        let got = peaks(comp_0.view(), comp_90.view());
        let want = brute_peaks(comp_0.view(), comp_90.view());
        for theta in 0..180 {
            assert!(
                (got[theta] - want[theta]).abs() <= 1e-9 * want[theta].max(1.0),
                "angle {theta}: culled {} != brute {}", got[theta], want[theta]
            );
        }

        let mut sorted = got;
        sorted.sort_unstable_by(f64::total_cmp);
        let [min, median, max] = rotd_calculation(comp_0.view(), comp_90.view());
        assert!((sorted[0] - min).abs() <= 1e-9 * min.max(1.0));
        assert!(((sorted[89] + sorted[90]) / 2.0 - median).abs() <= 1e-9 * median.max(1.0));
        assert!((sorted[179] - max).abs() <= 1e-9 * max.max(1.0));
    }

    #[test]
    fn rotd180_matches_brute_on_circular_record() {
        // A near-circular trajectory is the culling's worst case: almost every
        // point sits near the hull, so few are dropped. The result must still
        // be exact.
        let nt = 2000;
        let comp_0 = Array1::from_shape_fn(nt, |i| (TAU * i as f64 / nt as f64).cos());
        let comp_90 = Array1::from_shape_fn(nt, |i| (TAU * i as f64 / nt as f64).sin());
        let got = peaks(comp_0.view(), comp_90.view());
        let want = brute_peaks(comp_0.view(), comp_90.view());
        for theta in 0..180 {
            assert!((got[theta] - want[theta]).abs() <= 1e-9, "circular angle {theta}");
        }
    }

    #[test]
    fn rotd180_handles_degenerate_records() {
        // Zero, single-sample and collinear records must not panic and must
        // agree with the reference peaks (all zero, or |projection|).
        let zeros = Array1::<f64>::zeros(16);
        assert_eq!(peaks(zeros.view(), zeros.view()), [0.0; 180]);

        let single = array![3.5];
        let single_peaks = peaks(single.view(), array![0.0].view());
        assert!((single_peaks[0] - 3.5).abs() < 1e-12);
        assert!(single_peaks[90].abs() < 1e-12);

        // A linearly polarised (collinear) record: peaks trace |cos(theta)|.
        let line_0 = array![1.0, -2.0, 3.0, -4.0];
        let line_90 = &line_0 * 2.0;
        let got = peaks(line_0.view(), line_90.view());
        let want = brute_peaks(line_0.view(), line_90.view());
        for theta in 0..180 {
            assert!((got[theta] - want[theta]).abs() < 1e-9, "collinear angle {theta}");
        }
    }

    #[test]
    fn rotd180_buffers_reset_between_records() {
        // Reusing the same buffers across records of different lengths and
        // shapes must give exactly the same answer as fresh buffers each call:
        // a guard against a missing clear() leaking state between stations.
        let mut survivors = Vec::new();
        let mut hull = Vec::new();
        let records = [
            (array![1.0, -2.0, 3.0], array![0.5, 0.5, -1.0]),
            (Array1::zeros(5), Array1::zeros(5)),
            (array![9.81, -3.0, 2.0, 7.0, -8.0, 1.0], array![1.0, 4.0, -4.0, 0.0, 2.0, -2.0]),
            (array![2.5], array![-1.5]),
        ];
        for (comp_0, comp_90) in &records {
            let reused = rotd180_peaks(comp_0.view(), comp_90.view(), &mut survivors, &mut hull);
            let want = brute_peaks(comp_0.view(), comp_90.view());
            for theta in 0..180 {
                assert!(
                    (reused[theta] - want[theta]).abs() <= 1e-9 * want[theta].max(1.0),
                    "reused buffers diverged at angle {theta}"
                );
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
