use std::f64::consts::PI;

use ndarray::prelude::*;

const DEGREES: f64 = PI / 180.0;

/// Integer rotation angles RotD samples: 0, 1, ..., 179 degrees.
pub const N_ANGLES: usize = 180;

/// Columns in a RotD statistics row. Peak amplitudes for RotD00, RotD50 and
/// RotD100 come first. The RotD00 and RotD100 orientations, in degrees,
/// follow them.
pub const N_ROTD_STATS: usize = 5;

const fn cross(o: [f64; 2], u: [f64; 2], v: [f64; 2]) -> f64 {
    (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
}

/// Extend `vertices` with one monotone chain over `points`, dropping any
/// trailing vertex that would make a non-left turn.
///
/// `floor` is the number of vertices already in `vertices` that belong to an
/// earlier chain and survive every pop: 1 for the lower hull (its own first
/// point), and the whole lower hull for the upper one.
fn monotone_chain(
    vertices: &mut Vec<[f64; 2]>,
    points: impl Iterator<Item = [f64; 2]>,
    floor: usize,
) {
    for p in points {
        while vertices.len() > floor {
            let (o, u) = (vertices[vertices.len() - 2], vertices[vertices.len() - 1]);
            if cross(o, u, p) > 0.0 {
                break;
            }
            vertices.pop();
        }
        vertices.push(p);
    }
}

/// The convex hull of a response trajectory, and the scratch space used to
/// find it.
#[derive(Default)]
pub struct Hull {
    /// Points left after the Akl-Toussaint cull, sorted lexicographically.
    survivors: Vec<[f64; 2]>,
    /// The hull vertices themselves, the only points [`Hull::peaks`] scans.
    vertices: Vec<[f64; 2]>,
}

impl Hull {
    /// A hull sized for records of `nt` timesteps.
    pub fn with_capacity(nt: usize) -> Self {
        Self {
            survivors: Vec::with_capacity(nt),
            vertices: Vec::with_capacity(256),
        }
    }

    /// Peak rotated amplitude at every integer angle 0..=179 degrees for one
    /// pair of components.
    pub fn peaks(&mut self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> [f64; N_ANGLES] {
        let n = x.len();
        // Axis extremes, in order around the trajectory, starting at min x
        // and going through max y, max x and min y.
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
        // In some cases the Akl-Toussaint culling box degenerates into a
        // triangle. In that case (a, b, c, d) contains repeat points which
        // creates problems for the culling because then the edge = 0 and
        // cross(0, u, v) == 0 for all u, v, which stops the hull doing
        // anything. This loop removes the repeats.
        let mut ring = [p0; 5];
        let mut corners = 0;
        for corner in [a, b, c, d] {
            if !ring[..corners].contains(&corner) {
                ring[corners] = corner;
                corners += 1;
            }
        }
        ring[corners] = ring[0];

        // Now for the culling box edges. A box that's actually a triangle
        // repeats an edge twice. That duplicates work in the culling loop
        // below, but it beats dropping the extra edge, because Rust optimises
        // the predictable cross-products well. Deciding the edge count at run
        // time costs 10x in the calculation.
        let [edge_0, edge_1, edge_2, edge_3] = std::array::from_fn(|j| {
            let corner = j.min(corners - 1);
            [ring[corner], ring[corner + 1]]
        });

        // Now cull all points inside the box.
        self.survivors.clear();
        for i in 0..n {
            let p = [x[i], y[i]];
            let (e0, e1, e2, e3) = (
                cross(edge_0[0], edge_0[1], p),
                cross(edge_1[0], edge_1[1], p),
                cross(edge_2[0], edge_2[1], p),
                cross(edge_3[0], edge_3[1], p),
            );
            let inside = (e0 > 0.0 && e1 > 0.0 && e2 > 0.0 && e3 > 0.0)
                || (e0 < 0.0 && e1 < 0.0 && e2 < 0.0 && e3 < 0.0);
            if !inside {
                self.survivors.push(p);
            }
        }
        self.survivors
            .sort_unstable_by(|p, q| p[0].total_cmp(&q[0]).then(p[1].total_cmp(&q[1])));
        self.survivors.dedup();

        self.vertices.clear();
        if self.survivors.len() < 3 {
            // Degenerate triangular case. Triangle is always its own convex hull.
            self.vertices.extend_from_slice(&self.survivors);
        } else {
            monotone_chain(&mut self.vertices, self.survivors.iter().copied(), 1);
            let lower_hull = self.vertices.len();
            monotone_chain(
                &mut self.vertices,
                self.survivors.iter().rev().copied(),
                lower_hull,
            );
            // The upper chain closes back on the lower chain's first point.
            self.vertices.pop();
        }
        std::array::from_fn(|theta| {
            let (sin_theta, cos_theta) = (theta as f64 * DEGREES).sin_cos();
            self.vertices.iter().fold(0.0f64, |peak, &[hx, hy]| {
                peak.max((cos_theta * hx + sin_theta * hy).abs())
            })
        })
    }
}

/// Reduce the 180 per-angle peaks to the (min, median, max) rotated
/// amplitude. The row also gives the orientation of the min and the max.
pub(crate) fn rotd_stats(peaks: [f64; N_ANGLES]) -> [f64; N_ROTD_STATS] {
    // Strict comparisons keep the lowest angle in place on a tie.
    let (mut min_angle, mut max_angle) = (0usize, 0usize);
    for theta in 1..N_ANGLES {
        if peaks[theta] < peaks[min_angle] {
            min_angle = theta;
        }
        if peaks[theta] > peaks[max_angle] {
            max_angle = theta;
        }
    }
    // An even number of samples. The median falls between the two central
    // peaks, and takes the average of that pair.
    let mut ranked = peaks;
    ranked.sort_unstable_by(f64::total_cmp);
    [
        peaks[min_angle],
        (ranked[89] + ranked[90]) / 2.0,
        peaks[max_angle],
        min_angle as f64,
        max_angle as f64,
    ]
}

/// Fill an `(ns, 5)` array with the RotD statistics of each waveform pair:
/// three peak amplitudes then two orientations, as laid out by
/// [`rotd_stats`].
pub fn rotd(comp_0: ArrayView2<f64>, comp_90: ArrayView2<f64>) -> Array2<f64> {
    assert_eq!(
        comp_0.nrows(),
        comp_90.nrows(),
        "Components must have the same number of waveforms"
    );
    let mut out = Array2::zeros((comp_0.nrows(), N_ROTD_STATS));
    let mut hull = Hull::with_capacity(comp_0.ncols());
    for s in 0..comp_0.nrows() {
        let stats = rotd_stats(hull.peaks(comp_0.row(s), comp_90.row(s)));
        out.row_mut(s).assign(&ArrayView1::from(&stats));
    }
    out
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{SQRT_2, TAU};

    use ndarray::Zip;
    use ndarray::prelude::*;
    use proptest::prelude::*;

    use crate::rotd::{DEGREES, Hull, N_ANGLES, N_ROTD_STATS, rotd, rotd_stats};

    /// Fill an `(ns, 180)` array with the per-angle peaks of each response pair,
    /// serially, allocating the [`Hull`] work buffers once.
    fn rotd180_rows(comp_0: ArrayView2<f64>, comp_90: ArrayView2<f64>) -> Array2<f64> {
        let ns = comp_0.nrows();
        let mut out = Array2::zeros((ns, N_ANGLES));
        let mut hull = Hull::with_capacity(comp_0.ncols());
        for s in 0..ns {
            let peaks = hull.peaks(comp_0.row(s), comp_90.row(s));
            out.row_mut(s).assign(&ArrayView1::from(&peaks));
        }
        out
    }

    /// Slack allowed on the sqrt(2) bound. Linearly polarised records reach
    /// the bound exactly, so this leaves room for floating point error only.
    const RATIO_TOL: f64 = 1e-12;

    /// Longest generated waveform. RotD is O(180 * nt), so this keeps a full
    /// proptest run to well under a second.
    const MAX_NT: usize = 128;

    /// The direct scan the culled [`Hull::peaks`] must reproduce: every angle
    /// against every timestep, over the full point set.
    fn brute_peaks(x: ArrayView1<f64>, y: ArrayView1<f64>) -> [f64; N_ANGLES] {
        std::array::from_fn(|theta| {
            let (sin_theta, cos_theta) = (theta as f64 * DEGREES).sin_cos();
            Zip::from(x).and(y).fold(0.0f64, |m: f64, &a, &b| {
                m.max((cos_theta * a + sin_theta * b).abs())
            })
        })
    }

    /// [`Hull::peaks`] with a fresh hull, for tests that don't exercise
    /// buffer reuse themselves.
    fn peaks(x: ArrayView1<f64>, y: ArrayView1<f64>) -> [f64; N_ANGLES] {
        Hull::default().peaks(x, y)
    }

    /// The statistics the hull-based [`rotd`] must reproduce, taken from the
    /// direct scan rather than from the hull.
    fn brute_stats(x: ArrayView1<f64>, y: ArrayView1<f64>) -> [f64; N_ROTD_STATS] {
        rotd_stats(brute_peaks(x, y))
    }

    /// Assert that the RotD00 and RotD100 orientations locate their own
    /// statistic in the sweep `rotd_stats` reduced.
    fn assert_orientations_locate_statistics(peaks: [f64; N_ANGLES], case: &str) {
        let [rotd00, rotd50, rotd100, at_00, at_100] = rotd_stats(peaks);
        for (angle, statistic) in [(at_00, "RotD00"), (at_100, "RotD100")] {
            assert!(
                (0.0..N_ANGLES as f64).contains(&angle) && angle.fract() == 0.0,
                "{case}: {statistic} orientation {angle} is not an integer angle in 0..180"
            );
        }
        // The extremes land exactly on their own angle.
        assert_eq!(
            peaks[at_00 as usize], rotd00,
            "{case}: RotD00 is not the peak at {at_00} degrees"
        );
        assert_eq!(
            peaks[at_100 as usize], rotd100,
            "{case}: RotD100 is not the peak at {at_100} degrees"
        );
        assert!(
            rotd00 <= rotd50 && rotd50 <= rotd100,
            "{case}: RotD50 {rotd50} is not between RotD00 {rotd00} and RotD100 {rotd100}"
        );
        // The lowest angle of a tie, for both extremes.
        for (angle, value) in [(at_00, rotd00), (at_100, rotd100)] {
            let first = peaks.iter().position(|&peak| peak == value).unwrap();
            assert_eq!(
                angle as usize, first,
                "{case}: {value} is first attained at {first} degrees, not {angle}"
            );
        }
    }

    /// Assert RotD100 <= sqrt(2) * RotD50 for one pair of components, returning
    /// the ratio.
    ///
    /// The bound holds because the rotated traces at angles theta and
    /// theta + 90 degrees, sampled at the time of the RotD100 peak, have squared
    /// amplitudes summing to RotD100^2 at least. So one angle of every such
    /// pair at minimum (90 of the 180 angles) peaks at RotD100 / sqrt(2) or
    /// higher, which puts the median there too.
    fn assert_ratio_bounded(comp_0: ArrayView1<f64>, comp_90: ArrayView1<f64>, case: &str) -> f64 {
        let [rotd00, rotd50, rotd100, ..] = brute_stats(comp_0, comp_90);
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
    /// invariant. Varying the amplitude tests nothing new, and the
    /// fixed tests cover the extremes of the floating point range instead.
    /// This strategy leaves out the all-zero record, which has no
    /// polarisation direction;
    /// `test_ratio_bound_degenerate_records` covers that one.
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
    /// and the waveform together instead of stopping at a `prop_oneof` branch.
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
            // Linearly polarised records don't merely respect the bound, they
            // reach it exactly: their peak in every direction is
            // |cos(theta - angle)| of the RotD100 peak, so the ratio is a
            // property of the 1 degree sampling alone and comes out at sqrt(2)
            // whatever the waveform.
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
                // Both entry points reduce the same hull, so the (ns, 3)
                // statistics must be exactly the reduction of the (ns, 180)
                // curve (no tolerance needed to tie them together).
                let row_peaks: [f64; N_ANGLES] = std::array::from_fn(|theta| peaks[(i, theta)]);
                let curve_stats = rotd_stats(row_peaks);
                prop_assert_eq!(
                    row,
                    ArrayView1::from(&curve_stats),
                    "row {} disagrees with its own RotD180 curve", i
                );
                // And the hull must reproduce the direct scan's peaks, up
                // to the floating point slack of evaluating fewer points. The
                // orientations stay out of this comparison: a near-tie at an
                // extreme can settle on either of two angles under that slack,
                // so prop_orientations_locate_their_statistics pins them to
                // their own sweep instead.
                let expected = brute_stats(comp_0.row(i), comp_90.row(i));
                for (stat, (&got, &want)) in row.iter().zip(expected.iter()).take(3).enumerate() {
                    prop_assert!(
                        (got - want).abs() <= 1e-9 * want.max(1.0),
                        "row {} stat {}: hull {} != brute {}", i, stat, got, want
                    );
                }
                assert_ratio_bounded(comp_0.row(i), comp_90.row(i), &format!("row {i}"));
            }
        }

        #[test]
        fn prop_orientations_locate_their_statistics((comp_0, comp_90) in arb_record()) {
            // Whatever the record, each reported orientation must point at
            // the angle its statistic came from.
            assert_orientations_locate_statistics(
                peaks(comp_0.view(), comp_90.view()),
                "generated record",
            );
            assert_orientations_locate_statistics(
                brute_peaks(comp_0.view(), comp_90.view()),
                "generated record (brute)",
            );
        }

        #[test]
        fn prop_culled_matches_brute((comp_0, comp_90) in arb_record()) {
            // The interior culling must never change one per-angle peak,
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
                "angle {theta}: culled {} != brute {}",
                got[theta],
                want[theta]
            );
        }

        let mut sorted = got;
        sorted.sort_unstable_by(f64::total_cmp);
        let [min, median, max, ..] = brute_stats(comp_0.view(), comp_90.view());
        assert!((sorted[0] - min).abs() <= 1e-9 * min.max(1.0));
        assert!(((sorted[89] + sorted[90]) / 2.0 - median).abs() <= 1e-9 * median.max(1.0));
        assert!((sorted[179] - max).abs() <= 1e-9 * max.max(1.0));
    }

    #[test]
    fn rotd180_matches_brute_on_circular_record() {
        // A near-circular trajectory is the culling's worst case: almost every
        // point lies near the hull. The cull removes few of them, and the
        // result must still be exact.
        let nt = 2000;
        let comp_0 = Array1::from_shape_fn(nt, |i| (TAU * i as f64 / nt as f64).cos());
        let comp_90 = Array1::from_shape_fn(nt, |i| (TAU * i as f64 / nt as f64).sin());
        let got = peaks(comp_0.view(), comp_90.view());
        let want = brute_peaks(comp_0.view(), comp_90.view());
        for theta in 0..180 {
            assert!(
                (got[theta] - want[theta]).abs() <= 1e-9,
                "circular angle {theta}"
            );
        }
    }

    #[test]
    fn rotd180_handles_degenerate_records() {
        // Zero, single-sample and collinear records must not panic and must
        // agree with the reference peaks (all zero, or |projection|).
        let zeros = Array1::<f64>::zeros(16);
        assert_eq!(peaks(zeros.view(), zeros.view()), [0.0; N_ANGLES]);

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
            assert!(
                (got[theta] - want[theta]).abs() < 1e-9,
                "collinear angle {theta}"
            );
        }
    }

    #[test]
    fn cull_drops_the_interior_when_a_point_is_extreme_in_two_axes() {
        // One timestep that's both the max in x and the min in y collapses
        // the extreme quadrilateral onto a triangle. The cull must still drop
        // the interior. With a zero-length quad edge every cross product comes
        // out zero, which puts every point outside the box, and the sort then
        // handles the whole record, which is how 3497857_PARS_HN_20 lost 7x
        // of its speedup.
        let interior = 1000;
        let mut comp_0 = Array1::zeros(interior + 3);
        let mut comp_90 = Array1::zeros(interior + 3);
        for i in 0..interior {
            let t = TAU * i as f64 / interior as f64;
            comp_0[i] = 0.01 * t.cos();
            comp_90[i] = 0.01 * t.sin();
        }
        // min x, max y, and one corner that's both max x and min y.
        let extremes = [[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]];
        for (i, [px, py]) in extremes.into_iter().enumerate() {
            comp_0[interior + i] = px;
            comp_90[interior + i] = py;
        }

        let mut hull = Hull::default();
        let got = hull.peaks(comp_0.view(), comp_90.view());
        assert!(
            hull.survivors.len() < interior / 10,
            "cull kept {} of {} points",
            hull.survivors.len(),
            interior + 3
        );
        let want = brute_peaks(comp_0.view(), comp_90.view());
        for theta in 0..180 {
            assert!(
                (got[theta] - want[theta]).abs() <= 1e-9 * want[theta].max(1.0),
                "doubly extreme angle {theta}"
            );
        }
    }

    #[test]
    fn rotd180_buffers_reset_between_records() {
        // Reusing the same buffers across records of different lengths and
        // shapes must give exactly the same answer as fresh buffers each call:
        // a guard against a missing clear() leaking state between stations.
        let mut hull = Hull::default();
        let records = [
            (array![1.0, -2.0, 3.0], array![0.5, 0.5, -1.0]),
            (Array1::zeros(5), Array1::zeros(5)),
            (
                array![9.81, -3.0, 2.0, 7.0, -8.0, 1.0],
                array![1.0, 4.0, -4.0, 0.0, 2.0, -2.0],
            ),
            (array![2.5], array![-1.5]),
        ];
        for (comp_0, comp_90) in &records {
            let reused = hull.peaks(comp_0.view(), comp_90.view());
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
    fn test_rotd_statistics() {
        let comp_0 = array![1.0f64, 0.0f64];
        let comp_90 = array![0.0f64, 1.0f64];
        let [min, median, max, at_min, at_max] = rotd_stats(peaks(comp_0.view(), comp_90.view()));
        let expected_min = 2.0f64.sqrt() / 2.0; // such as at pi / 4 degrees
        let expected_max = 1.0; // such as at 0 degrees
        let expected_median = 0.9238443540096138; // derived independently with numpy
        // The sweep is max(|cos theta|, |sin theta|): least at 45 degrees, and
        // 1 at both 0 and 90 degrees, of which `rotd_stats` reports the lower.
        assert_eq!(
            [at_min, at_max],
            [45.0, 0.0],
            "Orientations wrong: found {at_min}, {at_max} degrees"
        );
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
        // Cases beyond proptest's own reach: exact zeros, single samples, and
        // the ends of the floating point range.
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
        // putting it at the opposite extreme from the bound, with a ratio of 1.
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
