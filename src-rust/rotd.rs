use std::f64::consts::PI;

use ndarray::prelude::*;

const DEGREES: f64 = PI / 180.0;

/// Integer rotation angles for RotD: 0..=179 degrees. The peak is
/// an absolute value, so angles beyond 180 degrees repeat.
pub const N_ANGLES: usize = 180;

/// Columns in a RotD statistics row. First the RotD00, RotD50 and RotD100
/// peak amplitudes, then the orientation in degrees of each one.
pub const N_ROTD_STATS: usize = 6;

/// RotD00 and RotD100 read off the hull geometry rather than the 1 degree
/// grid, with the orientation each one occurs at.
///
/// Both are exact: the grid can only bracket them, since the true extreme of
/// the support function rarely falls on a whole degree. That makes `at_00`
/// and `at_100` real angles in `[0, 180)`, rather than the whole degrees the
/// RotD50 orientation still reports.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Extremes {
    pub rotd00: f64,
    pub at_00: f64,
    pub rotd100: f64,
    pub at_100: f64,
}

/// Fold a direction in radians into the `[0, 180)` degrees RotD covers. `u`
/// and `-u` give the same rotated peak, because the peak is an absolute
/// value, so only the half circle means anything.
fn fold_degrees(radians: f64) -> f64 {
    let degrees = radians.to_degrees().rem_euclid(180.0);
    // `rem_euclid` returns the modulus itself for an input a hair below zero,
    // because 180 minus a subnormal rounds back to 180.
    if degrees >= 180.0 { 0.0 } else { degrees }
}

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
///
/// Every RotD figure this module produces is a peak rotated amplitude, and
/// every one of them comes from [`Hull::peaks`]. The buffers live in the
/// struct rather than in locals so a batch of stations allocates once and
/// reuses the same two vectors for every record.
#[derive(Default)]
pub struct Hull {
    /// Points left after the Akl-Toussaint cull, sorted lexicographically.
    survivors: Vec<[f64; 2]>,
    /// The hull vertices themselves, the only points [`Hull::peaks`] scans.
    vertices: Vec<[f64; 2]>,
    /// `vertices` together with their negations, sorted lexicographically:
    /// the input to the origin-symmetric hull [`Hull::extremes`] reduces.
    reflected: Vec<[f64; 2]>,
    /// Vertices of that origin-symmetric hull.
    symmetric: Vec<[f64; 2]>,
}

impl Hull {
    /// A hull sized for records of `nt` timesteps.
    pub fn with_capacity(nt: usize) -> Self {
        Self {
            survivors: Vec::with_capacity(nt),
            vertices: Vec::with_capacity(256),
            reflected: Vec::with_capacity(512),
            symmetric: Vec::with_capacity(512),
        }
    }

    /// Build the convex hull of the response trajectory `(x, y)` into
    /// `self.vertices`.
    ///
    /// The peak at angle theta is the support function of the trajectory along
    /// the rotated axis, and it takes its maximum at a vertex of the
    /// trajectory's convex hull. Akl-Toussaint culling followed by a monotone
    /// chain gives that hull. A first O(n) pass takes the four axis-extreme
    /// points. Only a point outside the polygon they span can be a hull
    /// vertex, so the cull drops the rest. That leaves the sort with a few
    /// hundred of the tens of thousands of timesteps.
    fn build(&mut self, x: ArrayView1<f64>, y: ArrayView1<f64>) {
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
    }

    /// The peak rotated amplitude at every integer angle 0..=179 degrees, over
    /// the hull `build` left behind.
    ///
    /// The 180 evaluations over a few dozen hull vertices are exact and
    /// cheap. `rotd180_matches_brute` pins the result against the direct scan.
    fn sweep(&self) -> [f64; N_ANGLES] {
        std::array::from_fn(|theta| {
            let (sin_theta, cos_theta) = (theta as f64 * DEGREES).sin_cos();
            self.vertices.iter().fold(0.0f64, |peak, &[hx, hy]| {
                peak.max((cos_theta * hx + sin_theta * hy).abs())
            })
        })
    }

    /// RotD00 and RotD100 off the hull geometry, at their exact angles.
    ///
    /// **RotD100.** For one point `p` the rotated amplitude `|p . u_theta|` is
    /// largest at `theta = atan2(p_y, p_x)`, where it equals `|p|`. So the
    /// largest rotated amplitude of the whole trajectory is the largest `|p|`,
    /// and its own direction is the angle. The norm is convex, so a hull
    /// vertex holds that maximum.
    ///
    /// **RotD00.** `max_p |p . u|` is the support function of the
    /// origin-symmetric hull `S = conv(V union -V)`, since
    /// `max(max_v v.u, max_v -v.u)` is exactly `max_v |v.u|`. Restricted to one
    /// vertex's normal cone a support function is `|v| cos(theta - phi_v)`,
    /// which is concave, so its minimum over the cone falls on a boundary,
    /// and those boundaries are the edge normals of `S`. So the smallest
    /// rotated amplitude is the smallest distance from the origin to an edge
    /// of `S`: the rotating-calipers minimum width of a centrally symmetric
    /// polygon, halved. The origin lies inside `S` for any trajectory, which
    /// keeps every one of those distances positive.
    ///
    /// Both are exact rather than sampled, so each brackets the 1 degree grid:
    /// `rotd00 <= min(sweep)` and `rotd100 >= max(sweep)`.
    fn extremes(&mut self) -> Extremes {
        // `hypot` rather than the square root of a sum of squares. A record
        // scaled to 1e-300 underflows `px * px` to zero, and one scaled to
        // 1e300 overflows it. Either way RotD100 comes out below a RotD50 that
        // kept its scale. The vertices are a few hundred points at most, so the
        // slower call costs nothing that matters.
        let mut rotd100 = 0.0f64;
        let mut at_100 = 0.0f64;
        for &[px, py] in &self.vertices {
            let norm = px.hypot(py);
            let angle = fold_degrees(py.atan2(px));
            // Ties go to the lowest angle, the same rule the grid follows.
            if norm > rotd100 || (norm == rotd100 && angle < at_100) {
                rotd100 = norm;
                at_100 = angle;
            }
        }

        self.reflected.clear();
        for &[px, py] in &self.vertices {
            self.reflected.push([px, py]);
            self.reflected.push([-px, -py]);
        }
        self.reflected
            .sort_unstable_by(|p, q| p[0].total_cmp(&q[0]).then(p[1].total_cmp(&q[1])));
        self.reflected.dedup();

        self.symmetric.clear();
        if self.reflected.len() >= 3 {
            monotone_chain(&mut self.symmetric, self.reflected.iter().copied(), 1);
            let lower_hull = self.symmetric.len();
            monotone_chain(
                &mut self.symmetric,
                self.reflected.iter().rev().copied(),
                lower_hull,
            );
            self.symmetric.pop();
        }

        let (rotd00, at_00) = if self.symmetric.len() < 3 {
            // `S` came out a segment or a point, which is what a trajectory
            // confined to a line through the origin gives, whether that's a
            // linearly polarised record or one that never moves. The peak
            // across the polarisation is exactly zero, at a right angle to
            // the direction RotD100 came from.
            (0.0, fold_degrees((at_100 + 90.0).to_radians()))
        } else {
            let mut rotd00 = f64::INFINITY;
            let mut at_00 = 0.0f64;
            for i in 0..self.symmetric.len() {
                let a = self.symmetric[i];
                let b = self.symmetric[(i + 1) % self.symmetric.len()];
                let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                let length = dx.hypot(dy);
                if length == 0.0 {
                    continue;
                }
                // The outward normal of a counter-clockwise edge a -> b, as a
                // unit vector. Normalising before the dot product rather than
                // dividing a cross product by the edge length afterwards is
                // what keeps a record scaled to 1e300 from overflowing an
                // intermediate to infinity: every factor here is either a
                // coordinate or a number in [-1, 1].
                let (nx, ny) = (dy / length, -dx / length);
                // The height of the origin over the edge, which is the
                // support function in the normal's own direction.
                let distance = (a[0] * nx + a[1] * ny).abs();
                let angle = fold_degrees(ny.atan2(nx));
                if distance < rotd00 || (distance == rotd00 && angle < at_00) {
                    rotd00 = distance;
                    at_00 = angle;
                }
            }
            (rotd00, at_00)
        };

        Extremes {
            rotd00,
            at_00,
            rotd100,
            at_100,
        }
    }

    /// The peak rotated amplitude at every integer angle 0..=179 degrees for
    /// one pair of components.
    pub fn peaks(&mut self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> [f64; N_ANGLES] {
        self.build(x, y);
        self.sweep()
    }

    /// The 1 degree sweep and the exact extremes, off one hull.
    pub fn analyse(
        &mut self,
        x: ArrayView1<f64>,
        y: ArrayView1<f64>,
    ) -> ([f64; N_ANGLES], Extremes) {
        self.build(x, y);
        (self.sweep(), self.extremes())
    }
}

/// Assemble a statistics row from the exact extremes and the 1 degree sweep
/// the median comes off.
///
/// RotD00 and RotD100 come from [`Hull::extremes`], which reads them off the
/// hull geometry instead of the sweep, so both are exact and their
/// orientations are real angles rather than whole degrees.
///
/// RotD50 stays on the grid. The median of an even number of samples falls
/// between the two central ones, so its value remains the average of that
/// pair, and the orientation alongside it takes the lower of the two.
pub(crate) fn rotd_stats(peaks: [f64; N_ANGLES], extremes: Extremes) -> [f64; N_ROTD_STATS] {
    // Pair each peak with its angle through the sort, so the median has an
    // orientation. The angle breaks ties, putting equal peaks in ascending
    // angle whatever the sort's internal order.
    let mut ranked: [(f64, u8); N_ANGLES] =
        std::array::from_fn(|theta| (peaks[theta], theta as u8));
    ranked.sort_unstable_by(|p, q| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1)));
    let (lower_median, upper_median) = (ranked[89], ranked[90]);
    [
        extremes.rotd00,
        (lower_median.0 + upper_median.0) / 2.0,
        extremes.rotd100,
        extremes.at_00,
        f64::from(lower_median.1),
        extremes.at_100,
    ]
}

/// Fill an `(ns, 6)` array with the RotD statistics of each waveform pair:
/// three peak amplitudes then their three orientations, as laid out by
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
        let (peaks, extremes) = hull.analyse(comp_0.row(s), comp_90.row(s));
        out.row_mut(s)
            .assign(&ArrayView1::from(&rotd_stats(peaks, extremes)));
    }
    out
}

/// Fill an `(ns, 180)` array with the per-angle peaks of each response pair,
/// serially, allocating the [`Hull`] work buffers once.
pub fn rotd180_rows(comp_0: ArrayView2<f64>, comp_90: ArrayView2<f64>) -> Array2<f64> {
    let ns = comp_0.nrows();
    let mut out = Array2::zeros((ns, N_ANGLES));
    let mut hull = Hull::with_capacity(comp_0.ncols());
    for s in 0..ns {
        let peaks = hull.peaks(comp_0.row(s), comp_90.row(s));
        out.row_mut(s).assign(&ArrayView1::from(&peaks));
    }
    out
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{SQRT_2, TAU};

    use ndarray::Zip;
    use ndarray::prelude::*;
    use proptest::prelude::*;

    use crate::rotd::{
        DEGREES, Extremes, Hull, N_ANGLES, fold_degrees, rotd, rotd_stats, rotd180_rows,
    };

    /// Floating point slack, relative to the scale of the record.
    const TOL: f64 = 1e-9;

    /// Longest generated waveform. RotD is O(180 * nt), so this keeps a full
    /// proptest run to well under a second.
    const MAX_NT: usize = 128;

    /// Longest waveform handed to [`brute_extremes`], which is cubic.
    const MAX_BRUTE_NT: usize = 24;

    /// The tightest bound on RotD100 / RotD50, now that RotD100 comes off the
    /// hull and RotD50 still comes off the 1 degree grid.
    ///
    /// `f(theta) >= RotD100 * |cos(theta - phi)|` for every theta, phi being
    /// the direction RotD100 points along, because the furthest timestep on
    /// its own already projects that far. Order statistics inherit the
    /// inequality, so the two central grid peaks are at least
    /// `RotD100 * cos((45 -+ beta) deg)`, beta being phi's offset from the
    /// nearest whole degree, which puts RotD50 at `RotD100 * cos(beta) /
    /// sqrt(2)` or higher.
    ///
    /// Reading RotD100 off the grid as well forced beta to zero and made the
    /// bound sqrt(2) exactly. Reading it off the hull instead moves the peak
    /// up to half a degree off-grid, and that half degree is the whole of the
    /// difference. Polarised records still attain the bound.
    fn ratio_bound() -> f64 {
        SQRT_2 / (0.5 * DEGREES).cos()
    }

    /// The rotated peak at an arbitrary angle in degrees, straight off the
    /// timesteps: the function every RotD statistic is an extreme or a
    /// quantile of.
    fn support(x: ArrayView1<f64>, y: ArrayView1<f64>, degrees: f64) -> f64 {
        let (sin_theta, cos_theta) = (degrees * DEGREES).sin_cos();
        Zip::from(x).and(y).fold(0.0f64, |m: f64, &a, &b| {
            m.max((cos_theta * a + sin_theta * b).abs())
        })
    }

    /// The direct scan the culled [`Hull::peaks`] must reproduce: every angle
    /// against every timestep, with the hull reduction skipped.
    fn brute_peaks(x: ArrayView1<f64>, y: ArrayView1<f64>) -> [f64; N_ANGLES] {
        std::array::from_fn(|theta| support(x, y, theta as f64))
    }

    /// [`Hull::peaks`] with a fresh hull, for tests that don't exercise
    /// buffer reuse themselves.
    fn peaks(x: ArrayView1<f64>, y: ArrayView1<f64>) -> [f64; N_ANGLES] {
        Hull::default().peaks(x, y)
    }

    /// The exact extremes for one record, with a fresh hull.
    fn extremes(x: ArrayView1<f64>, y: ArrayView1<f64>) -> Extremes {
        Hull::default().analyse(x, y).1
    }

    /// RotD00 and RotD100 found without a hull, for the calipers to answer to.
    ///
    /// RotD100 falls out without searching, as the furthest timestep from
    /// the origin. RotD00 takes a search. The minimum of `f` can only fall
    /// where the timestep holding the peak changes hands, which is where
    /// `|p_i . u| = |p_j . u|` for some pair, and that puts `u` at a right
    /// angle to either `p_i - p_j` or `p_i + p_j`. Every edge of the
    /// origin-symmetric hull joins two points of `P union -P`, so its normal
    /// is in that set whichever way the signs fall. Taking `i = j` adds the
    /// one-point case, where the minimum is zero across the one direction of
    /// motion.
    ///
    /// Quadratic in the candidate count and cubic overall, so the callers keep
    /// their records inside [`MAX_BRUTE_NT`].
    fn brute_extremes(x: ArrayView1<f64>, y: ArrayView1<f64>) -> Extremes {
        let n = x.len();
        let mut rotd100 = 0.0f64;
        let mut at_100 = 0.0f64;
        for i in 0..n {
            let norm = x[i].hypot(y[i]);
            let angle = fold_degrees(y[i].atan2(x[i]));
            if norm > rotd100 || (norm == rotd100 && angle < at_100) {
                rotd100 = norm;
                at_100 = angle;
            }
        }

        let mut candidates = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in i..n {
                for (dx, dy) in [(x[i] - x[j], y[i] - y[j]), (x[i] + x[j], y[i] + y[j])] {
                    if dx != 0.0 || dy != 0.0 {
                        candidates.push(fold_degrees((-dx).atan2(dy)));
                    }
                }
            }
        }
        let mut rotd00 = f64::INFINITY;
        let mut at_00 = 0.0f64;
        for &angle in &candidates {
            let peak = support(x, y, angle);
            if peak < rotd00 || (peak == rotd00 && angle < at_00) {
                rotd00 = peak;
                at_00 = angle;
            }
        }
        if !rotd00.is_finite() {
            // The record never leaves the origin, so every direction is a
            // minimum and the candidate list came out empty.
            rotd00 = 0.0;
            at_00 = fold_degrees((at_100 + 90.0).to_radians());
        }

        Extremes {
            rotd00,
            at_00,
            rotd100,
            at_100,
        }
    }

    /// Assert that each reported orientation locates its own statistic, and
    /// that the two exact extremes bracket the grid the median comes off.
    fn assert_statistics_locate_themselves(x: ArrayView1<f64>, y: ArrayView1<f64>, case: &str) {
        let peaks = brute_peaks(x, y);
        let found = extremes(x, y);
        let [rotd00, rotd50, rotd100, at_00, at_50, at_100] = rotd_stats(peaks, found);
        let scale = rotd100.max(1.0);

        for (angle, statistic) in [(at_00, "RotD00"), (at_50, "RotD50"), (at_100, "RotD100")] {
            assert!(
                (0.0..N_ANGLES as f64).contains(&angle),
                "{case}: {statistic} orientation {angle} is outside 0..180"
            );
        }
        assert!(
            at_50.fract() == 0.0,
            "{case}: RotD50 orientation {at_50} is not a whole degree"
        );

        // Each exact extreme is the rotated peak at the angle reported with
        // it, which is what makes that angle a certificate rather than a
        // label.
        for (angle, value, statistic) in [(at_00, rotd00, "RotD00"), (at_100, rotd100, "RotD100")] {
            let peak = support(x, y, angle);
            assert!(
                (peak - value).abs() <= TOL * scale,
                "{case}: {statistic} {value} is not the peak {peak} at {angle} degrees"
            );
        }

        // Exact against sampled: the grid can only bracket the extremes, and
        // never reach them.
        let grid_min = peaks.iter().copied().fold(f64::INFINITY, f64::min);
        let grid_max = peaks.iter().copied().fold(0.0f64, f64::max);
        assert!(
            rotd00 <= grid_min + TOL * scale,
            "{case}: exact RotD00 {rotd00} exceeds the grid minimum {grid_min}"
        );
        assert!(
            rotd100 >= grid_max - TOL * scale,
            "{case}: exact RotD100 {rotd100} falls below the grid maximum {grid_max}"
        );
        // And no further than half a degree of turn away from it. The support
        // function moves by at most RotD100 * |u - u'| between two directions,
        // and no grid angle is more than half a degree from either extreme.
        let half_step = (0.5 * DEGREES).sin() * rotd100;
        assert!(
            grid_min - rotd00 <= half_step + TOL * scale,
            "{case}: exact RotD00 {rotd00} is further than half a degree below the grid minimum {grid_min}"
        );
        assert!(
            rotd100 - grid_max <= half_step + TOL * scale,
            "{case}: exact RotD100 {rotd100} is further than half a degree above the grid maximum {grid_max}"
        );

        // The median falls between the two central peaks, so its angle ranks
        // ninetieth of 180: fewer than 90 angles peak below it, and at least
        // 90 peak at or below it.
        let median_peak = peaks[at_50 as usize];
        let below = peaks.iter().filter(|&&peak| peak < median_peak).count();
        let at_or_below = peaks.iter().filter(|&&peak| peak <= median_peak).count();
        assert!(
            below <= 89 && at_or_below >= 90,
            "{case}: RotD50 orientation {at_50} is not the lower median:              {below} angles below it, {at_or_below} at or below"
        );
        assert!(
            rotd00 <= rotd50 && rotd50 <= rotd100,
            "{case}: RotD00 <= RotD50 <= RotD100 violated: {rotd00}, {rotd50}, {rotd100}"
        );
        // The lowest angle of a tie, for the one statistic still on the grid.
        let first = peaks.iter().position(|&peak| peak == median_peak).unwrap();
        assert_eq!(
            at_50 as usize, first,
            "{case}: {median_peak} is first attained at {first} degrees, not {at_50}"
        );
    }

    /// Assert the RotD100 / RotD50 bound for one pair of components, returning
    /// the ratio. See [`ratio_bound`] for where the bound comes from.
    fn assert_ratio_bounded(comp_0: ArrayView1<f64>, comp_90: ArrayView1<f64>, case: &str) -> f64 {
        let found = extremes(comp_0, comp_90);
        let [rotd00, rotd50, rotd100, ..] = rotd_stats(brute_peaks(comp_0, comp_90), found);
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
        let bound = ratio_bound();
        assert!(
            ratio <= bound * (1.0 + TOL),
            "{case}: RotD100 / RotD50 = {ratio} exceeds {bound}"
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
    /// invariant, so amplitude isn't worth exploring here. The fixed tests
    /// cover the extremes of the floating point range instead. This strategy
    /// leaves out the all-zero record, which has no polarisation direction;
    /// `test_ratio_bound_degenerate_records` covers that one.
    fn arb_waveform() -> impl Strategy<Value = Array1<f64>> {
        prop::collection::vec(-1.0f64..1.0, 1..=MAX_NT)
            .prop_filter("waveform is identically zero", |samples| {
                samples.iter().any(|&sample| sample != 0.0)
            })
            .prop_map(Array1::from_vec)
    }

    /// A linearly polarised record at an arbitrary angle: the family that
    /// attains the bound. The angle comes back with it, because the ratio the
    /// record attains depends on how far that angle sits off the grid.
    fn arb_polarised_components() -> impl Strategy<Value = (Array1<f64>, Array1<f64>, f64)> {
        (arb_waveform(), 0.0f64..TAU).prop_map(|(waveform, angle)| {
            let (comp_0, comp_90) = polarised(&waveform, angle);
            (comp_0, comp_90, angle)
        })
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
        arb_record_of(MAX_NT)
    }

    /// [`arb_record`] capped at `max_nt` timesteps.
    fn arb_record_of(max_nt: usize) -> impl Strategy<Value = (Array1<f64>, Array1<f64>)> {
        let samples = prop::collection::vec((-1.0f64..1.0, -1.0f64..1.0, -1.0f64..1.0), 1..=max_nt);
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
        fn prop_polarised_records_attain_the_bound(
            (comp_0, comp_90, angle) in arb_polarised_components()
        ) {
            // Linearly polarised records don't merely respect the bound, they
            // reach it exactly. Their peak in every direction is
            // |cos(theta - angle)| of the RotD100 peak, so the ratio depends
            // on the 1 degree sampling alone. RotD100 is the peak itself and
            // the two central grid peaks are cos((45 -+ beta) deg) of it, which
            // puts the ratio at sqrt(2) / cos(beta) whatever the waveform.
            let offset = angle.to_degrees().rem_euclid(1.0);
            let beta = offset.min(1.0 - offset);
            let expected = SQRT_2 / (beta * DEGREES).cos();
            let ratio = assert_ratio_bounded(comp_0.view(), comp_90.view(), "polarised record");
            prop_assert!(
                (ratio - expected).abs() <= TOL * expected,
                "expected a polarised record {beta} degrees off the grid to attain \
                 {expected}, found {ratio}"
            );
            prop_assert!(
                ratio >= SQRT_2 * (1.0 - TOL) && ratio <= ratio_bound() * (1.0 + TOL),
                "ratio {ratio} outside [sqrt(2), sqrt(2) / cos(0.5 deg)]"
            );
        }

        #[test]
        fn prop_exact_extremes_bracket_the_grid((comp_0, comp_90) in arb_record()) {
            // Whatever the record, each statistic's orientation must point at
            // the angle it came from. Both exact extremes must fall outside
            // the grid they replace, by no more than half a degree of turn.
            assert_statistics_locate_themselves(comp_0.view(), comp_90.view(), "generated record");
        }

        #[test]
        fn prop_batch_api_agrees_and_is_bounded((comp_0, comp_90) in arb_batch()) {
            let stats = rotd(comp_0.view(), comp_90.view());
            let peaks = rotd180_rows(comp_0.view(), comp_90.view());

            for (i, row) in stats.rows().into_iter().enumerate() {
                // Both entry points reduce the same hull, so the (ns, 3)
                // statistics must be exactly the reduction of the (ns, 180)
                // curve, with no tolerance needed to tie them together.
                let row_peaks: [f64; N_ANGLES] = std::array::from_fn(|theta| peaks[(i, theta)]);
                let curve_stats =
                    rotd_stats(row_peaks, extremes(comp_0.row(i), comp_90.row(i)));
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
                let expected = rotd_stats(
                    brute_peaks(comp_0.row(i), comp_90.row(i)),
                    extremes(comp_0.row(i), comp_90.row(i)),
                );
                for (stat, (&got, &want)) in row.iter().zip(expected.iter()).take(3).enumerate() {
                    prop_assert!(
                        (got - want).abs() <= TOL * want.max(1.0),
                        "row {} stat {}: hull {} != brute {}", i, stat, got, want
                    );
                }
                assert_ratio_bounded(comp_0.row(i), comp_90.row(i), &format!("row {i}"));
            }
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

    proptest! {
        // Fewer cases than the preceding block, because `brute_extremes` is
        // cubic. Short records keep the run under a second.
        #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

        #[test]
        fn prop_calipers_match_an_independent_search(
            (comp_0, comp_90) in arb_record_of(MAX_BRUTE_NT)
        ) {
            // The rotating-calipers minimum, against a search that knows
            // nothing about hulls: every direction where two timesteps could
            // swap the peak between them.
            let found = extremes(comp_0.view(), comp_90.view());
            let want = brute_extremes(comp_0.view(), comp_90.view());
            let scale = want.rotd100.max(1.0);
            prop_assert!(
                (found.rotd100 - want.rotd100).abs() <= TOL * scale,
                "RotD100: calipers {} != search {}", found.rotd100, want.rotd100
            );
            prop_assert!(
                (found.rotd00 - want.rotd00).abs() <= TOL * scale,
                "RotD00: calipers {} != search {}", found.rotd00, want.rotd00
            );
            // An orientation is a certificate rather than a label, and a tie
            // at either extreme can settle on either of two equal directions.
            // So what has to agree is the peak each angle points at.
            for (angle, value, statistic) in [
                (found.at_00, found.rotd00, "RotD00"),
                (found.at_100, found.rotd100, "RotD100"),
            ] {
                let peak = support(comp_0.view(), comp_90.view(), angle);
                prop_assert!(
                    (peak - value).abs() <= TOL * scale,
                    "{} {} is not the peak {} at {} degrees", statistic, value, peak, angle
                );
            }
        }
    }

    #[test]
    fn calipers_match_an_analytic_minimum() {
        // Two timesteps, at (1, 0) and (0, 2). The rotated peak is
        // max(|cos theta|, 2 |sin theta|), least where the two are equal, at
        // tan theta = 1/2, and worth cos(atan(1/2)) = 2 / sqrt(5) there.
        //
        // The origin-symmetric hull is the parallelogram through (1, 0),
        // (0, 2), (-1, 0), (0, -2), whose edges all lie 2 / sqrt(5) from the
        // origin, so the calipers have to find the same number off the
        // geometry that calculus finds off the function.
        let found = extremes(array![1.0, 0.0].view(), array![0.0, 2.0].view());
        let expected_rotd00 = 2.0 / 5.0f64.sqrt();
        let expected_at_00 = 0.5f64.atan().to_degrees();
        assert!(
            (found.rotd00 - expected_rotd00).abs() < 1e-15,
            "RotD00 {} is not 2 / sqrt(5) = {expected_rotd00}",
            found.rotd00
        );
        assert!(
            (found.at_00 - expected_at_00).abs() < 1e-12,
            "RotD00 orientation {} is not atan(1/2) = {expected_at_00} degrees",
            found.at_00
        );
        // The furthest timestep is (0, 2), straight up the 090 component.
        assert_eq!(found.rotd100, 2.0);
        assert_eq!(found.at_100, 90.0);
    }

    #[test]
    fn calipers_match_a_thousandth_of_a_degree_sweep() {
        // The calipers claim the exact minimum of a function the 1 degree grid
        // only samples. A sweep a thousand times finer than that grid has to
        // come down to it and must never undercut it.
        let nt = 401;
        let comp_0 = Array1::from_shape_fn(nt, |i| {
            let t = i as f64;
            (0.23 * t).sin() + 0.4 * (0.07 * t).cos()
        });
        let comp_90 = Array1::from_shape_fn(nt, |i| {
            let t = i as f64;
            0.6 * (0.19 * t).cos() - 0.25 * (0.03 * t).sin()
        });
        let found = extremes(comp_0.view(), comp_90.view());

        let steps = 180_000;
        let mut fine_min = f64::INFINITY;
        for step in 0..steps {
            let angle = step as f64 * 180.0 / steps as f64;
            fine_min = fine_min.min(support(comp_0.view(), comp_90.view(), angle));
        }
        assert!(
            found.rotd00 <= fine_min + 1e-12,
            "RotD00 {} undercuts a 0.001 degree sweep at {fine_min}",
            found.rotd00
        );
        // Half a step of that sweep is worth RotD100 * sin(0.0005 deg).
        let half_step = (0.0005 * DEGREES).sin() * found.rotd100;
        assert!(
            fine_min - found.rotd00 <= half_step + 1e-12,
            "RotD00 {} sits {} below a 0.001 degree sweep, more than the {half_step} that sweep can miss by",
            found.rotd00,
            fine_min - found.rotd00
        );
    }

    #[test]
    fn calipers_zero_out_on_a_polarised_record() {
        // A trajectory confined to a line through the origin has a direction
        // it never moves in at all, and the calipers report an exact zero
        // where the grid reports 6e-17.
        for polarisation in [0.0f64, 30.0, 45.0, 100.5, 179.0] {
            let motion = Array1::from_shape_fn(64, |i| (0.3 * i as f64).sin());
            let (comp_0, comp_90) = polarised(&motion, polarisation.to_radians());
            let found = extremes(comp_0.view(), comp_90.view());
            let peak = motion.iter().fold(0.0f64, |m, &u| m.max(u.abs()));

            // Zero to rounding rather than exactly zero, because the sine and
            // cosine of a whole number of degrees aren't exactly orthogonal in
            // binary. The record follows that line to within an ulp.
            assert!(
                found.rotd00 <= 1e-15 * peak,
                "polarisation {polarisation}: RotD00 {} is not zero to rounding",
                found.rotd00
            );
            let across = (polarisation + 90.0).rem_euclid(180.0);
            assert!(
                (found.at_00 - across).abs() < 1e-6,
                "polarisation {polarisation}: RotD00 orientation {} is not {across}",
                found.at_00
            );
            assert!(
                (found.rotd100 - peak).abs() <= 1e-12 * peak,
                "polarisation {polarisation}: RotD100 {} is not the peak {peak}",
                found.rotd100
            );

            // And what the grid reported instead. The nearest whole degree to
            // the direction of no motion is `beta` off it, and the peak there
            // is RotD100 sin(beta): nothing at all for a polarisation on the
            // grid, 0.87% of RotD100 for one half a degree off it.
            let grid_min = brute_peaks(comp_0.view(), comp_90.view())
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let offset = polarisation.rem_euclid(1.0);
            let beta = offset.min(1.0 - offset);
            assert!(
                (grid_min - peak * (beta * DEGREES).sin()).abs() <= 1e-12 * peak,
                "polarisation {polarisation}: grid minimum {grid_min} is not RotD100 sin({beta} deg)"
            );
        }
    }

    #[test]
    fn extremes_handle_degenerate_records() {
        // A record that never moves, and one with one timestep only. Neither
        // has an origin-symmetric hull with any area to run calipers over.
        let zeros = Array1::<f64>::zeros(16);
        let still = extremes(zeros.view(), zeros.view());
        assert_eq!([still.rotd00, still.rotd100], [0.0, 0.0]);

        let one = extremes(array![3.0].view(), array![4.0].view());
        assert_eq!(one.rotd00, 0.0, "one timestep leaves a direction unmoved");
        assert_eq!(one.rotd100, 5.0);
        assert!((one.at_100 - (4.0f64 / 3.0).atan().to_degrees()).abs() < 1e-12);

        // Collinear but off the origin. The symmetric hull is a parallelogram
        // with area, so the calipers run over it, and the minimum comes out as
        // the distance to the line the record follows.
        let off_origin = extremes(
            Array1::from_elem(8, 2.0).view(),
            Array1::from_shape_fn(8, |i| i as f64 - 4.0).view(),
        );
        assert!(
            (off_origin.rotd00 - 2.0).abs() < 1e-12,
            "a record on the line x = 2 should have RotD00 2, found {}",
            off_origin.rotd00
        );
        assert!((off_origin.at_00 - 0.0).abs() < 1e-12);
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
        let mut reference = want;
        reference.sort_unstable_by(f64::total_cmp);
        let median = (reference[89] + reference[90]) / 2.0;
        assert!(((sorted[89] + sorted[90]) / 2.0 - median).abs() <= TOL * median.max(1.0));

        // The extremes no longer come off this curve, so what the curve pins
        // is that they bracket it.
        let found = extremes(comp_0.view(), comp_90.view());
        assert!(
            found.rotd00 <= sorted[0] && sorted[179] <= found.rotd100,
            "exact extremes {} and {} do not bracket the sweep {} to {}",
            found.rotd00,
            found.rotd100,
            sorted[0],
            sorted[179]
        );
    }

    #[test]
    fn rotd180_matches_brute_on_circular_record() {
        // A near-circular trajectory is the culling's worst case: almost every
        // point lies near the hull, so the cull removes few of them. The result
        // must still be exact.
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
        let (sweep, found) = Hull::default().analyse(comp_0.view(), comp_90.view());
        let [min, median, max, at_min, at_median, at_max] = rotd_stats(sweep, found);
        let expected_min = 2.0f64.sqrt() / 2.0; // such as at pi / 4 degrees
        let expected_max = 1.0; // such as at 0 degrees
        let expected_median = 0.9238443540096138; // derived independently with numpy
        // The sweep is max(|cos theta|, |sin theta|): least at 45 degrees, and
        // 1 at both 0 and 90 degrees, of which rotd_stats takes the lower. The
        // median falls between the peaks at 157 and 158 degrees.
        assert_eq!(
            [at_min, at_median, at_max],
            [45.0, 157.0, 0.0],
            "Orientations wrong: found {at_min}, {at_median}, {at_max} degrees"
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
    fn rotd180_rows_and_rotd_share_a_hull() {
        // The curve entry point and the statistics entry point run over the
        // same records, so the statistics must be the reduction of the curve
        // beside them, with no tolerance needed to tie the two together.
        let nt = 512;
        let comp_0 = Array2::from_shape_fn((3, nt), |(s, i)| {
            let t = i as f64;
            (0.03 * t + s as f64).sin() * (1.0 + s as f64)
        });
        let comp_90 = Array2::from_shape_fn((3, nt), |(s, i)| {
            let t = i as f64;
            (0.05 * t).cos() - 0.3 * (0.11 * t + s as f64).sin()
        });
        let curve = rotd180_rows(comp_0.view(), comp_90.view());
        let stats = rotd(comp_0.view(), comp_90.view());
        for s in 0..3 {
            let row = curve.row(s);
            let reduced = rotd_stats(
                std::array::from_fn(|theta| row[theta]),
                extremes(comp_0.row(s), comp_90.row(s)),
            );
            assert_eq!(stats.row(s), ArrayView1::from(&reduced), "row {s}");
        }
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
