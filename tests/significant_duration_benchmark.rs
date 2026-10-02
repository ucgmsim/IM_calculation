//! Regression test for the `Significant_Duration` criterion benchmark
//! (see `benches/intensity_benchmarks.rs` / `benches/common.rs`).
//!
//! `significant_duration::significant_duration` expects a cumulative Arias
//! intensity array (non-decreasing, total in the last element), the same way
//! the Python binding `_significant_duration` (src-rust/lib.rs) feeds it. The
//! benchmark used to pass raw, sinusoidal waveforms straight in instead,
//! which are not monotonic and go negative, so `threshold_search` either took
//! its early `0.0` return or binary-searched unsorted data. This test exercises
//! the shared helper the benchmark now uses and checks it reports a sane,
//! bounded Ds duration for realistic waveform data, not the degenerate `0.0`
//! that feeding raw waveforms directly into `significant_duration` produces.

#[path = "../benches/common.rs"]
mod common;

const SAMPLING_RATE: f64 = 0.005;

#[test]
fn significant_duration_from_waveforms_matches_full_pipeline() {
    let stations = 10;
    let samples = 2000;
    let waveforms = common::generate_waveforms(stations, samples);
    let total_duration = (samples - 1) as f64 * SAMPLING_RATE;

    let ds =
        common::significant_duration_from_waveforms(waveforms.view(), SAMPLING_RATE, 0.05, 0.95);

    assert_eq!(ds.len(), stations);
    for (station, &duration) in ds.iter().enumerate() {
        assert!(
            duration > 0.0 && duration <= total_duration,
            "station {station}: expected a Ds duration in (0, {total_duration}], got {duration}; \
             feeding raw waveforms directly into significant_duration (instead of their cumulative \
             Arias intensity) collapses this to 0.0 for most stations",
        );
    }
}
