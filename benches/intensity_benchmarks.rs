use _core::{arias_intensity, cav, psa};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ndarray::array;
use std::hint::black_box;

#[path = "common.rs"]
mod common;
use common::generate_waveforms;

// Configuration constants for test scenarios
const SAMPLING_RATE: f64 = 0.005; // 200 Hz
const STATION_COUNTS: &[usize] = &[1, 10, 100, 1000, 10000];
const SAMPLE_LENGTHS: &[usize] = &[
    1000,   // 5 seconds
    5_000,  // 25 seconds
    10_000, // 50 seconds
    20_000, // 100 seconds
    40_000, // 200 seconds
];
// Oscillator coefficients for T = 1 s, 5% damping at SAMPLING_RATE, from
// `IM.ims._nigam_jennings_coefficients(np.array([1.0]), 0.005)`. The cost of
// the recurrence does not depend on their values.
const PSA_COEFFICIENTS: [f64; psa::N_COEFFICIENTS] = [
    0.9995070766804318,
    0.19704993255509468,
    -0.004991333100781255,
    0.996370929600225,
    -0.00032856713331003905,
    -0.0024941543916031455,
    -0.00016435618625817934,
    -0.00249717870917811,
];

/// Benchmark CAV (Cumulative Absolute Velocity) calculations
fn bench_cav(c: &mut Criterion) {
    let mut group = c.benchmark_group("CAV");

    for &stations in STATION_COUNTS {
        for &samples in SAMPLE_LENGTHS {
            let waveforms = generate_waveforms(stations, samples);
            let view = waveforms.view();
            let param = format!("{}stn_{}smp", stations, samples);

            group.throughput(Throughput::Bytes((stations * samples * 8) as u64));

            group.bench_with_input(BenchmarkId::new("Sequential", &param), &view, |b, &v| {
                b.iter(|| cav::cav(black_box(v), black_box(SAMPLING_RATE)))
            });
        }
    }

    group.finish();
}

/// Benchmark Arias Intensity calculations
fn bench_arias_intensity(c: &mut Criterion) {
    let mut group = c.benchmark_group("Arias_Intensity");

    for &stations in STATION_COUNTS {
        for &samples in SAMPLE_LENGTHS {
            let waveforms = generate_waveforms(stations, samples);
            let view = waveforms.view();
            let param = format!("{}stn_{}smp", stations, samples);

            group.throughput(Throughput::Bytes((stations * samples * 8) as u64));

            group.bench_with_input(BenchmarkId::new("Sequential", &param), &view, |b, &v| {
                b.iter(|| arias_intensity::arias_intensity(black_box(v), black_box(SAMPLING_RATE)))
            });
        }
    }

    group.finish();
}

/// Benchmark Cumulative Arias Intensity calculations
fn bench_cumulative_arias(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cumulative_Arias");

    for &stations in STATION_COUNTS {
        for &samples in SAMPLE_LENGTHS {
            let waveforms = generate_waveforms(stations, samples);
            let view = waveforms.view();
            let param = format!("{}stn_{}smp", stations, samples);

            // Cumulative version produces more output data
            group.throughput(Throughput::Bytes((stations * samples * 8) as u64));

            group.bench_with_input(BenchmarkId::new("Sequential", &param), &view, |b, &v| {
                b.iter(|| {
                    arias_intensity::cumulative_arias_intensity(
                        black_box(v),
                        black_box(SAMPLING_RATE),
                    )
                })
            });
        }
    }

    group.finish();
}

/// Benchmark Significant Duration calculations.
///
/// This measures the full Ds pipeline as the Python binding
/// (`_significant_duration` in `src-rust/lib.rs`) uses it: integrating raw
/// waveforms to cumulative Arias intensity, then binary-searching that for
/// the threshold crossings. `significant_duration` itself requires a
/// non-decreasing cumulative intensity array, so the integration step can't
/// be skipped without benchmarking an unrepresentative, degenerate input.
fn bench_significant_duration(c: &mut Criterion) {
    let mut group = c.benchmark_group("Significant_Duration_Full_Pipeline");

    for &stations in STATION_COUNTS {
        for &samples in SAMPLE_LENGTHS {
            let waveforms = generate_waveforms(stations, samples);
            let view = waveforms.view();
            let param = format!("{}stn_{}smp", stations, samples);

            group.throughput(Throughput::Bytes((stations * samples * 8) as u64));

            group.bench_with_input(BenchmarkId::new("Sequential", &param), &view, |b, &v| {
                b.iter(|| {
                    common::significant_duration_from_waveforms(
                        black_box(v),
                        black_box(SAMPLING_RATE),
                        0.05,
                        0.95,
                    )
                })
            });
        }
    }

    group.finish();
}

/// Benchmark Pseudo-Spectral Acceleration (pSA) calculations
/// This is typically the most expensive calculation
fn bench_psa(c: &mut Criterion) {
    let mut group = c.benchmark_group("PSA");
    // PSA is expensive, so we might want to use a smaller sample size
    group.sample_size(10);

    let coefficients = array![PSA_COEFFICIENTS];
    for &stations in STATION_COUNTS {
        // For PSA, use smaller sample sets to keep benchmark times reasonable
        for &samples in SAMPLE_LENGTHS {
            let waveforms = generate_waveforms(stations, samples);
            let view = waveforms.view();
            let param = format!("T1.0s_{}stn_{}smp", stations, samples);

            group.throughput(Throughput::Bytes((stations * samples * 8) as u64));

            group.bench_with_input(BenchmarkId::new("Sequential", &param), &view, |b, &v| {
                b.iter(|| {
                    psa::psa(
                        black_box(&v),
                        black_box(&v),
                        black_box(&v),
                        black_box(&coefficients.view()),
                    )
                })
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cav,
    bench_arias_intensity,
    bench_cumulative_arias,
    bench_significant_duration,
    bench_psa,
);
criterion_main!(benches);
