use _utils::{arias_intensity, cav, psa, significant_duration};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array2;
use std::hint::black_box;

// Configuration constants for test scenarios
const SAMPLING_RATE: f64 = 0.005; // 200 Hz
const STATION_COUNTS: &[usize] = &[1, 5, 10, 25, 50, 100];
const SAMPLE_LENGTHS: &[usize] = &[
    1_000,  // 5 seconds
    5_000,  // 25 seconds
    10_000, // 50 seconds
    20_000, // 100 seconds (standard)
    40_000, // 200 seconds
];
const PSA_PERIODS: &[f64] = &[0.1, 0.5, 1.0, 2.0];
const DAMPING: f64 = 0.05;

/// Generate synthetic waveform data for benchmarking
fn generate_waveforms(stations: usize, samples: usize) -> Array2<f64> {
    // Using a simple sine wave with some noise for realistic computation
    Array2::from_shape_fn((stations, samples), |(i, j)| {
        0.5 * ((j as f64 * 0.01 + i as f64).sin() + 0.1 * (j as f64 * 0.1).cos())
    })
}

/// Benchmark CAV (Cumulative Absolute Velocity) calculations
fn bench_cav(c: &mut Criterion) {
    let mut group = c.benchmark_group("CAV");

    for &stations in STATION_COUNTS {
        for &samples in SAMPLE_LENGTHS {
            let waveforms = generate_waveforms(stations, samples);
            let view = waveforms.view();
            let param = format!("{}stn_{}smp", stations, samples);

            // Set throughput for better comparison (stations * samples * sizeof(f64))
            group.throughput(Throughput::Bytes((stations * samples * 8) as u64));

            group.bench_with_input(BenchmarkId::new("Sequential", &param), &view, |b, &v| {
                b.iter(|| cav::cav(black_box(v), black_box(SAMPLING_RATE)))
            });

            // Only benchmark parallel for multiple stations
            if stations > 1 {
                group.bench_with_input(BenchmarkId::new("Parallel", &param), &view, |b, &v| {
                    b.iter(|| cav::parallel_cav(black_box(v), black_box(SAMPLING_RATE)))
                });
            }
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

            if stations > 1 {
                group.bench_with_input(BenchmarkId::new("Parallel", &param), &view, |b, &v| {
                    b.iter(|| {
                        arias_intensity::parallel_arias_intensity(
                            black_box(v),
                            black_box(SAMPLING_RATE),
                        )
                    })
                });
            }
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

            if stations > 1 {
                group.bench_with_input(BenchmarkId::new("Parallel", &param), &view, |b, &v| {
                    b.iter(|| {
                        arias_intensity::parallel_cumulative_arias_intensity(
                            black_box(v),
                            black_box(SAMPLING_RATE),
                        )
                    })
                });
            }
        }
    }

    group.finish();
}

/// Benchmark Significant Duration calculations
fn bench_significant_duration(c: &mut Criterion) {
    let mut group = c.benchmark_group("Significant_Duration");

    for &stations in STATION_COUNTS {
        for &samples in SAMPLE_LENGTHS {
            let waveforms = generate_waveforms(stations, samples);
            let view = waveforms.view();
            let param = format!("{}stn_{}smp", stations, samples);

            group.throughput(Throughput::Bytes((stations * samples * 8) as u64));

            group.bench_with_input(BenchmarkId::new("Sequential", &param), &view, |b, &v| {
                b.iter(|| {
                    significant_duration::significant_duration(
                        black_box(v),
                        black_box(SAMPLING_RATE),
                        0.05,
                        0.95,
                    )
                })
            });

            if stations > 1 {
                group.bench_with_input(BenchmarkId::new("Parallel", &param), &view, |b, &v| {
                    b.iter(|| {
                        significant_duration::parallel_significant_duration(
                            black_box(v),
                            black_box(SAMPLING_RATE),
                            0.05,
                            0.95,
                        )
                    })
                });
            }
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

    for &period in PSA_PERIODS {
        for &stations in STATION_COUNTS {
            // For PSA, use smaller sample sets to keep benchmark times reasonable
            let sample_subset = if stations >= 50 {
                &SAMPLE_LENGTHS[..3] // Only test shorter durations for many stations
            } else {
                SAMPLE_LENGTHS
            };

            for &samples in sample_subset {
                let waveforms = generate_waveforms(stations, samples);
                let view = waveforms.view();
                let param = format!("T{:.1}s_{}stn_{}smp", period, stations, samples);

                group.throughput(Throughput::Bytes((stations * samples * 8) as u64));

                // Note: Only parallel version exists in your original code
                group.bench_with_input(BenchmarkId::new("Parallel", &param), &view, |b, &v| {
                    b.iter(|| {
                        psa::newmark_beta_method_parallel(
                            black_box(&v),
                            black_box(SAMPLING_RATE),
                            black_box(period),
                            black_box(DAMPING),
                        )
                    })
                });
            }
        }
    }

    group.finish();
}

/// Benchmark to compare all intensity measures at a fixed configuration
/// Useful for understanding relative computational costs
fn bench_comparison_fixed_config(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comparison_50stations_20k_samples");

    let stations = 50;
    let samples = 20_000;
    let waveforms = generate_waveforms(stations, samples);
    let view = waveforms.view();

    group.throughput(Throughput::Bytes((stations * samples * 8) as u64));

    group.bench_function("CAV_Sequential", |b| {
        b.iter(|| cav::cav(black_box(view), black_box(SAMPLING_RATE)))
    });

    group.bench_function("CAV_Parallel", |b| {
        b.iter(|| cav::parallel_cav(black_box(view), black_box(SAMPLING_RATE)))
    });

    group.bench_function("Arias_Sequential", |b| {
        b.iter(|| arias_intensity::arias_intensity(black_box(view), black_box(SAMPLING_RATE)))
    });

    group.bench_function("Arias_Parallel", |b| {
        b.iter(|| {
            arias_intensity::parallel_arias_intensity(black_box(view), black_box(SAMPLING_RATE))
        })
    });

    group.bench_function("CumulativeArias_Sequential", |b| {
        b.iter(|| {
            arias_intensity::cumulative_arias_intensity(black_box(view), black_box(SAMPLING_RATE))
        })
    });

    group.bench_function("CumulativeArias_Parallel", |b| {
        b.iter(|| {
            arias_intensity::parallel_cumulative_arias_intensity(
                black_box(view),
                black_box(SAMPLING_RATE),
            )
        })
    });

    group.bench_function("SigDuration_Sequential", |b| {
        b.iter(|| {
            significant_duration::significant_duration(
                black_box(view),
                black_box(SAMPLING_RATE),
                0.05,
                0.95,
            )
        })
    });

    group.bench_function("SigDuration_Parallel", |b| {
        b.iter(|| {
            significant_duration::parallel_significant_duration(
                black_box(view),
                black_box(SAMPLING_RATE),
                0.05,
                0.95,
            )
        })
    });

    group.bench_function("PSA_T1.0_Parallel", |b| {
        b.iter(|| {
            psa::newmark_beta_method_parallel(
                black_box(&view),
                black_box(SAMPLING_RATE),
                1.0,
                DAMPING,
            )
        })
    });

    group.finish();
}

/// Benchmark to identify parallel crossover point for each intensity measure
/// Tests a focused range around expected crossover
fn bench_parallel_crossover(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel_Crossover_Analysis");

    // Fine-grained station counts near expected crossover
    let crossover_stations = &[1, 2, 3, 4, 5, 7, 10, 15, 20];
    let samples = 20_000; // Standard record length

    for &stations in crossover_stations {
        let waveforms = generate_waveforms(stations, samples);
        let view = waveforms.view();

        // CAV crossover
        group.bench_with_input(
            BenchmarkId::new("CAV_Sequential", stations),
            &view,
            |b, &v| b.iter(|| cav::cav(black_box(v), black_box(SAMPLING_RATE))),
        );

        if stations > 1 {
            group.bench_with_input(
                BenchmarkId::new("CAV_Parallel", stations),
                &view,
                |b, &v| b.iter(|| cav::parallel_cav(black_box(v), black_box(SAMPLING_RATE))),
            );
        }

        // Arias crossover
        group.bench_with_input(
            BenchmarkId::new("Arias_Sequential", stations),
            &view,
            |b, &v| {
                b.iter(|| arias_intensity::arias_intensity(black_box(v), black_box(SAMPLING_RATE)))
            },
        );

        if stations > 1 {
            group.bench_with_input(
                BenchmarkId::new("Arias_Parallel", stations),
                &view,
                |b, &v| {
                    b.iter(|| {
                        arias_intensity::parallel_arias_intensity(
                            black_box(v),
                            black_box(SAMPLING_RATE),
                        )
                    })
                },
            );
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
    bench_comparison_fixed_config,
    bench_parallel_crossover
);
criterion_main!(benches);
