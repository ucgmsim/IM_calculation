use crate::utils::parallel_reduce_rows;
use ndarray::prelude::*;

fn threshold_search(normalised_intensities: ArrayView1<f64>, dt: f64, low: f64, high: f64) -> f64 {
    if let Some(&max) = normalised_intensities.last()
        && max >= 1e-10
    {
        let intensities = normalised_intensities.as_slice().unwrap();
        let low_idx = intensities
            .binary_search_by(|x| (x / max).total_cmp(&low))
            .unwrap_or_else(|i| i);

        let dx = intensities[low_idx..]
            .binary_search_by(|x| (x / max).total_cmp(&high))
            .unwrap_or_else(|i| i);
        (dx as f64) * dt
    } else {
        0.0
    }
}

pub fn significant_duration(
    arias_intensity: ArrayView2<f64>,
    dt: f64,
    low: f64,
    high: f64,
) -> Array1<f64> {
    // Binary search for values above threshold
    arias_intensity.map_axis(Axis(1), |normalised_intensity| {
        threshold_search(normalised_intensity, dt, low, high)
    })
}

pub fn parallel_significant_duration(
    arias_intensity: ArrayView2<f64>,
    dt: f64,
    low: f64,
    high: f64,
) -> Array1<f64> {
    // Normalise arias intensity
    // NOTE: this is subtly different to parallel_map because it does
    // not create a copy of the array for output. It simply updates in-place.
    parallel_reduce_rows(arias_intensity.view(), |normalised_intensity| {
        threshold_search(normalised_intensity, dt, low, high)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_significant_duration_simple() {
        // Create a linear ramp from 0 to 10
        // Normalised, this becomes 0.0 to 1.0
        // dt = 1.0. Indices are 0, 1, 2, ... 10
        let arias = array![[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let dt = 1.0;

        // D_20-80: should find index 2 (val 0.2) and index 8 (val 0.8)
        // Duration = (8 - 2) * 1.0 = 6.0
        let result = significant_duration(arias.view(), dt, 0.2, 0.8);

        assert_abs_diff_eq!(result[0], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_zero_duration() {
        // If low and high are the same, duration should be 0
        let arias = array![[0.0, 5.0, 10.0]];
        let dt = 0.1;
        let result = significant_duration(arias.view(), dt, 0.5, 0.5);

        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn test_parallel_matches_sequential() {
        let arias = array![[0.0, 0.1, 0.5, 0.8, 1.0], [0.0, 0.4, 0.7, 0.9, 1.0]];
        let dt = 0.01;
        let low = 0.05;
        let high = 0.95;

        let seq = significant_duration(arias.view(), dt, low, high);
        let par = parallel_significant_duration(arias.view(), dt, low, high);

        assert_abs_diff_eq!(seq, par, epsilon = 1e-10);
    }

    #[test]
    fn test_output_shape() {
        let arias = Array2::<f64>::zeros((5, 100));
        let result = significant_duration(arias.view(), 0.01, 0.05, 0.95);

        assert_eq!(result.len(), 5);
    }
}
