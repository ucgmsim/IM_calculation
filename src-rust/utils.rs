//! Parallel processing utilities for 2D arrays.
//!
//! This module provides generic abstractions for performing row-wise operations
//! in parallel using `ndarray` and `rayon`.

use ndarray::parallel::prelude::*;
use ndarray::prelude::*;

/// Processes each row of a 2D array in parallel and reduces each row to a single value.
///
/// This is ideal for operations like row-wise sums, means, or integrations (e.g., trapz).
///
/// # Arguments
/// * `matrix` - An `ArrayView2` of the data to process.
/// * `f` - A function or closure that takes an `ArrayView1<T>` (a row) and returns a single value `U`.
///
/// # Returns
/// An `Array1<U>` containing one result per row.
///
pub fn parallel_reduce_rows<T, U, F>(matrix: ArrayView2<T>, f: F) -> Array1<U>
where
    T: Sync + Send,
    U: Send + Default,
    F: Fn(ArrayView1<T>) -> U + Sync + Send,
{
    let (n_rows, _) = matrix.dim();
    let mut results = Vec::with_capacity(n_rows);

    matrix
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(f)
        .collect_into_vec(&mut results);

    Array1::from_vec(results)
}

/// Transforms each row of a 2D array in parallel, producing a new 2D array.
///
/// This function maps a row of length $N$ to a new row of length $M$, maintaining
/// the overall structure of the matrix.
///
/// # Arguments
/// * `input` - An `ArrayView2` to be transformed.
/// * `f` - A function or closure that takes an `ArrayView1<T>` and returns an `Array1<U>`.
///
/// # Returns
/// A new `Array2<U>` with the same number of rows as the input.
///
pub fn parallel_map_rows<T, F>(input: ArrayView2<T>, f: F) -> Array2<T>
where
    T: Sync + Send + Default,
    F: Fn(ArrayView1<T>, ArrayViewMut1<T>) + Sync + Send,
{
    let mut out = Array2::<T>::default(input.dim());

    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(input.axis_iter(Axis(0)).into_par_iter())
        .for_each(|(out_row, in_row)| {
            f(in_row, out_row);
        });

    out
}
