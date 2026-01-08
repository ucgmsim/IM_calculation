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
/// # Example
/// ```
/// use ndarray::prelude::*;
/// use crate::utils::parallel_reduce_rows;
///
/// let data = array![[1.0, 2.0], [3.0, 4.0]];
/// let sums = parallel_reduce_rows(data.view(), |row| row.sum());
/// assert_eq!(sums, array![3.0, 7.0]);
/// ```
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
/// # Example
/// ```
/// use ndarray::prelude::*;
/// use crate::utils::parallel_map_rows;
///
/// let data = Array2::<f64>::ones((5, 10));
/// // Multiply every element in every row by 2.0
/// // This would be better achieved by ndarray's standard mapv or the parallel mapv
/// // but this just a simple illustration.
/// let doubled = parallel_map_rows(data.view(), |row| row * 2.0);
/// ```
pub fn parallel_map_rows<T, U, F>(input: ArrayView2<T>, f: F) -> Array2<U>
where
    T: Sync + Send,
    U: Send + Sync + Default + Clone,
    F: Fn(ArrayView1<T>) -> Array1<U> + Sync + Send,
{
    let mut out = Array2::<U>::default(input.dim());

    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(input.axis_iter(Axis(0)).into_par_iter())
        .for_each(|(mut out_row, in_row)| {
            let result_row = f(in_row);
            out_row.assign(&result_row);
        });

    out
}
