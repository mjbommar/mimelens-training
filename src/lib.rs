//! `binary_embedding._native` — Rust hot-path for the binary-embedding-paper project.
//!
//! Three concerns:
//!
//! - [`sampling`] — read head/tail/window/strided/random byte slices from a file path.
//! - [`tokenize`] — wrap `bbpe::BinaryTokenizer` for fast encode/decode in Python.
//! - [`mlm`] — Rust-side MLM mask generation + window sampling from cached token streams.
//!
//! The Python surface is intentionally narrow: a small set of free functions plus
//! one [`tokenize::BinaryTokenizer`] class. Everything else stays in Python.

#![forbid(unsafe_code)]
#![warn(missing_docs, clippy::all, rust_2018_idioms, future_incompatible)]

use pyo3::prelude::*;

pub mod errors;
pub mod mlm;
pub mod sampling;
pub mod tokenize;

/// Build the `binary_embedding._native` Python module.
#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Sampling: free functions on file paths.
    m.add_function(wrap_pyfunction!(sampling::file_size, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::read_head, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::read_tail, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::read_window, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::read_strided_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::read_random_window, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::read_random_windows, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::read_paths_to_file_as_bytes, m)?)?;

    // Tokenization: BinaryTokenizer class wrapping bbpe.
    m.add_class::<tokenize::BinaryTokenizer>()?;
    m.add_function(wrap_pyfunction!(tokenize::tokenize_paths, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize::tokenize_paths_to_file, m)?)?;

    // MLM utilities.
    m.add_function(wrap_pyfunction!(mlm::mlm_mask, m)?)?;
    m.add_function(wrap_pyfunction!(mlm::mlm_mask_many, m)?)?;
    m.add_function(wrap_pyfunction!(mlm::sample_token_window, m)?)?;
    m.add_function(wrap_pyfunction!(mlm::sample_token_windows, m)?)?;

    let _ = py;
    Ok(())
}
