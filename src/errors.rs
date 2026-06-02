//! Cross-module error helpers for converting Rust errors into Python exceptions.

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::PyErr;

/// Convert any [`std::io::Error`] into a Python `IOError`, preserving the message.
pub fn io_to_py(err: std::io::Error) -> PyErr {
    PyIOError::new_err(err.to_string())
}

/// Convert any [`anyhow::Error`] into a Python `RuntimeError`.
pub fn anyhow_to_py(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{err:#}"))
}

/// Build a Python `ValueError` from a static or owned message.
pub fn value_error<S: Into<String>>(msg: S) -> PyErr {
    PyValueError::new_err(msg.into())
}
