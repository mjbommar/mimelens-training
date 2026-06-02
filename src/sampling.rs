//! File-level byte sampling primitives.
//!
//! All functions operate on a path and never load more bytes than asked for.
//! Files up to tens of megabytes are common in our corpora; we use seek+read
//! rather than mmap to keep behavior predictable across platforms and FUSE mounts.

use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;

use crate::errors::{anyhow_to_py, io_to_py, value_error};

fn open_for_read(path: &str) -> PyResult<File> {
    File::open(path).map_err(io_to_py)
}

fn metadata_size(path: &str) -> PyResult<u64> {
    std::fs::metadata(path).map(|m| m.len()).map_err(io_to_py)
}

fn read_n_at(file: &mut File, offset: u64, n: usize) -> PyResult<Vec<u8>> {
    file.seek(SeekFrom::Start(offset)).map_err(io_to_py)?;
    let mut buf = vec![0u8; n];
    let mut filled = 0usize;
    while filled < n {
        match file.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(k) => filled += k,
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(io_to_py(err)),
        }
    }
    buf.truncate(filled);
    Ok(buf)
}

/// Return the size of `path` in bytes.
#[pyfunction]
pub fn file_size(path: &str) -> PyResult<u64> {
    metadata_size(path)
}

/// Read the first `n` bytes of `path`. Returns fewer bytes if the file is smaller.
#[pyfunction]
pub fn read_head<'py>(py: Python<'py>, path: &str, n: u64) -> PyResult<Bound<'py, PyBytes>> {
    let n_usize: usize = n
        .try_into()
        .map_err(|_| value_error("n exceeds usize range"))?;
    let mut file = open_for_read(path)?;
    let bytes = read_n_at(&mut file, 0, n_usize)?;
    Ok(PyBytes::new_bound(py, &bytes))
}

/// Read the last `n` bytes of `path`. Returns the entire file if it is smaller than `n`.
#[pyfunction]
pub fn read_tail<'py>(py: Python<'py>, path: &str, n: u64) -> PyResult<Bound<'py, PyBytes>> {
    let total = metadata_size(path)?;
    let n_clamped = n.min(total);
    let offset = total - n_clamped;
    let n_usize: usize = n_clamped
        .try_into()
        .map_err(|_| value_error("n exceeds usize range"))?;
    let mut file = open_for_read(path)?;
    let bytes = read_n_at(&mut file, offset, n_usize)?;
    Ok(PyBytes::new_bound(py, &bytes))
}

/// Read `n` bytes starting at `offset`. Returns fewer if the file ends earlier.
#[pyfunction]
pub fn read_window<'py>(
    py: Python<'py>,
    path: &str,
    offset: u64,
    n: u64,
) -> PyResult<Bound<'py, PyBytes>> {
    let n_usize: usize = n
        .try_into()
        .map_err(|_| value_error("n exceeds usize range"))?;
    let mut file = open_for_read(path)?;
    let bytes = read_n_at(&mut file, offset, n_usize)?;
    Ok(PyBytes::new_bound(py, &bytes))
}

/// Return `n_chunks` evenly-spaced byte windows of size `chunk_size`. The first
/// chunk starts at offset 0; the last finishes at the end of the file (modulo
/// chunk size). Useful for "begin/middle/end" Magika-style sampling.
#[pyfunction]
pub fn read_strided_chunks<'py>(
    py: Python<'py>,
    path: &str,
    n_chunks: u32,
    chunk_size: u64,
) -> PyResult<Vec<Bound<'py, PyBytes>>> {
    if n_chunks == 0 {
        return Ok(Vec::new());
    }
    if chunk_size == 0 {
        return Err(value_error("chunk_size must be > 0"));
    }
    let chunk_usize: usize = chunk_size
        .try_into()
        .map_err(|_| value_error("chunk_size exceeds usize range"))?;
    let total = metadata_size(path)?;
    let mut file = open_for_read(path)?;
    let mut out = Vec::with_capacity(n_chunks as usize);

    if total <= chunk_size || n_chunks == 1 {
        let bytes = read_n_at(&mut file, 0, chunk_usize)?;
        out.push(PyBytes::new_bound(py, &bytes));
        for _ in 1..n_chunks {
            out.push(PyBytes::new_bound(py, &[]));
        }
        return Ok(out);
    }

    let last_offset = total - chunk_size;
    for i in 0..n_chunks {
        let offset = if n_chunks == 1 {
            0
        } else {
            (last_offset * u64::from(i)) / u64::from(n_chunks - 1)
        };
        let bytes = read_n_at(&mut file, offset, chunk_usize)?;
        out.push(PyBytes::new_bound(py, &bytes));
    }
    Ok(out)
}

fn rng_for(seed: u64) -> Pcg64Mcg {
    Pcg64Mcg::seed_from_u64(seed)
}

/// Read one randomly-positioned window of size `window` from `path`.
/// Returns `(offset, bytes)`. If the file is smaller than `window`, returns
/// `(0, full_file)`.
#[pyfunction]
pub fn read_random_window<'py>(
    py: Python<'py>,
    path: &str,
    window: u64,
    seed: u64,
) -> PyResult<(u64, Bound<'py, PyBytes>)> {
    if window == 0 {
        return Err(value_error("window must be > 0"));
    }
    let total = metadata_size(path)?;
    let mut file = open_for_read(path)?;
    let n_usize: usize = window
        .try_into()
        .map_err(|_| value_error("window exceeds usize range"))?;
    if total <= window {
        let bytes = read_n_at(&mut file, 0, n_usize)?;
        return Ok((0, PyBytes::new_bound(py, &bytes)));
    }
    let mut rng = rng_for(seed);
    let max_offset = total - window;
    let offset: u64 = rng.gen_range(0..=max_offset);
    let bytes = read_n_at(&mut file, offset, n_usize)?;
    Ok((offset, PyBytes::new_bound(py, &bytes)))
}

/// Streaming byte-cache builder: read each path, optionally truncate to
/// `max_bytes`, shift each byte by `byte_offset`, and append the resulting
/// `u16` ids to `output_path` (little-endian, no header). Returns the per-file
/// index `[(offset_in_elements, n_tokens, n_bytes), ...]` aligned with `paths`.
///
/// Mirrors `tokenize::tokenize_paths_to_file` for the byte-vocabulary variant.
/// Each chunk reads + shifts in rayon-parallel; chunk results are written
/// serially in input order to keep memory bounded.
#[pyfunction]
#[pyo3(signature = (paths, output_path, byte_offset, chunk_files=64, max_bytes=None))]
pub fn read_paths_to_file_as_bytes(
    py: Python<'_>,
    paths: &Bound<'_, PyList>,
    output_path: &str,
    byte_offset: u16,
    chunk_files: usize,
    max_bytes: Option<u64>,
) -> PyResult<Vec<(u64, u32, u64)>> {
    if chunk_files == 0 {
        return Err(value_error("chunk_files must be > 0"));
    }
    let path_strs: Vec<String> = paths
        .iter()
        .map(|p| p.extract::<String>())
        .collect::<PyResult<Vec<_>>>()?;
    let output_path = output_path.to_string();

    py.allow_threads(move || -> PyResult<Vec<(u64, u32, u64)>> {
        let mut writer = BufWriter::with_capacity(
            8 * 1024 * 1024,
            fs::File::create(&output_path).map_err(io_to_py)?,
        );
        let mut index: Vec<(u64, u32, u64)> = Vec::with_capacity(path_strs.len());
        let mut cursor: u64 = 0;

        for chunk in path_strs.chunks(chunk_files) {
            // Parallel read+shift, in-order output via collect.
            let chunk_results: Vec<(Vec<u16>, u64)> = chunk
                .par_iter()
                .map(|p| -> Result<(Vec<u16>, u64), anyhow::Error> {
                    let raw = match max_bytes {
                        Some(limit) => {
                            let mut file = fs::File::open(p)?;
                            let len = file.metadata()?.len();
                            let take = limit.min(len) as usize;
                            let mut buf = Vec::with_capacity(take);
                            Read::by_ref(&mut file)
                                .take(take as u64)
                                .read_to_end(&mut buf)?;
                            buf
                        }
                        None => fs::read(p)?,
                    };
                    let n_bytes = raw.len() as u64;
                    let shifted: Vec<u16> = raw.iter().map(|&b| u16::from(b) + byte_offset).collect();
                    Ok((shifted, n_bytes))
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(anyhow_to_py)?;

            for (ids, n_bytes) in chunk_results {
                let n_tokens = ids.len() as u32;
                let mut buf = Vec::with_capacity(ids.len() * 2);
                for id in &ids {
                    buf.extend_from_slice(&id.to_le_bytes());
                }
                writer.write_all(&buf).map_err(io_to_py)?;
                index.push((cursor, n_tokens, n_bytes));
                cursor += n_tokens as u64;
            }
        }

        writer.flush().map_err(io_to_py)?;
        Ok(index)
    })
}

/// Read `n_windows` randomly-positioned windows of size `window` from `path`.
/// Returns a list of `(offset, bytes)`. Sampling is with replacement; pass
/// distinct seeds if you need disjoint draws.
#[pyfunction]
pub fn read_random_windows<'py>(
    py: Python<'py>,
    path: &str,
    n_windows: u32,
    window: u64,
    seed: u64,
) -> PyResult<Vec<(u64, Bound<'py, PyBytes>)>> {
    if window == 0 {
        return Err(value_error("window must be > 0"));
    }
    if n_windows == 0 {
        return Ok(Vec::new());
    }
    let total = metadata_size(path)?;
    let mut file = open_for_read(path)?;
    let n_usize: usize = window
        .try_into()
        .map_err(|_| value_error("window exceeds usize range"))?;
    let mut out = Vec::with_capacity(n_windows as usize);
    if total <= window {
        let bytes = read_n_at(&mut file, 0, n_usize)?;
        for _ in 0..n_windows {
            out.push((0u64, PyBytes::new_bound(py, &bytes)));
        }
        return Ok(out);
    }
    let mut rng = rng_for(seed);
    let max_offset = total - window;
    for _ in 0..n_windows {
        let offset: u64 = rng.gen_range(0..=max_offset);
        let bytes = read_n_at(&mut file, offset, n_usize)?;
        out.push((offset, PyBytes::new_bound(py, &bytes)));
    }
    Ok(out)
}
