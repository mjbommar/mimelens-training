//! BPE tokenization for binary data, wrapping `bbpe::BinaryTokenizer`.
//!
//! Two entry points:
//!
//! - [`BinaryTokenizer`] — load a `tokenizer.json` once, then encode/decode many times.
//! - [`tokenize_paths`] — parallel batch tokenization over a list of file paths
//!   (rayon-parallel; used to pre-encode the whole corpus to a token cache).

use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use bbpe::{BinaryTokenizer as InnerTokenizer, BinaryTokenizerOptions, LegacyByteBehavior};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use rayon::prelude::*;
use tokenizers::Tokenizer;

use crate::errors::{anyhow_to_py, io_to_py, value_error};

/// Wrapper around `bbpe::BinaryTokenizer`, exposed to Python.
#[pyclass(module = "binary_embedding._native", frozen)]
pub struct BinaryTokenizer {
    inner: Arc<InnerTokenizer>,
    vocab_size: usize,
}

#[pymethods]
impl BinaryTokenizer {
    /// Load a tokenizer.json from disk.
    #[staticmethod]
    pub fn from_file(path: &str) -> PyResult<Self> {
        let path = Path::new(path);
        if !path.exists() {
            return Err(value_error(format!(
                "tokenizer file not found: {}",
                path.display()
            )));
        }
        let bytes = fs::read(path).map_err(io_to_py)?;
        let tokenizer = Tokenizer::from_bytes(&bytes)
            .map_err(|e| anyhow_to_py(anyhow::anyhow!(e.to_string())))?;
        let inner = InnerTokenizer::from_tokenizer_with_options(
            tokenizer,
            BinaryTokenizerOptions::default().legacy_byte_behavior(LegacyByteBehavior::Auto),
        )
        .map_err(|e| anyhow_to_py(anyhow::anyhow!(e.to_string())))?;
        let vocab_size = inner.inner().get_vocab_size(true);
        Ok(Self {
            inner: Arc::new(inner),
            vocab_size,
        })
    }

    /// Total vocabulary size including special tokens.
    #[getter]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Encode raw bytes to token IDs.
    #[pyo3(signature = (data, add_special_tokens=None))]
    pub fn encode(
        &self,
        py: Python<'_>,
        data: &[u8],
        add_special_tokens: Option<bool>,
    ) -> PyResult<Vec<u32>> {
        let add = add_special_tokens.unwrap_or(false);
        // Release the GIL for the actual encoding work.
        py.allow_threads(|| {
            self.inner
                .encode_bytes(data, add)
                .map_err(|e| anyhow_to_py(anyhow::anyhow!(e.to_string())))
        })
    }

    /// Encode a batch of byte sequences in parallel via rayon.
    #[pyo3(signature = (data, add_special_tokens=None))]
    pub fn encode_batch(
        &self,
        py: Python<'_>,
        data: Vec<Vec<u8>>,
        add_special_tokens: Option<bool>,
    ) -> PyResult<Vec<Vec<u32>>> {
        let add = add_special_tokens.unwrap_or(false);
        let inner = self.inner.clone();
        py.allow_threads(move || {
            data.par_iter()
                .map(|chunk| {
                    inner
                        .encode_bytes(chunk, add)
                        .map_err(|e| anyhow::anyhow!(e.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(anyhow_to_py)
        })
    }

    /// Decode token IDs back to bytes.
    #[pyo3(signature = (ids, skip_special_tokens=None))]
    pub fn decode<'py>(
        &self,
        py: Python<'py>,
        ids: Vec<u32>,
        skip_special_tokens: Option<bool>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let skip = skip_special_tokens.unwrap_or(true);
        let bytes = py
            .allow_threads(|| {
                self.inner
                    .decode_to_bytes(&ids, skip)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))
            })
            .map_err(anyhow_to_py)?;
        Ok(PyBytes::new_bound(py, &bytes))
    }

    /// Round-trip self-check on a byte slice. Returns whether
    /// `decode(encode(data), skip_special=False) == data`.
    pub fn round_trip(&self, py: Python<'_>, data: &[u8]) -> PyResult<bool> {
        let inner = self.inner.clone();
        let owned: Vec<u8> = data.to_vec();
        py.allow_threads(move || {
            let ids = inner
                .encode_bytes(&owned, false)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let back = inner
                .decode_to_bytes(&ids, false)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            Ok(back == owned)
        })
        .map_err(anyhow_to_py)
    }
}

/// Tokenize each file at `paths`, optionally truncated to `max_bytes` from offset 0.
/// Runs the file reads + encoding in parallel via rayon. Returns a list whose i-th
/// entry is the token IDs for `paths[i]`.
#[pyfunction]
#[pyo3(signature = (tokenizer, paths, max_bytes=None, add_special_tokens=false))]
pub fn tokenize_paths(
    py: Python<'_>,
    tokenizer: &BinaryTokenizer,
    paths: &Bound<'_, PyList>,
    max_bytes: Option<u64>,
    add_special_tokens: bool,
) -> PyResult<Vec<Vec<u32>>> {
    let path_strs: Vec<String> = paths
        .iter()
        .map(|p| p.extract::<String>())
        .collect::<PyResult<Vec<_>>>()?;
    let inner = tokenizer.inner.clone();
    py.allow_threads(move || {
        path_strs
            .par_iter()
            .map(|p| -> Result<Vec<u32>, anyhow::Error> {
                let bytes = match max_bytes {
                    Some(limit) => read_truncated(p, limit)?,
                    None => fs::read(p)?,
                };
                inner
                    .encode_bytes(&bytes, add_special_tokens)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(anyhow_to_py)
    })
}

/// Tokenize each file at `paths` and stream the resulting `u16` token IDs to
/// a single packed binary file at `output_path` (little-endian, no header).
///
/// Files are processed in chunks of `chunk_files` (default 64): each chunk is
/// rayon-parallel-encoded into memory, then written in input order, then freed.
/// This caps peak RSS regardless of corpus size — the FFI boundary never sees
/// the full token stream.
///
/// Returns an index `[(offset, n_tokens, n_bytes), …]` aligned with `paths`,
/// where `offset` is the *element* index into the output stream (multiply by
/// 2 to get the byte offset). The caller (Python) writes the per-file metadata
/// (sha256 etc.) to a parquet sidecar.
///
/// Fails fast if any token id exceeds `u16::MAX`. Our 5 production tokenizers
/// all fit (≤ 64 K + 7 specials = 65 543 < 65 536? no — at 64 K that's 65 543
/// total which already exceeds u16). For 64 K we keep the index in u32. The
/// `dtype` argument controls this: "u16" or "u32".
#[pyfunction]
#[pyo3(signature = (tokenizer, paths, output_path, dtype="u16", chunk_files=64, max_bytes=None, add_special_tokens=false))]
#[allow(clippy::too_many_arguments)]
pub fn tokenize_paths_to_file(
    py: Python<'_>,
    tokenizer: &BinaryTokenizer,
    paths: &Bound<'_, PyList>,
    output_path: &str,
    dtype: &str,
    chunk_files: usize,
    max_bytes: Option<u64>,
    add_special_tokens: bool,
) -> PyResult<Vec<(u64, u32, u64)>> {
    if chunk_files == 0 {
        return Err(value_error("chunk_files must be > 0"));
    }
    let elem_size: usize = match dtype {
        "u16" => 2,
        "u32" => 4,
        other => return Err(value_error(format!("dtype must be 'u16' or 'u32', got {other:?}"))),
    };
    let cap_id = if dtype == "u16" {
        u32::from(u16::MAX)
    } else {
        u32::MAX
    };

    let path_strs: Vec<String> = paths
        .iter()
        .map(|p| p.extract::<String>())
        .collect::<PyResult<Vec<_>>>()?;
    let inner = tokenizer.inner.clone();
    let output_path = output_path.to_string();
    let dtype_owned = dtype.to_string();

    py.allow_threads(move || -> PyResult<Vec<(u64, u32, u64)>> {
        let mut writer = BufWriter::with_capacity(
            8 * 1024 * 1024,
            fs::File::create(&output_path).map_err(io_to_py)?,
        );
        let mut index: Vec<(u64, u32, u64)> = Vec::with_capacity(path_strs.len());
        let mut offset: u64 = 0;

        for chunk in path_strs.chunks(chunk_files) {
            // Parallel encode this chunk; errors short-circuit.
            let chunk_results: Vec<(Vec<u32>, u64)> = chunk
                .par_iter()
                .map(|p| -> Result<(Vec<u32>, u64), anyhow::Error> {
                    let bytes = match max_bytes {
                        Some(limit) => read_truncated(p, limit)?,
                        None => fs::read(p)?,
                    };
                    let n_bytes = bytes.len() as u64;
                    let ids = inner
                        .encode_bytes(&bytes, add_special_tokens)
                        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    Ok((ids, n_bytes))
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(anyhow_to_py)?;

            // Serial write, in input order. Validate ids fit `dtype`.
            for (ids, n_bytes) in chunk_results {
                if let Some(&bad) = ids.iter().find(|&&id| id > cap_id) {
                    return Err(value_error(format!(
                        "token id {bad} does not fit in dtype={dtype_owned}; \
                         use dtype='u32' for vocabularies that exceed u16::MAX"
                    )));
                }
                let n_tokens = ids.len() as u32;
                if dtype_owned == "u16" {
                    let mut buf = Vec::with_capacity(ids.len() * 2);
                    for id in &ids {
                        buf.extend_from_slice(&(*id as u16).to_le_bytes());
                    }
                    writer.write_all(&buf).map_err(io_to_py)?;
                } else {
                    let mut buf = Vec::with_capacity(ids.len() * 4);
                    for id in &ids {
                        buf.extend_from_slice(&id.to_le_bytes());
                    }
                    writer.write_all(&buf).map_err(io_to_py)?;
                }
                index.push((offset, n_tokens, n_bytes));
                offset += n_tokens as u64;
            }
        }

        writer.flush().map_err(io_to_py)?;
        let _ = elem_size; // kept for callers that want the byte-offset multiplier
        Ok(index)
    })
}

fn read_truncated(path: &str, limit: u64) -> Result<Vec<u8>, anyhow::Error> {
    use std::io::Read;
    let mut file = fs::File::open(path)?;
    let len = file.metadata()?.len();
    let take = limit.min(len) as usize;
    let mut buf = Vec::with_capacity(take);
    Read::by_ref(&mut file)
        .take(take as u64)
        .read_to_end(&mut buf)?;
    Ok(buf)
}
