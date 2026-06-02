//! MLM masking and window sampling on already-tokenized streams.
//!
//! These functions exist to keep the dataloader hot path out of Python. They
//! operate on `Vec<u32>` token streams (the cached parquet rows) and return
//! ready-to-batch tensors — modulo the actual tensor allocation, which we
//! leave to PyTorch on the Python side.

use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;

use crate::errors::value_error;

fn rng_for(seed: u64) -> Pcg64Mcg {
    Pcg64Mcg::seed_from_u64(seed)
}

/// Sample one contiguous window of `seq_len` tokens from `tokens`. If the
/// stream is shorter than `seq_len`, the window is padded on the right with
/// `pad_id` and `attention_mask` is set to 0 in those positions.
///
/// Returns `(input_ids, attention_mask)` — both length `seq_len`.
#[pyfunction]
pub fn sample_token_window(
    tokens: Vec<u32>,
    seq_len: u32,
    pad_id: u32,
    seed: u64,
) -> PyResult<(Vec<u32>, Vec<u8>)> {
    if seq_len == 0 {
        return Err(value_error("seq_len must be > 0"));
    }
    let n = tokens.len();
    let m = seq_len as usize;
    if n <= m {
        let mut ids = vec![pad_id; m];
        let mut mask = vec![0u8; m];
        for (i, &t) in tokens.iter().enumerate() {
            ids[i] = t;
            mask[i] = 1;
        }
        return Ok((ids, mask));
    }
    let mut rng = rng_for(seed);
    let max_start = n - m;
    let start = rng.gen_range(0..=max_start);
    let ids = tokens[start..start + m].to_vec();
    let mask = vec![1u8; m];
    Ok((ids, mask))
}

/// Sample `n_windows` independent windows from a single token stream. Each
/// window draws its own random offset with the supplied seed (advancing the
/// PRNG between draws). Returns parallel lists `(input_ids, attention_mask)`
/// each of length `n_windows`.
#[pyfunction]
pub fn sample_token_windows(
    tokens: Vec<u32>,
    seq_len: u32,
    n_windows: u32,
    pad_id: u32,
    seed: u64,
) -> PyResult<(Vec<Vec<u32>>, Vec<Vec<u8>>)> {
    if seq_len == 0 {
        return Err(value_error("seq_len must be > 0"));
    }
    if n_windows == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let n = tokens.len();
    let m = seq_len as usize;
    let mut ids_out = Vec::with_capacity(n_windows as usize);
    let mut mask_out = Vec::with_capacity(n_windows as usize);
    if n <= m {
        let mut ids = vec![pad_id; m];
        let mut mask = vec![0u8; m];
        for (i, &t) in tokens.iter().enumerate() {
            ids[i] = t;
            mask[i] = 1;
        }
        for _ in 0..n_windows {
            ids_out.push(ids.clone());
            mask_out.push(mask.clone());
        }
        return Ok((ids_out, mask_out));
    }
    let mut rng = rng_for(seed);
    let max_start = n - m;
    for _ in 0..n_windows {
        let start = rng.gen_range(0..=max_start);
        ids_out.push(tokens[start..start + m].to_vec());
        mask_out.push(vec![1u8; m]);
    }
    Ok((ids_out, mask_out))
}

fn mlm_one(
    input_ids: &[u32],
    special: &std::collections::HashSet<u32>,
    mask_id: u32,
    first_real_id: u32,
    n_real: u32,
    mask_ratio: f64,
    p_mask_only: f64,
    p_mask_or_random: f64,
    seed: u64,
) -> (Vec<u32>, Vec<i64>) {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut ids: Vec<u32> = input_ids.to_vec();
    let mut labels = vec![-100i64; ids.len()];
    for i in 0..ids.len() {
        if special.contains(&ids[i]) {
            continue;
        }
        if rng.r#gen::<f64>() >= mask_ratio {
            continue;
        }
        labels[i] = i64::from(ids[i]);
        let r: f64 = rng.r#gen();
        if r < p_mask_only {
            ids[i] = mask_id;
        } else if r < p_mask_or_random {
            let pick = first_real_id + rng.gen_range(0..n_real);
            ids[i] = pick;
        }
    }
    (ids, labels)
}

/// Batched form of [`mlm_mask`]: apply MLM masking to many sequences in one
/// PyO3 call, parallelising across rows with rayon. Per-row seeds are derived
/// from `base_seed` so the result is fully reproducible.
///
/// Returns `(input_ids_masked, labels)` — both lists of length `len(batch)`,
/// each row matching the corresponding input row.
#[pyfunction]
#[pyo3(signature = (batch_input_ids, special_ids, mask_id, vocab_size, first_real_id, mask_ratio, base_seed, keep_random_split=(0.8, 0.1, 0.1)))]
#[allow(clippy::too_many_arguments)]
pub fn mlm_mask_many(
    py: Python<'_>,
    batch_input_ids: Vec<Vec<u32>>,
    special_ids: Vec<u32>,
    mask_id: u32,
    vocab_size: u32,
    first_real_id: u32,
    mask_ratio: f64,
    base_seed: u64,
    keep_random_split: (f64, f64, f64),
) -> PyResult<(Vec<Vec<u32>>, Vec<Vec<i64>>)> {
    if !(0.0..=1.0).contains(&mask_ratio) {
        return Err(value_error("mask_ratio must be in [0, 1]"));
    }
    let (p_mask, p_random, p_keep) = keep_random_split;
    if (p_mask + p_random + p_keep - 1.0).abs() > 1e-6 {
        return Err(value_error("keep_random_split must sum to 1.0"));
    }
    if first_real_id >= vocab_size {
        return Err(value_error("first_real_id must be < vocab_size"));
    }
    let n_real = vocab_size - first_real_id;
    if n_real == 0 {
        return Err(value_error("no real tokens to draw random replacements from"));
    }
    let p_mask_only = p_mask;
    let p_mask_or_random = p_mask + p_random;
    let special: std::collections::HashSet<u32> = special_ids.into_iter().collect();

    let result: Vec<(Vec<u32>, Vec<i64>)> = py.allow_threads(|| {
        batch_input_ids
            .par_iter()
            .enumerate()
            .map(|(i, row)| {
                let row_seed = base_seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(i as u64);
                mlm_one(
                    row,
                    &special,
                    mask_id,
                    first_real_id,
                    n_real,
                    mask_ratio,
                    p_mask_only,
                    p_mask_or_random,
                    row_seed,
                )
            })
            .collect()
    });

    let mut ids_out = Vec::with_capacity(result.len());
    let mut labels_out = Vec::with_capacity(result.len());
    for (ids, labels) in result {
        ids_out.push(ids);
        labels_out.push(labels);
    }
    Ok((ids_out, labels_out))
}

/// Apply BERT/MosaicBERT-style MLM masking to `input_ids` in place.
///
/// Each non-special position is selected for masking with probability
/// `mask_ratio`. Selected positions follow the canonical 80/10/10 split:
///
/// - 80 % replaced with `mask_id`,
/// - 10 % replaced with a random in-vocab token in `[first_real_id, vocab_size)`,
/// - 10 % left unchanged.
///
/// Returns `(input_ids_masked, labels)`. `labels[i]` is the original token at
/// position `i` if it was selected for the loss, else `-100` (the standard
/// HuggingFace ignore index).
///
/// `special_ids` is the list of vocabulary IDs that must never be masked
/// (`<|cls|>`, `<|sep|>`, `<|pad|>`, etc.). Positions in `input_ids` whose
/// token is in this set are skipped.
#[pyfunction]
#[pyo3(signature = (input_ids, special_ids, mask_id, vocab_size, first_real_id, mask_ratio, seed, keep_random_split=(0.8, 0.1, 0.1)))]
#[allow(clippy::too_many_arguments)]
pub fn mlm_mask(
    input_ids: Vec<u32>,
    special_ids: Vec<u32>,
    mask_id: u32,
    vocab_size: u32,
    first_real_id: u32,
    mask_ratio: f64,
    seed: u64,
    keep_random_split: (f64, f64, f64),
) -> PyResult<(Vec<u32>, Vec<i64>)> {
    if !(0.0..=1.0).contains(&mask_ratio) {
        return Err(value_error("mask_ratio must be in [0, 1]"));
    }
    let (p_mask, p_random, p_keep) = keep_random_split;
    let total = p_mask + p_random + p_keep;
    if (total - 1.0).abs() > 1e-6 {
        return Err(value_error("keep_random_split must sum to 1.0"));
    }
    if first_real_id >= vocab_size {
        return Err(value_error("first_real_id must be < vocab_size"));
    }
    let special: std::collections::HashSet<u32> = special_ids.into_iter().collect();
    let mut rng = rng_for(seed);
    let mut ids = input_ids.clone();
    let mut labels = vec![-100i64; ids.len()];
    let n_real = (vocab_size - first_real_id) as usize;
    if n_real == 0 {
        return Err(value_error(
            "no real (non-special) tokens to draw random replacements from",
        ));
    }
    let p_mask_only = p_mask;
    let p_mask_or_random = p_mask + p_random;
    for i in 0..ids.len() {
        if special.contains(&ids[i]) {
            continue;
        }
        if rng.r#gen::<f64>() >= mask_ratio {
            continue;
        }
        labels[i] = i64::from(ids[i]);
        let r: f64 = rng.r#gen();
        if r < p_mask_only {
            ids[i] = mask_id;
        } else if r < p_mask_or_random {
            let pick = first_real_id as usize + rng.gen_range(0..n_real);
            ids[i] = pick as u32;
        }
        // else: 10% — leave unchanged
    }
    Ok((ids, labels))
}
