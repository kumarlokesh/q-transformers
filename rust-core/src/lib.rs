use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use std::cmp::Ordering;

use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;

/// High-performance quantum-inspired attention kernel
#[pyfunction]
fn quantum_attention_rs<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<'py, f32>,
    k: PyReadonlyArray2<'py, f32>,
    v: PyReadonlyArray2<'py, f32>,
    top_k: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let q = q.as_array();
    let k = k.as_array();
    let v = v.as_array();

    let seq_len = q.shape()[0];
    let d_model = q.shape()[1];

    if k.shape()[0] != seq_len
        || v.shape()[0] != seq_len
        || k.shape()[1] != d_model
        || v.shape()[1] != d_model
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "q, k, v must have shapes (seq_len, d_model) with matching dims",
        ));
    }

    // Sampling-based approximation of attention probabilities
    // logits_i = (K · q_i) / sqrt(d)
    let scale = (d_model as f32).sqrt();
    let num_samples = top_k.max(1).min(seq_len.max(1));

    // Compute per-row outputs in parallel
    let rows: Vec<Vec<f32>> = (0..seq_len)
        .into_par_iter()
        .map(|i| {
            // Compute logits for row i: K @ q[i]
            let mut logits: Vec<f32> = vec![0.0; seq_len];
            for j in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..d_model {
                    dot += k[[j, d]] * q[[i, d]];
                }
                logits[j] = dot / scale;
            }

            // Amplitude encoding and probabilities: p_j ∝ exp(logit/2)^2 = exp(logit)
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();

            let sum_w: f32 = weights.iter().sum();
            if !sum_w.is_finite() || sum_w <= 0.0 {
                return vec![0.0f32; d_model];
            }

            // Build categorical sampler
            let dist = match WeightedIndex::new(&weights) {
                Ok(d) => d,
                Err(_) => return vec![0.0f32; d_model],
            };
            let mut rng = thread_rng();

            // Empirical probability via sampling
            let mut counts = vec![0usize; seq_len];
            for _ in 0..num_samples {
                let j = dist.sample(&mut rng);
                counts[j] += 1;
            }
            let inv_samples = 1.0f32 / (num_samples as f32);

            // Weighted sum over V using empirical probabilities
            let mut out = vec![0.0f32; d_model];
            for (j, &c) in counts.iter().enumerate() {
                if c > 0 {
                    let w = (c as f32) * inv_samples;
                    for d in 0..d_model {
                        out[d] += v[[j, d]] * w;
                    }
                }
            }

            out
        })
        .collect();

    // Create 2D array from rows
    let result = PyArray2::from_vec2_bound(py, &rows)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {}", e)))?;

    Ok(result)
}

/// Classical efficient attention approximation
#[pyfunction]
fn classical_attention_rs<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<'py, f32>,
    k: PyReadonlyArray2<'py, f32>,
    v: PyReadonlyArray2<'py, f32>,
    top_k: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let q = q.as_array();
    let k = k.as_array();
    let v = v.as_array();

    let seq_len = q.shape()[0];
    let d_model = q.shape()[1];

    if k.shape()[0] != seq_len
        || v.shape()[0] != seq_len
        || k.shape()[1] != d_model
        || v.shape()[1] != d_model
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "q, k, v must have shapes (seq_len, d_model) with matching dims",
        ));
    }

    let scale = (d_model as f32).sqrt();
    let k_eff = top_k.clamp(1, seq_len.max(1));

    // Compute per-row outputs in parallel
    let rows: Vec<Vec<f32>> = (0..seq_len)
        .into_par_iter()
        .map(|i| {
            // Compute scores for row i: K @ q[i] / sqrt(d)
            let mut scores: Vec<f32> = vec![0.0; seq_len];
            for j in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..d_model {
                    dot += k[[j, d]] * q[[i, d]];
                }
                scores[j] = dot / scale;
            }

            // Select top-k indices by score (descending)
            let mut idx: Vec<usize> = (0..seq_len).collect();
            idx.sort_unstable_by(|&a, &b| {
                let sa = scores[a];
                let sb = scores[b];
                sb.partial_cmp(&sa).unwrap_or(Ordering::Equal)
            });
            let top_idx = &idx[0..k_eff];

            // Stable softmax over top-k
            let max_top = top_idx
                .iter()
                .map(|&j| scores[j])
                .fold(f32::NEG_INFINITY, f32::max);
            let mut exp_scores: Vec<f32> = top_idx
                .iter()
                .map(|&j| (scores[j] - max_top).exp())
                .collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            if !sum_exp.is_finite() || sum_exp <= 0.0 {
                return vec![0.0f32; d_model];
            }
            for e in exp_scores.iter_mut() {
                *e /= sum_exp;
            }

            // Weighted sum of V rows
            let mut out = vec![0.0f32; d_model];
            for (w, &j) in exp_scores.iter().zip(top_idx.iter()) {
                for d in 0..d_model {
                    out[d] += v[[j, d]] * *w;
                }
            }
            out
        })
        .collect();

    // Create 2D array from rows
    let result = PyArray2::from_vec2_bound(py, &rows)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {}", e)))?;

    Ok(result)
}

/// Python module definition
#[pymodule]
fn qtransformers_core(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quantum_attention_rs, m)?)?;
    m.add_function(wrap_pyfunction!(classical_attention_rs, m)?)?;
    Ok(())
}

// Note: Unit tests for PyO3 extension modules cannot run via `cargo test`
// because they require Python symbols at link time. Tests are run via:
//   maturin develop && pytest tests/rust/
