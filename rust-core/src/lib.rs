use ndarray::{Array2, Axis};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use std::cmp::Ordering;

use ndarray::{Array1};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;

/// High-performance quantum-inspired attention kernel
#[pyfunction]
fn quantum_attention_rs(
    q: PyReadonlyArray2<f32>,
    k: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    top_k: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let q = q.as_array();
    let k = k.as_array();
    let v = v.as_array();

    let seq_len = q.len_of(Axis(0));
    let d_model = q.len_of(Axis(1));

    if k.len_of(Axis(0)) != seq_len || v.len_of(Axis(0)) != seq_len
        || k.len_of(Axis(1)) != d_model || v.len_of(Axis(1)) != d_model
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
    let rows: Vec<Array1<f32>> = (0..seq_len)
        .into_par_iter()
        .map(|i| {
            let qi = q.row(i).to_owned(); // (d_model)
            let mut logits = k.dot(&qi); // (seq_len)
            logits.mapv_inplace(|x| x / scale);

            // Amplitude encoding and probabilities: p_j ∝ exp(logit/2)^2 = exp(logit)
            // So p ∝ exp(logit). We'll compute stable softmax-like weights for sampling only.
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut weights: Vec<f32> = logits
                .iter()
                .map(|&x| (x - max_logit).exp())
                .collect();

            let sum_w: f32 = weights.iter().sum();
            if !sum_w.is_finite() || sum_w <= 0.0 {
                // Fallback to zeros if invalid
                return Array1::<f32>::zeros(d_model);
            }

            // Build categorical sampler
            // Normalize not required by WeightedIndex; it uses relative weights
            let dist = WeightedIndex::new(&weights).ok();
            if dist.is_none() {
                return Array1::<f32>::zeros(d_model);
            }
            let dist = dist.unwrap();
            let mut rng = thread_rng();

            // Empirical probability via sampling
            let mut counts = vec![0usize; seq_len];
            for _ in 0..num_samples {
                let j = dist.sample(&mut rng);
                counts[j] += 1;
            }
            let inv_samples = 1.0f32 / (num_samples as f32);

            // Weighted sum over V using empirical probabilities
            let mut out = Array1::<f32>::zeros(d_model);
            for (j, &c) in counts.iter().enumerate() {
                if c > 0 {
                    let w = (c as f32) * inv_samples;
                    let vj = v.row(j);
                    out = out + &(vj.to_owned() * w);
                }
            }

            out
        })
        .collect();

    let mut result = Array2::<f32>::zeros((seq_len, d_model));
    for (i, row) in rows.into_iter().enumerate() {
        result.row_mut(i).assign(&row);
    }

    Python::with_gil(|py| Ok(result.into_pyarray(py).to_owned()))
}

/// Classical efficient attention approximation
#[pyfunction]
fn classical_attention_rs(
    q: PyReadonlyArray2<f32>,
    k: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    top_k: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let q = q.as_array();
    let k = k.as_array();
    let v = v.as_array();

    let seq_len = q.len_of(Axis(0));
    let d_model = q.len_of(Axis(1));

    if k.len_of(Axis(0)) != seq_len || v.len_of(Axis(0)) != seq_len
        || k.len_of(Axis(1)) != d_model || v.len_of(Axis(1)) != d_model
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "q, k, v must have shapes (seq_len, d_model) with matching dims",
        ));
    }

    let scale = (d_model as f32).sqrt();
    let k_eff = top_k.clamp(1, seq_len.max(1));

    // Compute per-row outputs in parallel
    let rows: Vec<Array1<f32>> = (0..seq_len)
        .into_par_iter()
        .map(|i| {
            let qi = q.row(i).to_owned(); // (d_model)

            // scores = K · qi / sqrt(d)
            let mut scores = k.dot(&qi);
            scores.mapv_inplace(|x| x / scale);

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
            let mut exp_scores: Vec<f32> = top_idx.iter().map(|&j| (scores[j] - max_top).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            if !sum_exp.is_finite() || sum_exp <= 0.0 {
                return Array1::<f32>::zeros(d_model);
            }
            for e in exp_scores.iter_mut() {
                *e /= sum_exp;
            }

            // Weighted sum of V rows
            let mut out = Array1::<f32>::zeros(d_model);
            for (w, &j) in exp_scores.iter().zip(top_idx.iter()) {
                let vj = v.row(j);
                out = out + &(vj.to_owned() * *w);
            }
            out
        })
        .collect();

    let mut result = Array2::<f32>::zeros((seq_len, d_model));
    for (i, row) in rows.into_iter().enumerate() {
        result.row_mut(i).assign(&row);
    }

    Python::with_gil(|py| Ok(result.into_pyarray(py).to_owned()))
}

/// Python module definition
#[pymodule]
fn qtransformers_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quantum_attention_rs, m)?)?;
    m.add_function(wrap_pyfunction!(classical_attention_rs, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests;
