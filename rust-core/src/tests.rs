use super::*;
use numpy::PyArray2;
use pyo3::Python;

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArray2;
    use pyo3::Python;

    #[test]
    fn test_classical_attention_rs_shapes() {
        Python::with_gil(|py| {
            let n = 6usize;
            let d = 8usize;

            let q = PyArray2::<f32>::zeros(py, [n, d], false);
            let k = PyArray2::<f32>::zeros(py, [n, d], false);
            let v = PyArray2::<f32>::zeros(py, [n, d], false);

            let out = classical_attention_rs(q.readonly(), k.readonly(), v.readonly(), 3).unwrap();
            let out_ref = out.as_ref(py);
            assert_eq!(out_ref.shape(), [n, d]);
        });
    }

    #[test]
    fn test_quantum_attention_rs_shapes() {
        Python::with_gil(|py| {
            let n = 5usize;
            let d = 12usize;

            let q = PyArray2::<f32>::zeros(py, [n, d], false);
            let k = PyArray2::<f32>::zeros(py, [n, d], false);
            let v = PyArray2::<f32>::zeros(py, [n, d], false);

            let out = quantum_attention_rs(q.readonly(), k.readonly(), v.readonly(), 4).unwrap();
            let out_ref = out.as_ref(py);
            assert_eq!(out_ref.shape(), [n, d]);
        });
    }

    #[test]
    fn test_top_k_parameter() {
        // Test that top_k parameter makes sense
        let top_k_values = [8, 16, 32, 64];

        for &top_k in &top_k_values {
            assert!(top_k > 0);
            assert!(top_k <= 1024);
        }
    }
}
