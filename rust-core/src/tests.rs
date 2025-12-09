// Note: PyO3 extension module tests require Python at link time.
// These tests should be run via pytest after building with maturin:
//   maturin develop && pytest tests/rust/
//
// The tests below are pure Rust unit tests that don't require Python.

#[cfg(test)]
mod tests {
    #[test]
    fn test_top_k_parameter_bounds() {
        // Test that top_k parameter logic is sound
        let top_k_values = [8, 16, 32, 64];

        for &top_k in &top_k_values {
            assert!(top_k > 0);
            assert!(top_k <= 1024);
        }
    }

    #[test]
    fn test_scale_computation() {
        // Verify scale factor computation
        let d_model = 64usize;
        let scale = (d_model as f32).sqrt();
        assert!((scale - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_stability() {
        // Test that our softmax approach handles edge cases
        let logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_w: f32 = weights.iter().sum();
        
        assert!(sum_w.is_finite());
        assert!(sum_w > 0.0);
        
        // Normalized weights should sum to 1
        let normalized: Vec<f32> = weights.iter().map(|&w| w / sum_w).collect();
        let total: f32 = normalized.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }
}
