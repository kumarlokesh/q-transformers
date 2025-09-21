"""
Quantum Error Mitigation Techniques for Q-Transformers

Implements error mitigation methods from quantum computing
to improve attention approximation accuracy:
- Zero-noise extrapolation
- Symmetry verification and correction
- Probabilistic error cancellation
- Virtual distillation
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import curve_fit


class ZeroNoiseExtrapolation:
    """
    Zero-Noise Extrapolation (ZNE) for quantum attention.

    Runs attention computation at multiple noise levels and
    extrapolates to the zero-noise limit using polynomial fitting.
    """

    def __init__(
        self,
        noise_levels: List[float] = [0.0, 0.01, 0.02, 0.05],
        extrapolation_method: str = "polynomial",
        polynomial_degree: int = 2,
    ):
        """
        Initialize ZNE parameters.

        Args:
            noise_levels: List of noise levels to sample
            extrapolation_method: "polynomial", "exponential", or "richardson"
            polynomial_degree: Degree for polynomial extrapolation
        """
        self.noise_levels = sorted(noise_levels)
        self.extrapolation_method = extrapolation_method
        self.polynomial_degree = polynomial_degree

    def extrapolate_zero_noise(
        self, noisy_results: List[torch.Tensor], noise_levels: List[float]
    ) -> torch.Tensor:
        """
        Extrapolate to zero noise from multiple noisy measurements.

        Args:
            noisy_results: List of attention outputs at different noise levels
            noise_levels: Corresponding noise levels

        Returns:
            Extrapolated zero-noise attention output
        """
        if len(noisy_results) != len(noise_levels):
            raise ValueError("Number of results must match number of noise levels")

        # Convert to numpy for fitting
        _noise_array = np.array(noise_levels)

        # Extrapolate each element independently
        _result_shape = noisy_results[0].shape
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _dtype = noisy_results[0].dtype

        _extrapolated = torch.zeros(result_shape, _device=device, _dtype=dtype)

        # Flatten for easier processing
        _flattened_results = [result.flatten() for result in noisy_results]
        _n_elements = flattened_results[0].numel()

        for i in range(n_elements):
            # Extract values for this element across noise levels
            _values = np.array([result[i].cpu().item() for result in flattened_results])

            if self.extrapolation_method == "polynomial":
                _extrapolated_value = self._polynomial_extrapolation(
                    noise_array, values
                )
            elif self.extrapolation_method == "exponential":
                _extrapolated_value = self._exponential_extrapolation(
                    noise_array, values
                )
            elif self.extrapolation_method == "richardson":
                _extrapolated_value = self._richardson_extrapolation(
                    noise_array, values
                )
            else:
                raise ValueError(
                    "Unknown extrapolation method: {self.extrapolation_method}"
                )

            extrapolated.flatten()[i] = extrapolated_value

        return extrapolated.reshape(result_shape)

    def _polynomial_extrapolation(
        self, noise_levels: np.ndarray, values: np.ndarray
    ) -> float:
        """Polynomial extrapolation to zero noise."""
        try:
            # Fit polynomial: f(x) = a0 + a1*x + a2*x^2 + ...
            _coeffs = np.polyfit(noise_levels, values, self.polynomial_degree)
            # Extrapolate to x=0 (zero noise)
            return float(coeffs[-1])  # Constant term
        except Exception:
            # Fallback to linear extrapolation
            if len(values) >= 2:
                _slope = (values[1] - values[0]) / (
                    noise_levels[1] - noise_levels[0] + 1e-12
                )
                return float(values[0] - slope * noise_levels[0])
            else:
                return float(values[0])

    def _exponential_extrapolation(
        self, noise_levels: np.ndarray, values: np.ndarray
    ) -> float:
        """Exponential decay extrapolation: f(x) = A * exp(-B*x) + C."""
        try:

            def exp_model(x, A, B, C):
                return A * np.exp(-B * x) + C

            # Initial guess
            _p0 = [values[0] - values[-1], 1.0, values[-1]]

            popt, _ = curve_fit(exp_model, noise_levels, values, _p0=p0, _maxfev=1000)
            A, B, C = popt

            # Extrapolate to x=0
            return float(A + C)
        except Exception:
            # Fallback to polynomial
            return self._polynomial_extrapolation(noise_levels, values)

    def _richardson_extrapolation(
        self, noise_levels: np.ndarray, values: np.ndarray
    ) -> float:
        """Richardson extrapolation for systematic error removal."""
        if len(values) < 2:
            return float(values[0])

        # Simple Richardson: R(0) = 2*f(h/2) - f(h)
        if len(values) >= 2:
            h1, _h2 = noise_levels[0], noise_levels[1]
            f1, _f2 = values[0], values[1]

            if abs(h2 - 2 * h1) < 1e-6:  # h2 ≈ 2*h1
                return float(2 * f1 - f2)
            else:
                # General Richardson extrapolation
                _ratio = h2 / h1
                return float((f1 * ratio - f2) / (ratio - 1))

        return float(values[0])

    def mitigate_quantum_attention(
        self,
        attention_fn: Callable,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply ZNE to quantum attention computation.

        Args:
            attention_fn: Quantum attention function with noise_level parameter
            Q, K, V: Attention tensors
            **kwargs: Additional arguments for attention function

        Returns:
            Mitigated attention output and metrics
        """
        _noisy_results = []

        # Compute attention at different noise levels
        for noise_level in self.noise_levels:
            _result = attention_fn(Q, K, V, _noise_level=noise_level, **kwargs)
            noisy_results.append(result)

        # Extrapolate to zero noise
        _mitigated_output = self.extrapolate_zero_noise(
            noisy_results, self.noise_levels
        )

        # Compute metrics
        _variance_across_noise = torch.var(torch.stack(noisy_results), _dim=0).mean()
        _extrapolation_confidence = 1.0 / (1.0 + float(variance_across_noise))

        _metrics = {
            "noise_levels_used": self.noise_levels,
            "variance_across_noise": float(variance_across_noise),
            "extrapolation_confidence": extrapolation_confidence,
            "num_measurements": len(self.noise_levels),
        }

        return mitigated_output, metrics


class SymmetryVerification:
    """
    Symmetry verification and correction for quantum attention.

    Verifies expected symmetries in attention patterns and
    corrects violations to improve approximation quality.
    """

    def __init__(
        self,
        symmetries: List[str] = ["row_normalization", "causality", "positivity"],
        correctionstrength: float = 1.0,
    ):
        """
        Initialize symmetry verifier.

        Args:
            symmetries: List of symmetries to verify/enforce
            correctionstrength: Strength of symmetry corrections (0=none, 1=full)
        """
        self.symmetries = symmetries
        self.correctionstrength = correctionstrength

    def verify_attention_symmetries(
        self,
        attention_weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Verify attention pattern symmetries.

        Args:
            attention_weights: Attention weight matrix (batch, seq_len_q, seq_len_k)
            attention_mask: Optional mask for padding/causality

        Returns:
            Dictionary of symmetry violation metrics
        """
        _violations = {}

        if "row_normalization" in self.symmetries:
            # Check if attention weights sum to 1 along key dimension
            _row_sums = torch.sum(attention_weights, _dim=-1)  # (batch, seq_len_q)
            if attention_mask is not None:
                _valid_rows = attention_mask.sum(dim=-1) > 0
                _row_sums = row_sums[valid_rows]

            _normalization_error = torch.mean(torch.abs(row_sums - 1.0))
            violations["row_normalization"] = float(normalization_error)

        if "positivity" in self.symmetries:
            # Check if all attention weights are non-negative
            _negative_weights = torch.sum(attention_weights < 0).float()
            _total_weights = attention_weights.numel()
            violations["positivity"] = float(negative_weights / total_weights)

        if "causality" in self.symmetries and attention_mask is not None:
            # Check causal mask violations (for autoregressive models)
            _seq_len = attention_weights.shape[-1]
            _causal_mask = torch.triu(torch.ones(seq_len, seq_len), _diagonal=1).bool()
            _causal_mask = causal_mask.to(attention_weights.device)

            if (
                attention_weights.shape[-2] == attention_weights.shape[-1]
            ):  # Square attention
                _causal_violations = torch.sum(
                    attention_weights[:, causal_mask] > 1e-6
                ).float()
                _total_causal_positions = (
                    causal_mask.sum().float() * attention_weights.shape[0]
                )
                violations["causality"] = float(
                    causal_violations / total_causal_positions
                )

        if "attention_entropy" in self.symmetries:
            # Check if attention entropy is reasonable (not too peaked or too uniform)
            _entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-12), _dim=-1
            )  # (batch, seq_len_q)

            _max_entropy = math.log(
                attention_weights.shape[-1]
            )  # Uniform distribution entropy
            _normalized_entropy = entropy / max_entropy

            # Ideal entropy range: [0.2, 0.8] (not too peaked, not too uniform)
            _entropy_violations = torch.sum(
                (normalized_entropy < 0.2) | (normalized_entropy > 0.8)
            ).float()
            _total_queries = entropy.numel()
            violations["attention_entropy"] = float(entropy_violations / total_queries)

        return violations

    def correct_attention_symmetries(
        self,
        attention_weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Correct attention symmetry violations.

        Args:
            attention_weights: Original attention weights
            attention_mask: Optional attention mask

        Returns:
            Corrected attention weights
        """
        _corrected_weights = attention_weights.clone()

        if "positivity" in self.symmetries:
            # Enforce non-negativity
            _corrected_weights = torch.clamp(corrected_weights, _min=0.0)

        if "causality" in self.symmetries and attention_mask is not None:
            # Enforce causal mask
            _seq_len = corrected_weights.shape[-1]
            if corrected_weights.shape[-2] == corrected_weights.shape[-1]:
                _causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len), _diagonal=1
                ).bool()
                _causal_mask = causal_mask.to(corrected_weights.device)
                corrected_weights[:, causal_mask] = 0.0

        if "row_normalization" in self.symmetries:
            # Re-normalize rows to sum to 1
            if attention_mask is not None:
                _corrected_weights = corrected_weights.masked_fill(
                    ~attention_mask.bool(), 0.0
                )

            _row_sums = torch.sum(corrected_weights, _dim=-1, _keepdim=True)
            _corrected_weights = corrected_weights / (row_sums + 1e-12)

        # Apply correction strength
        if self.correctionstrength < 1.0:
            _corrected_weights = (
                self.correctionstrength * corrected_weights
                + (1 - self.correctionstrength) * attention_weights
            )

        return corrected_weights

    def symmetry_guided_sampling(
        self,
        logits: torch.Tensor,
        num_samples: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform sampling with symmetry constraints.

        Args:
            logits: Attention logits
            num_samples: Number of samples
            attention_mask: Optional attention mask

        Returns:
            Symmetry-corrected attention weights
        """
        # Initial sampling
        _probs = F.softmax(logits, _dim=-1)

        if attention_mask is not None:
            _probs = probs.masked_fill(~attention_mask.bool(), 0.0)
            _probs = probs / (torch.sum(probs, _dim=-1, _keepdim=True) + 1e-12)

        batch_size, seq_len_q, _seq_len_k = probs.shape
        _sampled_weights = torch.zeros_like(probs)

        for b in range(batch_size):
            for q in range(seq_len_q):
                _prob_row = probs[b, q, :]

                if prob_row.sum() > 1e-8:
                    # Sample with replacement
                    _samples = torch.multinomial(
                        prob_row, num_samples, _replacement=True
                    )

                    # Count samples
                    _sample_counts = torch.bincount(samples, _minlength=seq_len_k)
                    sampled_weights[b, q, :] = sample_counts.float() / num_samples

        # Apply symmetry corrections
        _corrected_weights = self.correct_attention_symmetries(
            sampled_weights, attention_mask
        )

        return corrected_weights


class ProbabilisticErrorCancellation:
    """
    Probabilistic Error Cancellation (PEC) for quantum attention.

    Uses probabilistic combinations of noisy operations to
    cancel systematic errors in quantum attention computation.
    """

    def __init__(self, error_rates: Dict[str, float] = None, num_trials: int = 10):
        """
        Initialize PEC with error rates.

        Args:
            error_rates: Dictionary of error rates for different noise types
            num_trials: Number of trials for probabilistic cancellation
        """
        self.error_rates = error_rates or {
            "depolarizing": 0.01,
            "amplitude_damping": 0.005,
            "phase_damping": 0.002,
        }
        self.num_trials = num_trials

    def generate_error_mitigation_circuits(
        self, base_operation: Callable, *args, **kwargs
    ) -> List[Tuple[Callable, float]]:
        """
        Generate error mitigation operation sequences.

        Args:
            base_operation: Base quantum attention operation
            *args, **kwargs: Arguments for base operation

        Returns:
            List of (operation, probability_weight) tuples
        """
        _circuits = []

        # Identity operation (no additional noise)
        circuits.append((lambda *a, **kw: base_operation(*a, **kw), 1.0))

        # Inverse operations for error cancellation
        for noise_type, error_rate in self.error_rates.items():
            if error_rate > 0:
                # Create operation that applies inverse of the systematic error
                def inverse_noise_op(*a, _noise_t=noise_type, _rate=error_rate, **kw):
                    # Apply base operation with reduced noise
                    _result = base_operation(*a, **kw)

                    # Apply inverse noise correction (simplified model)
                    if _noise_t == "depolarizing":
                        # Correct for depolarizing noise by sharpening distribution
                        _sharpening_factor = 1.0 + rate
                        if isinstance(result, torch.Tensor) and result.dim() >= 2:
                            # Apply sharpening to last dimension (attention weights)
                            _corrected = F.softmax(
                                F.log_softmax(result, _dim=-1) * sharpening_factor,
                                _dim=-1,
                            )
                            return corrected

                    return result

                # Weight for inverse operation (negative for cancellation)
                _weight = -error_rate / (1 - error_rate)
                circuits.append((inverse_noise_op, weight))

        return circuits

    def apply_probabilistic_cancellation(
        self,
        attention_fn: Callable,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply PEC to quantum attention.

        Args:
            attention_fn: Quantum attention function
            Q, K, V: Attention tensors
            **kwargs: Additional arguments

        Returns:
            Error-mitigated attention output and metrics
        """
        # Generate mitigation circuits
        _circuits = self.generate_error_mitigation_circuits(
            attention_fn, Q, K, V, **kwargs
        )

        # Run multiple trials
        _trial_results = []
        _total_weight = 0.0

        for trial in range(self.num_trials):
            # Sample circuit based on probabilities
            _weights = [abs(weight) for _, weight in circuits]
            _total_prob = sum(weights)
            _normalized_probs = [w / total_prob for w in weights]

            _circuit_idx = np.random.choice(len(circuits), p=normalized_probs)
            circuit_op, _circuit_weight = circuits[circuit_idx]

            # Run selected circuit
            _result = circuit_op(Q, K, V, **kwargs)

            # Weight the result
            _weighted_result = result * circuit_weight
            trial_results.append(weighted_result)
            total_weight += circuit_weight

        # Combine results
        if trial_results:
            _mitigated_output = sum(trial_results) / len(trial_results)
            if abs(total_weight) > 1e-8:
                _mitigated_output = mitigated_output / total_weight * len(trial_results)
        else:
            _mitigated_output = attention_fn(Q, K, V, **kwargs)

        _metrics = {
            "num_circuits": len(circuits),
            "num_trials": self.num_trials,
            "total_weight": total_weight,
            "avg_weight_per_trial": (
                total_weight / len(trial_results) if trial_results else 0.0
            ),
        }

        return mitigated_output, metrics


class VirtualDistillation:
    """
    Virtual Distillation for quantum attention error mitigation.

    Runs multiple copies of quantum attention computation and
    uses statistical post-processing to extract improved results.
    """

    def __init__(self, num_copies: int = 5, distillation_method: str = "majority_vote"):
        """
        Initialize virtual distillation.

        Args:
            num_copies: Number of virtual copies to run
            distillation_method: "majority_vote", "median", or "weighted_average"
        """
        self.num_copies = num_copies
        self.distillation_method = distillation_method

    def distill_quantum_attention(
        self,
        attention_fn: Callable,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply virtual distillation to quantum attention.

        Args:
            attention_fn: Quantum attention function
            Q, K, V: Attention tensors
            **kwargs: Additional arguments

        Returns:
            Distilled attention output and metrics
        """
        # Run multiple copies
        _copy_results = []
        _copy_weights = []

        for copy_idx in range(self.num_copies):
            # Run with different random seeds/noise realizations
            torch.manual_seed(42 + copy_idx)  # Different seed per copy

            _result = attention_fn(Q, K, V, **kwargs)
            copy_results.append(result)

            # Compute copy weight based on internal consistency
            if len(copy_results) > 1:
                # Weight based on similarity to previous results
                _similarities = [
                    1.0 / (1.0 + float(torch.norm(result - prev_result)))
                    for prev_result in copy_results[:-1]
                ]
                _weight = sum(similarities) / len(similarities) if similarities else 1.0
            else:
                _weight = 1.0

            copy_weights.append(weight)

        # Apply distillation method
        if self.distillation_method == "majority_vote":
            _distilled_output = self._majority_vote_distillation(copy_results)
        elif self.distillation_method == "median":
            _distilled_output = self._median_distillation(copy_results)
        elif self.distillation_method == "weighted_average":
            _distilled_output = self._weighted_average_distillation(
                copy_results, copy_weights
            )
        else:
            # Fallback to simple average
            _distilled_output = torch.stack(copy_results).mean(dim=0)

        # Compute distillation metrics
        _result_variance = torch.var(torch.stack(copy_results), _dim=0).mean()
        _consensus_score = 1.0 / (1.0 + float(result_variance))

        _metrics = {
            "num_copies": self.num_copies,
            "result_variance": float(result_variance),
            "consensus_score": consensus_score,
            "copy_weights": copy_weights,
            "distillation_method": self.distillation_method,
        }

        return distilled_output, metrics

    def _majority_vote_distillation(self, results: List[torch.Tensor]) -> torch.Tensor:
        """Apply majority vote distillation (for discrete outputs)."""
        # For continuous attention weights, use plurality of closest results
        _result_stack = torch.stack(results)  # (num_copies, ...)

        # Compute pairwise distances
        _n_results = len(results)
        _distances = torch.zeros(n_results, n_results)

        for i in range(n_results):
            for j in range(i + 1, n_results):
                _dist = torch.norm(results[i] - results[j])
                distances[i, j] = distances[j, i] = dist

        # Find result with minimum total distance to others (consensus)
        _total_distances = distances.sum(dim=1)
        _consensus_idx = torch.argmin(total_distances)

        return results[consensus_idx]

    def _median_distillation(self, results: List[torch.Tensor]) -> torch.Tensor:
        """Apply element-wise median distillation."""
        _result_stack = torch.stack(results)  # (num_copies, ...)
        _median_result = torch.median(result_stack, _dim=0)[0]
        return median_result

    def _weighted_average_distillation(
        self, results: List[torch.Tensor], weights: List[float]
    ) -> torch.Tensor:
        """Apply weighted average distillation."""
        _total_weight = sum(weights)
        if _total_weight == 0:
            return torch.stack(results).mean(dim=0)

        _normalized_weights = [w / total_weight for w in weights]

        _weighted_result = torch.zeros_like(results[0])
        for result, weight in zip(results, normalized_weights):
            weighted_result += weight * result

        return weighted_result
