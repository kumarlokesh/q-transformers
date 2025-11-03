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
        noise_array = np.array(noise_levels)

        # Extrapolate each element independently
        result_shape = noisy_results[0].shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = noisy_results[0].dtype

        extrapolated = torch.zeros(result_shape, device=device, dtype=dtype)

        # Flatten for easier processing
        flattened_results = [result.flatten() for result in noisy_results]
        n_elements = flattened_results[0].numel()

        for i in range(n_elements):
            # Extract values for this element across noise levels
            values = np.array([result[i].cpu().item() for result in flattened_results])

            if self.extrapolation_method == "polynomial":
                extrapolated_value = self._polynomial_extrapolation(
                    noise_array, values
                )
            elif self.extrapolation_method == "exponential":
                extrapolated_value = self._exponential_extrapolation(
                    noise_array, values
                )
            elif self.extrapolation_method == "richardson":
                extrapolated_value = self._richardson_extrapolation(
                    noise_array, values
                )
            else:
                raise ValueError(f"Unknown extrapolation method: {self.extrapolation_method}")

            extrapolated.flatten()[i] = extrapolated_value

        return extrapolated.reshape(result_shape)

    def _polynomial_extrapolation(
        self, noise_levels: np.ndarray, values: np.ndarray
    ) -> float:
        """Polynomial extrapolation to zero noise."""
        try:
            # Fit polynomial: f(x) = a0 + a1*x + a2*x^2 + ...
            coeffs = np.polyfit(noise_levels, values, self.polynomial_degree)
            # Extrapolate to x=0 (zero noise): constant term is last in polyfit output
            return float(coeffs[-1])
        except Exception:
            # Fallback to linear extrapolation
            if len(values) >= 2:
                slope = (values[1] - values[0]) / (
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
            p0 = [values[0] - values[-1], 1.0, values[-1]]

            popt, _ = curve_fit(exp_model, noise_levels, values, p0=p0, maxfev=1000)
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
            h1, h2 = noise_levels[0], noise_levels[1]
            f1, f2 = values[0], values[1]

            if abs(h2 - 2 * h1) < 1e-6:  # h2 ≈ 2*h1
                return float(2 * f1 - f2)
            else:
                # General Richardson extrapolation
                ratio = h2 / h1
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
        noisy_results: List[torch.Tensor] = []

        # Compute attention at different noise levels
        for noise_level in self.noise_levels:
            result = attention_fn(Q, K, V, noise_level=noise_level, **kwargs)
            noisy_results.append(result)

        # Extrapolate to zero noise
        mitigated_output = self.extrapolate_zero_noise(noisy_results, self.noise_levels)

        # Compute metrics
        variance_across_noise = torch.var(torch.stack(noisy_results), dim=0).mean()
        extrapolation_confidence = 1.0 / (1.0 + float(variance_across_noise))

        metrics = {
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
        violations: Dict[str, float] = {}

        if "row_normalization" in self.symmetries:
            # Check if attention weights sum to 1 along key dimension
            row_sums = torch.sum(attention_weights, dim=-1)  # (batch, seq_len_q)
            if attention_mask is not None:
                valid_rows = attention_mask.sum(dim=-1) > 0
                row_sums = row_sums[valid_rows]

            normalization_error = torch.mean(torch.abs(row_sums - 1.0))
            violations["row_normalization"] = float(normalization_error)

        if "positivity" in self.symmetries:
            # Check if all attention weights are non-negative
            negative_weights = torch.sum(attention_weights < 0).float()
            total_weights = float(attention_weights.numel())
            violations["positivity"] = float(negative_weights / total_weights)

        if "causality" in self.symmetries and attention_mask is not None:
            # Check causal mask violations (for autoregressive models)
            seq_len = attention_weights.shape[-1]
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(
                attention_weights.device
            )

            if (
                attention_weights.shape[-2] == attention_weights.shape[-1]
            ):  # Square attention
                causal_violations = torch.sum((attention_weights[:, causal_mask] > 1e-6)).float()
                total_causal_positions = causal_mask.sum().float() * attention_weights.shape[0]
                violations["causality"] = float(causal_violations / total_causal_positions)

        if "attention_entropy" in self.symmetries:
            # Check if attention entropy is reasonable (not too peaked or too uniform)
            entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-12), dim=-1
            )  # (batch, seq_len_q)

            max_entropy = math.log(float(attention_weights.shape[-1]))  # Uniform distribution entropy
            normalized_entropy = entropy / (max_entropy + 1e-12)

            # Ideal entropy range: [0.2, 0.8]
            entropy_violations = torch.sum((normalized_entropy < 0.2) | (normalized_entropy > 0.8)).float()
            total_queries = float(entropy.numel())
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
        corrected_weights = attention_weights.clone()

        if "positivity" in self.symmetries:
            # Enforce non-negativity
            corrected_weights = torch.clamp(corrected_weights, min=0.0)

        if "causality" in self.symmetries and attention_mask is not None:
            # Enforce causal mask
            seq_len = corrected_weights.shape[-1]
            if corrected_weights.shape[-2] == corrected_weights.shape[-1]:
                causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(
                    corrected_weights.device
                )
                corrected_weights[:, causal_mask] = 0.0

        if "row_normalization" in self.symmetries:
            # Re-normalize rows to sum to 1
            if attention_mask is not None:
                corrected_weights = corrected_weights.masked_fill(~attention_mask.bool(), 0.0)

            row_sums = torch.sum(corrected_weights, dim=-1, keepdim=True)
            corrected_weights = corrected_weights / (row_sums + 1e-12)

        # Apply correction strength
        if self.correctionstrength < 1.0:
            corrected_weights = (
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
        probs = F.softmax(logits, dim=-1)

        if attention_mask is not None:
            probs = probs.masked_fill(~attention_mask.bool(), 0.0)
            probs = probs / (torch.sum(probs, dim=-1, keepdim=True) + 1e-12)

        batch_size, seq_len_q, seq_len_k = probs.shape
        sampled_weights = torch.zeros_like(probs)

        for b in range(batch_size):
            for q in range(seq_len_q):
                prob_row = probs[b, q, :]

                if prob_row.sum() > 1e-8:
                    # Sample with replacement
                    samples = torch.multinomial(prob_row, num_samples, replacement=True)

                    # Count samples
                    sample_counts = torch.bincount(samples, minlength=seq_len_k)
                    sampled_weights[b, q, :] = sample_counts.float() / float(num_samples)

        # Apply symmetry corrections
        corrected_weights = self.correct_attention_symmetries(sampled_weights, attention_mask)

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
        _circuits: List[Tuple[Callable, float]] = []

        # Identity operation (no additional noise)
        _circuits.append((lambda *a, **kw: base_operation(*a, **kw), 1.0))

        # Inverse operations for error cancellation
        for noise_type, error_rate in self.error_rates.items():
            if error_rate > 0:
                # Create operation that applies inverse of the systematic error
                def inverse_noise_op(*a, _noise_t=noise_type, _rate=error_rate, **kw):
                    # Apply base operation with reduced noise
                    result = base_operation(*a, **kw)

                    # Apply inverse noise correction (simplified model)
                    if _noise_t == "depolarizing":
                        # Correct for depolarizing noise by sharpening distribution
                        sharpening_factor = 1.0 + _rate
                        if isinstance(result, torch.Tensor) and result.dim() >= 2:
                            # Apply sharpening to last dimension (attention weights)
                            corrected = F.softmax(
                                F.log_softmax(result, dim=-1) * sharpening_factor,
                                dim=-1,
                            )
                            return corrected

                    return result

                # Weight for inverse operation (negative for cancellation)
                weight = -error_rate / (1 - error_rate)
                _circuits.append((inverse_noise_op, weight))

        return _circuits

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
        circuits = self.generate_error_mitigation_circuits(attention_fn, Q, K, V, **kwargs)

        # Run multiple trials
        trial_results: List[torch.Tensor] = []
        total_weight = 0.0

        # Precompute sampling probabilities
        weights = [abs(w) for _, w in circuits]
        total_prob = sum(weights) if weights else 0.0
        if total_prob == 0.0:
            normalized_probs = [1.0 / len(circuits) for _ in circuits]
        else:
            normalized_probs = [w / total_prob for w in weights]

        for _ in range(self.num_trials):
            # Sample circuit based on probabilities
            circuit_idx = np.random.choice(len(circuits), p=normalized_probs)
            circuit_op, circuit_weight = circuits[circuit_idx]

            # Run selected circuit
            result = circuit_op(Q, K, V, **kwargs)

            # Weight the result
            weighted_result = result * circuit_weight
            trial_results.append(weighted_result)
            total_weight += circuit_weight

        # Combine results
        if trial_results:
            mitigated_output = sum(trial_results) / len(trial_results)
            if abs(total_weight) > 1e-8:
                mitigated_output = mitigated_output / total_weight * len(trial_results)
        else:
            mitigated_output = attention_fn(Q, K, V, **kwargs)

        metrics = {
            "num_circuits": len(circuits),
            "num_trials": self.num_trials,
            "total_weight": total_weight,
            "avg_weight_per_trial": (total_weight / len(trial_results) if trial_results else 0.0),
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
        copy_results: List[torch.Tensor] = []
        copy_weights: List[float] = []

        for copy_idx in range(self.num_copies):
            # Run with different random seeds/noise realizations
            torch.manual_seed(42 + copy_idx)  # Different seed per copy

            result = attention_fn(Q, K, V, **kwargs)
            copy_results.append(result)

            # Compute copy weight based on internal consistency
            if len(copy_results) > 1:
                # Weight based on similarity to previous results
                similarities = [
                    1.0 / (1.0 + float(torch.norm(result - prev_result)))
                    for prev_result in copy_results[:-1]
                ]
                weight = sum(similarities) / len(similarities) if similarities else 1.0
            else:
                weight = 1.0

            copy_weights.append(weight)

        # Apply distillation method
        if self.distillation_method == "majority_vote":
            distilled_output = self._majority_vote_distillation(copy_results)
        elif self.distillation_method == "median":
            distilled_output = self._median_distillation(copy_results)
        elif self.distillation_method == "weighted_average":
            distilled_output = self._weighted_average_distillation(copy_results, copy_weights)
        else:
            # Fallback to simple average
            distilled_output = torch.stack(copy_results).mean(dim=0)

        # Compute distillation metrics
        result_variance = torch.var(torch.stack(copy_results), dim=0).mean()
        consensus_score = 1.0 / (1.0 + float(result_variance))

        metrics = {
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
        result_stack = torch.stack(results)  # (num_copies, ...)

        # Compute pairwise distances
        n_results = len(results)
        distances = torch.zeros((n_results, n_results), device=result_stack.device)

        for i in range(n_results):
            for j in range(i + 1, n_results):
                d = torch.norm(results[i] - results[j])
                distances[i, j] = distances[j, i] = d

        # Find result with minimum total distance to others (consensus)
        total_distances = distances.sum(dim=1)
        consensus_idx = torch.argmin(total_distances)

        return results[int(consensus_idx)]

    def _median_distillation(self, results: List[torch.Tensor]) -> torch.Tensor:
        """Apply element-wise median distillation."""
        result_stack = torch.stack(results)  # (num_copies, ...)
        median_result = torch.median(result_stack, dim=0)[0]
        return median_result

    def _weighted_average_distillation(
        self, results: List[torch.Tensor], weights: List[float]
    ) -> torch.Tensor:
        """Apply weighted average distillation."""
        total_weight = sum(weights)
        if total_weight == 0:
            return torch.stack(results).mean(dim=0)

        normalized_weights = [w / total_weight for w in weights]

        weighted_result = torch.zeros_like(results[0])
        for result, weight in zip(results, normalized_weights):
            weighted_result += weight * result

        return weighted_result
