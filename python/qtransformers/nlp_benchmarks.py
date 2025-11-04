"""
Real-World NLP Benchmarking Suite for Q-Transformers

Implements comprehensive evaluation on standard NLP tasks:
- GLUE benchmark suite integration
- SuperGLUE advanced evaluation tasks
- Custom quantum-specific NLP metrics
- Comparison with classical transformer baselines
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
)

# Optional dependencies
try:
    import datasets
    from transformers import AutoTokenizer

    _TRANSFORMERS_AVAILABLE = True
except Exception:
    # If either `datasets` or `transformers` is missing, mark unavailable
    _TRANSFORMERS_AVAILABLE = False


@dataclass
class BenchmarkConfig:
    """Configuration for NLP benchmarking."""

    model_name: str
    task_name: str
    max_seq_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    quantum_config: Optional[Dict[str, Any]] = None
    use_cuda: bool = True
    eval_steps: int = 500


class GLUEBenchmarkSuite:
    """
    GLUE (General Language Understanding Evaluation) benchmark suite.

    Supports all 9 GLUE tasks with quantum transformer evaluation.
    """

    _GLUE_TASKS = [
        "cola",  # Corpus of Linguistic Acceptability
        "sst2",  # Stanford Sentiment Treebank
        "mrpc",  # Microsoft Research Paraphrase Corpus
        "stsb",  # Semantic Textual Similarity Benchmark
        "qqp",  # Quora Question Pairs
        "mnli",  # Multi-Genre Natural Language Inference
        "qnli",  # Question Natural Language Inference
        "rte",  # Recognizing Textual Entailment
        "wnli",  # Winograd Natural Language Inference
    ]

    def __init__(self, cache_dir: str = "./glue_cache"):
        """Initialize GLUE benchmark suite."""
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and datasets required for GLUE benchmarks")

        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Task configurations
        self.task_configs = {
            "cola": {"num_labels": 2, "metric": "matthews_corrcoef"},
            "sst2": {"num_labels": 2, "metric": "accuracy"},
            "mrpc": {"num_labels": 2, "metric": "f1"},
            "stsb": {
                "num_labels": 1,
                "metric": "pearson_spearman",
                "is_regression": True,
            },
            "qqp": {"num_labels": 2, "metric": "f1"},
            "mnli": {"num_labels": 3, "metric": "accuracy"},
            "qnli": {"num_labels": 2, "metric": "accuracy"},
            "rte": {"num_labels": 2, "metric": "accuracy"},
            "wnli": {"num_labels": 2, "metric": "accuracy"},
        }

        self.tokenizer = None
        self.datasets = {}

    def load_task_data(self, task_name: str) -> Dict[str, Any]:
        """Load GLUE task dataset."""
        if task_name not in self._GLUE_TASKS:
            raise ValueError(f"Unknown GLUE task: {task_name}")

        print(f"Loading GLUE task: {task_name}")

        # Load dataset (datasets.load_dataset returns a DatasetDict)
        dataset = datasets.load_dataset("glue", task_name, cache_dir=self.cache_dir)

        # MNLI has separate validation sets; normalize to 'validation'
        if "validation_matched" in dataset:
            dataset["validation"] = dataset["validation_matched"]

        self.datasets[task_name] = dataset
        return dataset

    def prepare_tokenizer(self, model_name: str = "bert-base-uncased"):
        """Initialize tokenizer for text preprocessing."""
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required to prepare tokenizer")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return self.tokenizer

    def preprocess_data(
        self, task_name: str, max_length: int = 512
    ) -> Dict[str, DataLoader]:
        """Preprocess and tokenize task data."""
        if task_name not in self.datasets:
            self.load_task_data(task_name)

        if self.tokenizer is None:
            self.prepare_tokenizer()
        dataset = self.datasets[task_name]

        def tokenize_function(examples):
            # Sentence-pair tasks
            if task_name in [
                "mrpc",
                "qqp",
                "stsb",
                "mnli",
                "qnli",
                "rte",
                "wnli",
            ]:
                return self.tokenizer(
                    examples.get("sentence1"),
                    examples.get("sentence2"),
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )

            # Single-sentence tasks
            sentence_key = "sentence" if task_name in ["cola", "sst2"] else "question"
            return self.tokenizer(
                examples.get(sentence_key),
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Create data loaders (use reasonable defaults; keep small and safe)
        train_loader = DataLoader(
            tokenized_dataset["train"], batch_size=16, shuffle=True
        )

        val_split = "validation"  # normalized earlier in load_task_data
        val_loader = DataLoader(
            tokenized_dataset[val_split], batch_size=16, shuffle=False
        )

        return {"train": train_loader, "validation": val_loader}

    def compute_metrics(
        self, task_name: str, predictions: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute task-specific metrics."""
        task_config = self.task_configs[task_name]
        metric_name = task_config["metric"]

        results = {}

        if metric_name == "accuracy":
            results["accuracy"] = accuracy_score(labels, predictions)

        elif metric_name == "f1":
            results["f1"] = f1_score(
                labels,
                predictions,
                average=("binary" if task_config["num_labels"] == 2 else "macro"),
            )
            results["accuracy"] = accuracy_score(labels, predictions)

        elif metric_name == "matthews_corrcoef":
            results["matthews_corrcoef"] = matthews_corrcoef(labels, predictions)

        elif metric_name == "pearson_spearman" and task_config.get(
            "is_regression", False
        ):
            from scipy.stats import pearsonr, spearmanr

            pearson_r, _ = pearsonr(predictions, labels)
            spearman_r, _ = spearmanr(predictions, labels)
            results["pearson"] = pearson_r
            results["spearman"] = spearman_r
            results["combined"] = (pearson_r + spearman_r) / 2

        return results

    def evaluate_quantum_model(
        self,
        model: nn.Module,
        task_name: str,
        data_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """Evaluate quantum transformer on GLUE task."""
        model.eval()
        all_predictions = []
        all_labels = []
        total_time = 0.0

        with torch.no_grad():
            for batch in data_loader:
                start_time = time.time()

                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch.get("labels", batch.get("label"))
                if labels is not None and hasattr(labels, "to"):
                    labels = labels.to(device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                if self.task_configs[task_name].get("is_regression", False):
                    predictions = outputs.squeeze().cpu().numpy()
                else:
                    predictions = torch.argmax(outputs, dim=-1).cpu().numpy()

                all_predictions.extend(predictions)
                if labels is not None:
                    all_labels.extend(labels.cpu().numpy())

                total_time += time.time() - start_time

        # Compute metrics
        metrics = self.compute_metrics(
            task_name, np.array(all_predictions), np.array(all_labels)
        )
        metrics["inference_time_ms"] = (total_time / max(1, len(data_loader))) * 1000

        return metrics


class SuperGLUEBenchmarkSuite:
    """
    SuperGLUE benchmark suite for advanced language understanding.

    More challenging tasks requiring sophisticated reasoning.
    """

    _SUPERGLUE_TASKS = [
        "boolq",  # Boolean Questions
        "cb",  # CommitmentBank
        "copa",  # Choice of Plausible Alternatives
        "multirc",  # Multi-Sentence Reading Comprehension
        "record",  # Reading Comprehension with Commonsense Reasoning
        "rte",  # Recognizing Textual Entailment
        "wic",  # Words in Context
        "wsc",  # Winograd Schema Challenge
    ]

    def __init__(self, cache_dir: str = "./superglue_cache"):
        """Initialize SuperGLUE benchmark suite."""
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and datasets required for SuperGLUE benchmarks"
            )

        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.task_configs = {
            "boolq": {"num_labels": 2, "metric": "accuracy"},
            "cb": {"num_labels": 3, "metric": "f1_macro"},
            "copa": {"num_labels": 2, "metric": "accuracy"},
            "multirc": {"num_labels": 2, "metric": "f1_and_exact_match"},
            "record": {"metric": "f1_and_exact_match"},
            "rte": {"num_labels": 2, "metric": "accuracy"},
            "wic": {"num_labels": 2, "metric": "accuracy"},
            "wsc": {"num_labels": 2, "metric": "accuracy"},
        }

        self.tokenizer = None
        self.datasets = {}

    def load_task_data(self, task_name: str) -> Dict[str, Any]:
        """Load SuperGLUE task dataset."""
        if task_name not in self._SUPERGLUE_TASKS:
            raise ValueError(f"Unknown SuperGLUE task: {task_name}")

        print(f"Loading SuperGLUE task: {task_name}")
        dataset = datasets.load_dataset(
            "super_glue", task_name, cache_dir=self.cache_dir
        )
        self.datasets[task_name] = dataset
        return dataset

    def evaluate_reasoning_capabilities(
        self,
        quantum_model: nn.Module,
        classical_model: nn.Module,
        task_name: str = "copa",
    ) -> Dict[str, Any]:
        """
        Evaluate quantum vs classical reasoning on challenging tasks.

        Focus on tasks that may benefit from quantum attention patterns.
        """
        if task_name not in self.datasets:
            self.load_task_data(task_name)

        # Prepare data
        data_loaders = self.preprocess_data(task_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        results: Dict[str, Any] = {}

        # Evaluate quantum model
        print(f"Evaluating quantum model on {task_name}...")
        quantum_metrics = self.evaluate_model(
            quantum_model, task_name, data_loaders["validation"], device
        )
        results["quantum"] = quantum_metrics

        # Evaluate classical model
        print(f"Evaluating classical model on {task_name}...")
        classical_metrics = self.evaluate_model(
            classical_model, task_name, data_loaders["validation"], device
        )
        results["classical"] = classical_metrics

        # Compute quantum advantage
        primary_metric = list(quantum_metrics.keys())[0]
        quantum_score = quantum_metrics[primary_metric]
        classical_score = classical_metrics[primary_metric]

        results["quantum_advantage"] = {
            "absolute_improvement": quantum_score - classical_score,
            "relative_improvement": (quantum_score - classical_score)
            / (classical_score + 1e-12)
            * 100,
            "statistical_significance": self._compute_significance(
                quantum_score, classical_score
            ),
        }

        return results

    def evaluate_model(
        self,
        model: nn.Module,
        task_name: str,
        data_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """Evaluate model on SuperGLUE task."""
        # Similar to GLUE evaluation but adapted for SuperGLUE tasks
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch.get("labels", batch.get("label"))
                if labels is not None and hasattr(labels, "to"):
                    labels = labels.to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs, dim=-1).cpu().numpy()

                all_predictions.extend(predictions)
                if labels is not None:
                    all_labels.extend(labels.cpu().numpy())

        # Compute task-specific metrics
        return self._compute_superglue_metrics(
            task_name, np.array(all_predictions), np.array(all_labels)
        )

    def _compute_superglue_metrics(
        self, task_name: str, predictions: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute SuperGLUE task-specific metrics."""
        task_config = self.task_configs[task_name]
        metric_name = task_config["metric"]

        results: Dict[str, float] = {}

        if metric_name == "accuracy":
            results["accuracy"] = accuracy_score(labels, predictions)

        elif metric_name == "f1_macro":
            results["f1_macro"] = f1_score(labels, predictions, average="macro")
            results["accuracy"] = accuracy_score(labels, predictions)

        elif metric_name == "f1_and_exact_match":
            results["f1"] = f1_score(labels, predictions, average="macro")
            results["exact_match"] = accuracy_score(labels, predictions)

        return results

    def _compute_significance(
        self, score1: float, score2: float, n_samples: int = 1000
    ) -> float:
        """Compute statistical significance of score difference."""
        # Simplified significance test
        # In practice, would use proper statistical tests
        difference = abs(score1 - score2)
        std_error = np.sqrt(
            (score1 * (1 - score1) + score2 * (1 - score2)) / max(1, n_samples)
        )
        z_score = difference / (std_error + 1e-8)

        # Convert to p-value approximation
        from scipy.stats import norm

        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        return p_value


class QuantumAdvantageAnalyzer:
    """
    Analyzer for detecting and measuring quantum advantage in NLP tasks.

    Provides statistical analysis and visualization of quantum vs classical performance.
    """

    def __init__(self):
        self.results_history = []

    def analyze_attention_patterns(
        self,
        quantum_model: nn.Module,
        classical_model: nn.Module,
        input_texts: List[str],
        task_type: str = "classification",
    ) -> Dict[str, Any]:
        """
        Analyze differences in attention patterns between quantum and classical models.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get attention weights from both models
        quantum_attentions: List[np.ndarray] = []
        # classical_attentions reserved for future use if classical models
        # expose attention weights; keep out to avoid unused-variable lints

        tokenizer = (
            AutoTokenizer.from_pretrained("bert-base-uncased")
            if _TRANSFORMERS_AVAILABLE
            else None
        )

        for text in input_texts:
            # Tokenize input
            if tokenizer is None:
                raise ImportError(
                    "transformers is required to analyze attention patterns"
                )

            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # Get quantum attention
            with torch.no_grad():
                quantum_out = quantum_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_attention_weights=True,
                )
                # try to extract attention weights if returned
                if isinstance(quantum_out, tuple) and len(quantum_out) >= 2:
                    _, quantum_attn = quantum_out
                else:
                    quantum_attn = None

                if quantum_attn is not None:
                    quantum_attentions.append(quantum_attn.cpu().numpy())

            # Get classical attention (if available) -- left as optional
            # if classical_model returns attention weights, append them similarly

        # Analyze attention pattern differences
        analysis = {
            "attention_entropy": self._compute_attention_entropy(quantum_attentions),
            "attention_sparsity": self._compute_attention_sparsity(quantum_attentions),
            "pattern_diversity": self._compute_pattern_diversity(quantum_attentions),
            "task_specific_focus": self._analyze_task_focus(
                quantum_attentions, task_type
            ),
        }

        return analysis

    def _compute_attention_entropy(
        self, attention_matrices: List[np.ndarray]
    ) -> Dict[str, float]:
        """Compute entropy of attention distributions."""
        entropies: List[float] = []

        if not attention_matrices:
            return {
                "mean_entropy": 0.0,
                "std_entropy": 0.0,
                "min_entropy": 0.0,
                "max_entropy": 0.0,
            }

        for attn_matrix in attention_matrices:
            # Compute entropy for each attention head
            head_entropies: List[float] = []
            # Assuming shape: [batch, heads, seq, seq]
            n_heads = attn_matrix.shape[1] if attn_matrix.ndim >= 4 else 1
            for head in range(n_heads):
                attn_dist = attn_matrix[0, head, :, :]
                entropy = -np.sum(attn_dist * np.log(attn_dist + 1e-12), axis=-1)
                head_entropies.append(float(np.mean(entropy)))

            entropies.append(float(np.mean(head_entropies)))

        ent_arr = np.array(entropies)
        return {
            "mean_entropy": float(np.mean(ent_arr)),
            "std_entropy": float(np.std(ent_arr)),
            "min_entropy": float(np.min(ent_arr)),
            "max_entropy": float(np.max(ent_arr)),
        }

    def _compute_attention_sparsity(
        self, attention_matrices: List[np.ndarray]
    ) -> Dict[str, float]:
        """Compute sparsity of attention patterns."""
        sparsities: List[float] = []

        if not attention_matrices:
            return {"mean_sparsity": 0.0, "std_sparsity": 0.0}

        for attn_matrix in attention_matrices:
            # Count near-zero attention weights
            threshold = 0.01
            total_weights = attn_matrix.size
            sparse_weights = np.sum(attn_matrix < threshold)
            sparsity = float(sparse_weights) / float(max(1, total_weights))
            sparsities.append(sparsity)

        spars_arr = np.array(sparsities)
        return {
            "mean_sparsity": float(np.mean(spars_arr)),
            "std_sparsity": float(np.std(spars_arr)),
        }

    def _compute_pattern_diversity(self, attention_matrices: List[np.ndarray]) -> float:
        """Compute diversity of attention patterns across examples."""
        if len(attention_matrices) < 2:
            return 0.0

        # Compute pairwise cosine similarities
        similarities: List[float] = []
        for i in range(len(attention_matrices)):
            for j in range(i + 1, len(attention_matrices)):
                attn1 = attention_matrices[i].flatten()
                attn2 = attention_matrices[j].flatten()

                # Cosine similarity
                dot_product = np.dot(attn1, attn2)
                norm_product = np.linalg.norm(attn1) * np.linalg.norm(attn2)
                similarity = dot_product / (norm_product + 1e-12)
                similarities.append(similarity)

        # Diversity is 1 - mean similarity
        return 1.0 - np.mean(similarities)

    def _analyze_task_focus(
        self, attention_matrices: List[np.ndarray], task_type: str
    ) -> Dict[str, float]:
        """Analyze task-specific attention focus patterns."""
        # This would be task-specific analysis
        # For now, return general focus metrics

        focus_metrics: Dict[str, float] = {}

        for i, attn_matrix in enumerate(attention_matrices):
            # Compute attention concentration on different token types
            seq_len = attn_matrix.shape[-1]

            # Focus on beginning tokens (often important for classification)
            beginning_focus = np.mean(attn_matrix[:, :, :, : min(3, seq_len)])

            # Focus on end tokens
            end_focus = np.mean(attn_matrix[:, :, :, max(0, seq_len - 3) :])

            focus_metrics[f"example_{i}_beginning_focus"] = float(beginning_focus)
            focus_metrics[f"example_{i}_end_focus"] = float(end_focus)

        return focus_metrics

    def generate_advantage_report(
        self,
        benchmark_results: Dict[str, Any],
        output_path: str = "quantum_advantage_report.json",
    ) -> Dict[str, Any]:
        """Generate comprehensive quantum advantage analysis report."""

        report = {
            "summary": {
                "total_tasks_evaluated": len(benchmark_results),
                "tasks_with_quantum_advantage": 0,
                "average_improvement": 0.0,
                "statistical_significance": 0.0,
            },
            "task_results": benchmark_results,
            "recommendations": [],
            "future_work": [],
        }

        # Analyze results
        improvements: List[float] = []
        significances: List[float] = []

        for task_name, results in benchmark_results.items():
            if "quantum_advantage" in results:
                advantage = results["quantum_advantage"]
                improvement = advantage.get("relative_improvement", 0)
                significance = advantage.get("statistical_significance", 1.0)

                improvements.append(improvement)
                significances.append(significance)

                if improvement > 0 and significance < 0.05:
                    report["summary"]["tasks_with_quantum_advantage"] += 1

        if improvements:
            report["summary"]["average_improvement"] = float(np.mean(improvements))
            report["summary"]["statistical_significance"] = float(
                np.mean(significances)
            )

        # Generate recommendations
        if report["summary"]["tasks_with_quantum_advantage"] > 0:
            report["recommendations"].append(
                "Quantum advantage demonstrated on specific NLP tasks. "
                "Focus development on these high-impact areas."
            )
        else:
            report["recommendations"].append(
                "No clear quantum advantage detected. "
                "Consider optimizing quantum sampling strategies or "
                "trying different task types."
            )

        # Save report
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        return report


class NLPBenchmarkRunner:
    """
    Main runner for comprehensive NLP benchmarking of quantum transformers.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.glue_suite = GLUEBenchmarkSuite()
        self.superglue_suite = SuperGLUEBenchmarkSuite()
        self.advantage_analyzer = QuantumAdvantageAnalyzer()

    def run_comprehensive_benchmark(
        self,
        quantum_model: nn.Module,
        classical_baseline: Optional[nn.Module] = None,
        tasks: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark on quantum transformer.

        Args:
            quantum_model: Quantum transformer model to evaluate
            classical_baseline: Classical transformer for comparison
            tasks: List of tasks to evaluate (default: all GLUE tasks)

        Returns:
            Comprehensive benchmark results
        """
        if tasks is None:
            tasks = ["sst2", "mrpc", "cola", "qnli"]  # Subset for faster evaluation

        results: Dict[str, Any] = {
            "config": {
                "model_name": self.config.model_name,
                "tasks": tasks,
                "quantum_config": self.config.quantum_config,
            },
            "glue_results": {},
            "superglue_results": {},
            "quantum_advantage_analysis": {},
        }

        device = torch.device(
            "cuda" if self.config.use_cuda and torch.cuda.is_available() else "cpu"
        )
        quantum_model.to(device)

        # Run GLUE benchmarks
        print("🧪 Running GLUE benchmarks...")
        for task in tasks:
            if task in self.glue_suite._GLUE_TASKS:
                print(f"  Evaluating on {task}...")

                try:
                    data_loaders = self.glue_suite.preprocess_data(
                        task, self.config.max_seq_length
                    )
                    task_results = self.glue_suite.evaluate_quantum_model(
                        quantum_model, task, data_loaders["validation"], device
                    )
                    results["glue_results"][task] = task_results

                except Exception as e:
                    print(f"    Error evaluating {task}: {e}")
                    results["glue_results"][task] = {"error": str(e)}

        # Run quantum advantage analysis if classical baseline provided
        if classical_baseline is not None:
            print("⚡ Analyzing quantum advantage...")
            classical_baseline.to(device)

            # Compare on a challenging SuperGLUE task
            copa_results = self.superglue_suite.evaluate_reasoning_capabilities(
                quantum_model, classical_baseline, "copa"
            )
            results["quantum_advantage_analysis"]["copa"] = copa_results

        return results

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save benchmark results to file."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        # use f-string so the path is interpolated correctly
        print(f"📊 Benchmark results saved to: {output_path}")


# Factory functions for easy usage
def create_glue_benchmark(cache_dir: str = "./glue_cache") -> GLUEBenchmarkSuite:
    """Create GLUE benchmark suite."""
    return GLUEBenchmarkSuite(cache_dir)


def create_quantum_nlp_benchmark(
    model_name: str = "quantum-bert-base",
    quantum_config: Optional[Dict[str, Any]] = None,
) -> NLPBenchmarkRunner:
    """Create comprehensive NLP benchmark runner."""

    if quantum_config is None:
        quantum_config = {
            "backend": "prototype",
            "num_samples": 32,
            "use_advanced_sampling": True,
            "use_error_mitigation": True,
        }

    config = BenchmarkConfig(
        model_name=model_name, task_name="comprehensive", quantum_config=quantum_config
    )

    return NLPBenchmarkRunner(config)
