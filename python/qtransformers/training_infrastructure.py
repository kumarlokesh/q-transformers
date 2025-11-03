"""
Large-Scale Training Infrastructure for Quantum Transformers

Training system with:
- Distributed quantum attention across multiple GPUs
- Scalable data pipeline for large NLP datasets
- Mixed-precision training with quantum-aware optimizations
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .memory_profiler import MemoryProfiler
from .quantum_transformer_blocks import (
    QuantumTransformerBlock,
    ScalableQuantumTransformer,
)


@dataclass
class TrainingConfig:
    """Configuration for large-scale quantum transformer training."""

    # Model architecture
    model_name: str = "quantum-bert-large"
    vocab_size: int = 30522
    max_position_embeddings: int = 512
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096

    # Quantum configuration
    quantum_config: Dict[str, Any] = field(
        _default_factory=lambda: {
            "backend": "prototype",
            "num_samples": 64,
            "use_advanced_sampling": True,
            "use_error_mitigation": True,
            "use_gpu_acceleration": True,
        }
    )

    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_steps: int = 100000
    warmup_steps: int = 10000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Distributed training
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    backend: str = "nccl"

    # Mixed precision
    use_amp: bool = True
    amp_opt_level: str = "O2"

    # Checkpointing
    save_steps: int = 5000
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None

    # Logging and evaluation
    logging_steps: int = 100
    eval_steps: int = 1000
    save_total_limit: int = 3

    # Data
    train_data_path: str = ""
    eval_data_path: str = ""
    max_seq_length: int = 512

    # Optimization
    optimizer_type: str = "adamw"
    scheduler_type: str = "linear_warmup"
    dataloader_num_workers: int = 4

    # Quantum-specific optimizations
    quantum_gradient_clipping: bool = True
    quantum_noise_schedule: str = "linear_decay"
    adaptive_sampling_schedule: bool = True


class QuantumDataCollator:
    """
    Data collator optimized for quantum transformer training.

    Handles batching with quantum-aware padding and attention masks.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        pad_to_multiple_of: Optional[int] = None,
        quantum_paddingstrategy: str = "attention_aware",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.quantum_paddingstrategy = quantum_paddingstrategy

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch with quantum-optimized padding.

        Args:
            features: List of input features

        Returns:
            Batched and padded tensors
        """
        _batch = {}

        # Standard tokenizer batching
        if "input_ids" in features[0]:
            # Text data
            _input_ids = [f["input_ids"] for f in features]
            _attention_mask = [f.get("attention_mask", None) for f in features]

            # Quantum-aware padding
            if self.quantum_paddingstrategy == "attention_aware":
                _batch = self._quantum_aware_padding(input_ids, attention_mask)
            else:
                _batch = self._standard_padding(input_ids, attention_mask)

        # Add labels if present
        if "labels" in features[0]:
            _labels = [f["labels"] for f in features]
            batch["labels"] = torch.tensor(labels, _dtype=torch.long)

        return batch

    def _quantum_aware_padding(
        self, input_ids: List[List[int]], attention_masks: List[Optional[List[int]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Quantum-aware padding that optimizes for attention computation.

        Reduces quantum sampling overhead by minimizing padding tokens.
        """
        # Find optimal batch length (minimize quantum computation on padding)
        _lengths = [len(ids) for ids in input_ids]
        _max_len = max(lengths)

        # Round to efficient quantum sampling size
        if self.pad_to_multiple_of:
            _max_len = (
                (max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of

        # Limit to max_length
        _max_len = min(max_len, self.max_length)

        # Pad sequences
        _padded_input_ids = []
        _padded_attention_masks = []

        for i, ids in enumerate(input_ids):
            # Truncate if necessary
            if len(ids) > max_len:
                _ids = ids[:max_len]

            # Pad to max_len
            _padded_ids = ids + [self.tokenizer.pad_token_id] * (max_len - len(ids))
            padded_input_ids.append(padded_ids)

            # Attention mask
            if attention_masks[i] is not None:
                _mask = attention_masks[i][: len(ids)] + [0] * (max_len - len(ids))
            else:
                _mask = [1] * len(ids) + [0] * (max_len - len(ids))
            padded_attention_masks.append(mask)

        return {
            "input_ids": torch.tensor(padded_input_ids, _dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, _dtype=torch.long),
        }

    def _standard_padding(
        self, input_ids: List[List[int]], attention_masks: List[Optional[List[int]]]
    ) -> Dict[str, torch.Tensor]:
        """Standard padding for comparison."""
        # Use tokenizer's built-in padding
        return self.tokenizer.pad(
            {"input_ids": input_ids},
            _padding=True,
            _max_length=self.max_length,
            _return_tensors="pt",
        )


class QuantumTrainer:
    """
    Trainer for large-scale quantum transformer models.

    Supports distributed training, mixed precision, and quantum-specific optimizations.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        tokenizer: Optional[Any] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = 0.0

        # Setup logging
        self.setup_logging()

        # Initialize distributed training if needed
        if config.world_size > 1:
            self.setup_distributed()

        # Setup device and model
        self.device = torch.device(
            "cuda:{config.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        # Wrap model for distributed training
        if config.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                _device_ids=[config.local_rank],
                _output_device=config.local_rank,
                _find_unused_parameters=True,  # For quantum attention modules
            )

        # Initialize optimizer and scheduler
        self.setup_optimization()

        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_amp else None

        # Memory profiler
        self.memory_profiler = MemoryProfiler()

        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, _exist_ok=True)

        # Resume from checkpoint if specified
        if config.resume_from_checkpoint:
            self.resume_from_checkpoint(config.resume_from_checkpoint)

    def setup_logging(self):
        """Setup logging infrastructure."""
        logging.basicConfig(
            _format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            _datefmt="%m/%d/%Y %H:%M:%S",
            _level=logging.INFO if self.config.local_rank in [-1, 0] else logging.WARN,
        )

        # Initialize wandb if rank 0
        if self.config.local_rank in [-1, 0]:
            wandb.init(
                _project="quantum-transformers",
                _config=self.config,
                _name="{self.config.model_name}-{time.strftime('%Y%m%d-%H%M%S')}",
            )

    def setup_distributed(self):
        """Setup distributed training."""
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = self.config.master_port

        dist.init_process_group(
            _backend=self.config.backend,
            _rank=self.config.rank,
            _world_size=self.config.world_size,
        )

        torch.cuda.set_device(self.config.local_rank)

    def setup_optimization(self):
        """Setup optimizer and learning rate scheduler."""

        # Separate quantum and classical parameters for different optimization
        _quantum_params = []
        _classical_params = []

        for name, param in self.model.named_parameters():
            if "quantum" in name.lower() or "attention" in name.lower():
                quantum_params.append(param)
            else:
                classical_params.append(param)

        # Different learning rates for quantum vs classical components
        _param_groups = [
            {"params": classical_params, "lr": self.config.learning_rate},
            {
                "params": quantum_params,
                "lr": self.config.learning_rate * 0.5,
            },  # Lower LR for quantum
        ]

        # Optimizer
        if self.config.optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                param_groups,
                _lr=self.config.learning_rate,
                _weight_decay=self.config.weight_decay,
                _eps=1e-6,  # More stable for quantum computations
            )
        elif self.config.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                param_groups,
                _lr=self.config.learning_rate,
                _momentum=0.9,
                _weight_decay=self.config.weight_decay,
            )

        # Learning rate scheduler
        if self.config.scheduler_type == "linear_warmup":
            from transformers import get_linear_schedule_with_warmup

            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                _num_warmup_steps=self.config.warmup_steps,
                _num_training_steps=self.config.max_steps,
            )
        elif self.config.scheduler_type == "cosine":
            from transformers import get_cosine_schedule_with_warmup

            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                _num_warmup_steps=self.config.warmup_steps,
                _num_training_steps=self.config.max_steps,
            )

    def get_train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        if self.train_dataset is None:
            raise ValueError("No training dataset provided")

        # Data collator
        _data_collator = QuantumDataCollator(
            _tokenizer=self.tokenizer,
            _max_length=self.config.max_seq_length,
            _pad_to_multiple_of=8,  # Optimize for tensor cores
            _quantum_paddingstrategy="attention_aware",
        )

        # Distributed sampler
        _sampler = None
        if self.config.world_size > 1:
            _sampler = DistributedSampler(
                self.train_dataset,
                _num_replicas=self.config.world_size,
                _rank=self.config.rank,
                _shuffle=True,
            )

        return DataLoader(
            self.train_dataset,
            _batch_size=self.config.batch_size,
            _sampler=sampler,
            _shuffle=(sampler is None),
            _collate_fn=data_collator,
            _num_workers=self.config.dataloader_num_workers,
            _pin_memory=True,
            _drop_last=True,
        )

    def get_eval_dataloader(self) -> Optional[DataLoader]:
        """Create evaluation data loader."""
        if self.eval_dataset is None:
            return None

        _data_collator = QuantumDataCollator(
            _tokenizer=self.tokenizer,
            _max_length=self.config.max_seq_length,
            _quantum_paddingstrategy="attention_aware",
        )

        return DataLoader(
            self.eval_dataset,
            _batch_size=self.config.batch_size,
            _shuffle=False,
            _collate_fn=data_collator,
            _num_workers=self.config.dataloader_num_workers,
            _pin_memory=True,
        )

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute single training step."""
        self.model.train()

        # Move batch to device
        _batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Forward pass with mixed precision
        if self.config.use_amp:
            with autocast():
                _outputs = self.model(**batch)
                _loss = outputs.loss if hasattr(outputs, "loss") else outputs
        else:
            _outputs = self.model(**batch)
            _loss = outputs.loss if hasattr(outputs, "loss") else outputs

        # Scale loss for gradient accumulation
        _loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.config.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.detach()

    def train(self):
        """Main training loop."""

        logging.info("***** Running training *****")
        logging.info(
            "  Num _examples = {len(self.train_dataset) if self.train_dataset else 'Unknown'}"
        )
        logging.info(
            "  Num _Epochs = {self.config.max_steps // len(self.get_train_dataloader())}"
        )
        logging.info("  Batch _size = {self.config.batch_size}")
        logging.info(
            "  Gradient Accumulation _steps = {self.config.gradient_accumulation_steps}"
        )
        logging.info("  Total optimization _steps = {self.config.max_steps}")

        _train_dataloader = self.get_train_dataloader()
        _eval_dataloader = self.get_eval_dataloader()

        # Training metrics
        _total_loss = 0.0
        _logging_loss = 0.0

        # Start memory profiling
        self.memory_profiler.start_profiling()

        # Training loop
        for epoch in range(1000):  # Large number, will be stopped by max_steps
            if self.config.world_size > 1:
                train_dataloader.sampler.set_epoch(epoch)

            for step, batch in enumerate(train_dataloader):
                # Training step
                _step_loss = self.training_step(batch)
                total_loss += step_loss.item()

                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Quantum-aware gradient clipping
                    if self.config.quantum_gradient_clipping:
                        if self.config.use_amp:
                            self.scaler.unscale_(self.optimizer)

                        # Separate clipping for quantum vs classical parameters
                        _quantum_params = [
                            p
                            for name, p in self.model.named_parameters()
                            if "quantum" in name.lower() or "attention" in name.lower()
                        ]
                        _classical_params = [
                            p
                            for name, p in self.model.named_parameters()
                            if not (
                                "quantum" in name.lower() or "attention" in name.lower()
                            )
                        ]

                        if quantum_params:
                            torch.nn.utils.clip_grad_norm_(
                                quantum_params, self.config.max_grad_norm * 0.5
                            )
                        if classical_params:
                            torch.nn.utils.clip_grad_norm_(
                                classical_params, self.config.max_grad_norm
                            )

                    # Optimizer step
                    if self.config.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        _avg_loss = (
                            total_loss - logging_loss
                        ) / self.config.logging_steps
                        _learning_rate = self.scheduler.get_last_lr()[0]

                        _logs = {
                            "loss": avg_loss,
                            "learning_rate": learning_rate,
                            "step": self.global_step,
                            "epoch": epoch,
                        }

                        # Memory usage
                        if torch.cuda.is_available():
                            logs["gpu_memory_mb"] = (
                                torch.cuda.memory_allocated() / 1024**2
                            )

                        if self.config.local_rank in [-1, 0]:
                            wandb.log(logs)
                            logging.info("Step {self.global_step}: {logs}")

                        _logging_loss = total_loss

                    # Evaluation
                    if (
                        eval_dataloader
                        and self.global_step % self.config.eval_steps == 0
                    ):
                        _eval_results = self.evaluate(eval_dataloader)

                        if self.config.local_rank in [-1, 0]:
                            wandb.log(eval_results)
                            logging.info("Eval results: {eval_results}")

                        # Save best model
                        _eval_metric = eval_results.get(
                            "eval_accuracy", eval_results.get("eval_f1", 0)
                        )
                        if eval_metric > self.best_metric:
                            self.best_metric = eval_metric
                            self.save_checkpoint("best")

                    # Checkpointing
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint("step-{self.global_step}")

                    # Stop condition
                    if self.global_step >= self.config.max_steps:
                        break

            if self.global_step >= self.config.max_steps:
                break

        # Final evaluation and save
        if eval_dataloader:
            _final_results = self.evaluate(eval_dataloader)
            logging.info("Final results: {final_results}")

        self.save_checkpoint("final")

        # Stop memory profiling
        _memory_report = self.memory_profiler.stop_profiling()
        logging.info("Memory usage report: {memory_report}")

        logging.info("Training completed!")

    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Run evaluation on the provided dataloader."""
        self.model.eval()

        _eval_loss = 0.0
        _eval_steps = 0
        _all_predictions = []
        _all_labels = []

        with torch.no_grad():
            for batch in eval_dataloader:
                _batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                if self.config.use_amp:
                    with autocast():
                        _outputs = self.model(**batch)
                else:
                    _outputs = self.model(**batch)

                if hasattr(outputs, "loss"):
                    eval_loss += outputs.loss.item()

                # Collect predictions for metrics
                if hasattr(outputs, "logits") and "labels" in batch:
                    _predictions = torch.argmax(outputs.logits, _dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(batch["labels"].cpu().numpy())

                eval_steps += 1

        # Compute metrics
        _results = {"eval_loss": eval_loss / eval_steps}

        if all_predictions and self.compute_metrics:
            _metric_results = self.compute_metrics((all_predictions, all_labels))
            results.update({"eval_{k}": v for k, v in metric_results.items()})
        elif all_predictions:
            # Basic accuracy
            _accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
            results["eval_accuracy"] = accuracy

        return results

    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        if self.config.local_rank not in [-1, 0]:
            return

        _checkpoint_path = self.checkpoint_dir / "{checkpoint_name}.pt"

        # Get model state dict
        _model_state_dict = (
            self.model.module.state_dict()
            if hasattr(self.model, "module")
            else self.model.state_dict()
        )

        _checkpoint = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "config": self.config,
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logging.info("Checkpoint saved: {checkpoint_path}")

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        logging.info("Loading checkpoint from {checkpoint_path}")

        _checkpoint = torch.load(checkpoint_path, _map_location=self.device)

        # Load model state
        _model_state_dict = checkpoint["model_state_dict"]
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)

        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_metric = checkpoint.get("best_metric", 0.0)

        # Load scaler if using AMP
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logging.info("Resumed from step {self.global_step}")

    def _cleanup_checkpoints(self):
        """Clean up old checkpoints keeping only the most recent ones."""
        _checkpoint_files = list(self.checkpoint_dir.glob("step-*.pt"))
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, _reverse=True)

        # Keep only the most recent checkpoints
        for old_checkpoint in checkpoint_files[self.config.save_total_limit :]:
            old_checkpoint.unlink()


def create_quantum_trainer(
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    train_dataset: torch.utils.data.Dataset,
    eval_dataset: Optional[torch.utils.data.Dataset] = None,
    tokenizer: Optional[Any] = None,
) -> QuantumTrainer:
    """
    Factory function to create a quantum transformer trainer.

    Args:
        model_config: Configuration for the quantum model
        training_config: Training configuration
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        tokenizer: Tokenizer for text processing

    Returns:
        Configured QuantumTrainer instance
    """

    # Create model
    _model = ScalableQuantumTransformer(**model_config)

    # Create training config
    _config = TrainingConfig(**training_config)

    # Create trainer
    _trainer = QuantumTrainer(
        _model=model,
        _config=config,
        _train_dataset=train_dataset,
        _eval_dataset=eval_dataset,
        _tokenizer=tokenizer,
    )

    return trainer


def launch_distributed_training(training_fn: Callable, world_size: int, **kwargs):
    """
    Launch distributed training across multiple GPUs.

    Args:
        training_fn: Training function to execute
        world_size: Number of processes (GPUs)
        **kwargs: Arguments to pass to training function
    """

    mp.spawn(training_fn, _args=(world_size, kwargs), _nprocs=world_size, _join=True)


# Multi-GPU training entry point
def distributed_training_main(rank: int, world_size: int, args: Dict[str, Any]):
    """Main function for distributed training."""

    # Update config with distributed settings
    args["training_config"]["rank"] = rank
    args["training_config"]["local_rank"] = rank
    args["training_config"]["world_size"] = world_size

    # Create and run trainer
    _trainer = create_quantum_trainer(**args)
    trainer.train()
