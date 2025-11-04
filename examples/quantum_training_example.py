#!/usr/bin/env python3
# flake8: noqa
"""
Example: Training a Quantum Transformer on GLUE Tasks

This script demonstrates how to train a quantum-enhanced transformer
on real NLP tasks using the Q-Transformers library.
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from qtransformers import (
    GLUEBenchmarkSuite,
    QuantumTrainer,
    TrainingConfig,
    create_quantum_trainer,
)


def main():
    """Main training example."""

    print("🚀 Quantum Transformer Training Example")
    print("=" * 50)

    # Configuration
    _model_config = {
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_hidden_layers": 6,  # Smaller model for demo
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "quantum_config": {
            "backend": "prototype",
            "num_samples": 32,
            "use_advanced_sampling": True,
            "use_error_mitigation": True,
            "use_gpu_acceleration": torch.cuda.is_available(),
        },
    }

    _training_config = TrainingConfig(
        _model_name="quantum-bert-demo",
        _learning_rate=2e-5,
        _batch_size=16,
        _max_steps=1000,  # Short training for demo
        _warmup_steps=100,
        _eval_steps=200,
        _save_steps=500,
        _checkpoint_dir="./demo_checkpoints",
        _logging_steps=50,
    )

    print("Model configuration: {model_config}")
    print("Training configuration: {training_config}")

    # Load demo dataset (CoLA for simplicity)
    print("\n📚 Loading CoLA dataset...")
    _dataset = load_dataset("glue", "cola")
    _tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            _truncation=True,
            _padding="max_length",
            _max_length=128,
        )

    _train_dataset = dataset["train"].map(tokenize_function, _batched=True)
    _eval_dataset = dataset["validation"].map(tokenize_function, _batched=True)

    # Add labels
    _train_dataset = train_dataset.rename_column("label", "labels")
    _eval_dataset = eval_dataset.rename_column("label", "labels")

    print("Training samples: {len(train_dataset)}")
    print("Validation samples: {len(eval_dataset)}")

    # Create trainer
    print("\n🔧 Creating quantum trainer...")
    _trainer = create_quantum_trainer(
        _model_config=model_config,
        _training_config=training_config,
        _train_dataset=train_dataset,
        _eval_dataset=eval_dataset,
        _tokenizer=tokenizer,
    )

    # Define evaluation metrics
    def compute_metrics(eval_pred):
        predictions, _labels = eval_pred
        _predictions = predictions.argmax(axis=1)
        _accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    trainer.compute_metrics = compute_metrics

    # Start training
    print("\n🎯 Starting quantum transformer training...")
    trainer.train()

    # Final evaluation
    print("\n📊 Running final evaluation...")
    _final_results = trainer.evaluate()
    print("Final results: {final_results}")

    # Compare with classical baseline
    print("\n🔬 Running quantum advantage analysis...")
    _benchmark_suite = GLUEBenchmarkSuite()

    # Load trained model
    _quantum_model = trainer.model

    # Create classical baseline (simplified for demo)
    _classical_config = model_config.copy()
    classical_config["quantum_config"]["backend"] = "classical"

    _classical_trainer = create_quantum_trainer(
        _model_config=classical_config,
        _training_config=training_config,
        _train_dataset=train_dataset,
        _eval_dataset=eval_dataset,
        _tokenizer=tokenizer,
    )

    print("Training classical baseline...")
    classical_trainer.train()
    _classical_model = classical_trainer.model

    # Compare models
    _quantum_results = trainer.evaluate()
    _classical_results = classical_trainer.evaluate()

    print("\n📈 Performance Comparison:")
    print("Quantum Model Accuracy: {quantum_results.get('eval_accuracy', 0):.4f}")
    print("Classical Model Accuracy: {classical_results.get('eval_accuracy', 0):.4f}")

    _improvement = quantum_results.get("eval_accuracy", 0) - classical_results.get(
        "eval_accuracy", 0
    )
    print("Quantum Advantage: {improvement:.4f} ({improvement*100:.2f}%)")

    print("\n✅ Training completed successfully!")
    print("Models saved to: {training_config.checkpoint_dir}")


if __name__ == "__main__":
    main()
