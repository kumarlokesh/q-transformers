#!/usr/bin/env python3
"""
Example: Training a Quantum Transformer on GLUE Tasks

This script demonstrates how to train a quantum-enhanced transformer
on real NLP tasks using the Q-Transformers library.
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from qtransformers import (
    create_quantum_trainer,
    TrainingConfig,
    QuantumTrainer,
    GLUEBenchmarkSuite
)


def main():
    """Main training example."""
    
    print("🚀 Quantum Transformer Training Example")
    print("=" * 50)
    
    # Configuration
    model_config = {
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_hidden_layers": 6,  # Smaller model for demo
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "quantum_config": {
            "backend": "phase0-proto",
            "num_samples": 32,
            "use_advanced_sampling": True,
            "use_error_mitigation": True,
            "use_gpu_acceleration": torch.cuda.is_available()
        }
    }
    
    training_config = TrainingConfig(
        model_name="quantum-bert-demo",
        learning_rate=2e-5,
        batch_size=16,
        max_steps=1000,  # Short training for demo
        warmup_steps=100,
        eval_steps=200,
        save_steps=500,
        checkpoint_dir="./demo_checkpoints",
        logging_steps=50
    )
    
    print(f"Model configuration: {model_config}")
    print(f"Training configuration: {training_config}")
    
    # Load demo dataset (CoLA for simplicity)
    print("\n📚 Loading CoLA dataset...")
    dataset = load_dataset("glue", "cola")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    
    train_dataset = dataset["train"].map(tokenize_function, batched=True)
    eval_dataset = dataset["validation"].map(tokenize_function, batched=True)
    
    # Add labels
    train_dataset = train_dataset.rename_column("label", "labels")
    eval_dataset = eval_dataset.rename_column("label", "labels")
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    # Create trainer
    print("\n🔧 Creating quantum trainer...")
    trainer = create_quantum_trainer(
        model_config=model_config,
        training_config=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    
    # Define evaluation metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}
    
    trainer.compute_metrics = compute_metrics
    
    # Start training
    print("\n🎯 Starting quantum transformer training...")
    trainer.train()
    
    # Final evaluation
    print("\n📊 Running final evaluation...")
    final_results = trainer.evaluate()
    print(f"Final results: {final_results}")
    
    # Compare with classical baseline
    print("\n🔬 Running quantum advantage analysis...")
    benchmark_suite = GLUEBenchmarkSuite()
    
    # Load trained model
    quantum_model = trainer.model
    
    # Create classical baseline (simplified for demo)
    classical_config = model_config.copy()
    classical_config["quantum_config"]["backend"] = "classical"
    
    classical_trainer = create_quantum_trainer(
        model_config=classical_config,
        training_config=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    
    print("Training classical baseline...")
    classical_trainer.train()
    classical_model = classical_trainer.model
    
    # Compare models
    quantum_results = trainer.evaluate()
    classical_results = classical_trainer.evaluate()
    
    print(f"\n📈 Performance Comparison:")
    print(f"Quantum Model Accuracy: {quantum_results.get('eval_accuracy', 0):.4f}")
    print(f"Classical Model Accuracy: {classical_results.get('eval_accuracy', 0):.4f}")
    
    improvement = quantum_results.get('eval_accuracy', 0) - classical_results.get('eval_accuracy', 0)
    print(f"Quantum Advantage: {improvement:.4f} ({improvement*100:.2f}%)")
    
    print("\n✅ Training completed successfully!")
    print(f"Models saved to: {training_config.checkpoint_dir}")


if __name__ == "__main__":
    main()
