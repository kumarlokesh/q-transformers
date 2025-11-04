"""
Distributed Quantum Attention with Multi-GPU Support

Advanced distributed computing for quantum transformers:
- Multi-GPU quantum attention parallelization
- Quantum state synchronization across devices
- Load balancing for quantum sampling operations
- Communication-efficient quantum gradient aggregation
"""

from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
from .attention import QuantumMultiheadAttention


class DistributedQuantumAttention(nn.Module):
    """
    Distributed quantum attention across multiple GPUs.

    Splits quantum attention computation across devices while maintaining
    coherence of quantum states and efficient communication.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        world_size: int,
        rank: int,
        quantum_config: Dict[str, Any] = None,
        communication_backend: str = "nccl",
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.world_size = world_size
        self.rank = rank
        self.quantum_config = quantum_config or {}

        # Ensure num_heads is divisible by world_size for even distribution
        assert (
            num_heads % world_size == 0
        ), f"num_heads ({num_heads}) must be divisible by world_size ({world_size})"

        self.heads_per_device = num_heads // world_size
        self.head_dim = embed_dim // num_heads

        # Local attention heads for this device
        # Local attention implementation (expected to be provided elsewhere in the package)
        self.local_attention = QuantumMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=self.heads_per_device,
            quantum_config=quantum_config,
            batch_first=True,
        )

        # Communication group for quantum state synchronization
        self.process_group = None
        self.setup_communication(communication_backend)

        # Quantum state buffers for distributed computation
        self.quantum_state_buffer = None
        self.gradient_buffer = None

    def setup_communication(self, backend: str):
        """Setup distributed communication for quantum operations."""
        if dist.is_initialized():
            # Create process group for quantum attention communication
            ranks = list(range(self.world_size))
            # use explicit kwargs expected by torch.distributed
            self.process_group = dist.new_group(ranks=ranks, backend=backend)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Distributed quantum attention forward pass.

        Args:
            query: Query tensor (batch_size, seq_len, embed_dim)
            key: Key tensor (batch_size, seq_len, embed_dim)
            value: Value tensor (batch_size, seq_len, embed_dim)
            attn_mask: Attention mask
            key_padding_mask: Key padding mask

        Returns:
            Attention output tensor
        """
        batch_size, seq_len, embed_dim = query.shape

        # Split heads across devices (bookkeeping)
        # start/end head indices are available if needed
        # start_head = self.rank * self.heads_per_device
        # end_head = (self.rank + 1) * self.heads_per_device

        # Compute local attention for assigned heads
        local_output, local_attn_weights = self.local_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )

        # Gather outputs from all devices
        if self.world_size > 1:
            output = self._all_gather_attention_outputs(local_output)
        else:
            output = local_output

        return output

    def _all_gather_attention_outputs(self, local_output: torch.Tensor) -> torch.Tensor:
        """
        Gather attention outputs from all devices and combine.

        Uses efficient communication patterns for quantum attention results.
        """
        batch_size, seq_len, local_dim = local_output.shape

        # Create buffer for gathering outputs
        gathered_outputs = [
            torch.zeros_like(local_output) for _ in range(self.world_size)
        ]

        # All-gather operation
        if self.process_group:
            dist.all_gather(gathered_outputs, local_output, group=self.process_group)
        else:
            gathered_outputs[0] = local_output  # Single GPU case

        # Concatenate along head dimension
        full_output = torch.cat(gathered_outputs, dim=-1)

        return full_output

    def synchronize_quantum_states(
        self, quantum_states: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Synchronize quantum states across devices for coherent computation.

        Ensures quantum coherence is maintained in distributed setting.
        """
        synchronized_states: List[torch.Tensor] = []

        for state in quantum_states:
            # Create buffer for state synchronization
            state_buffer = torch.zeros_like(state)

            # All-reduce to compute average quantum state
            if self.process_group:
                dist.all_reduce(
                    state_buffer, op=dist.ReduceOp.SUM, group=self.process_group
                )
                state_buffer = state_buffer / self.world_size
            else:
                state_buffer = state

            synchronized_states.append(state_buffer)

        return synchronized_states

    def distribute_quantum_sampling(
        self, attention_logits: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        """
        Distribute quantum sampling across multiple devices.

        Each device samples a portion of the attention distribution.
        """
        batch_size, seq_len_q, _seq_len_k = attention_logits.shape

        # Determine sampling distribution across devices
        samples_per_device = num_samples // self.world_size
        remaining_samples = num_samples % self.world_size

        # This device's sample count
        local_samples = samples_per_device
        if self.rank < remaining_samples:
            local_samples += 1

        # Perform local quantum sampling
        probs = F.softmax(attention_logits, dim=-1)
        local_attention_weights = torch.zeros_like(probs)

        for b in range(batch_size):
            for q in range(seq_len_q):
                prob_row = probs[b, q, :]
                if local_samples > 0 and prob_row.sum() > 1e-8:
                    samples = torch.multinomial(
                        prob_row, local_samples, replacement=True
                    )
                    sample_counts = torch.bincount(samples, minlength=_seq_len_k)
                    local_attention_weights[b, q, :] = sample_counts.float() / float(
                        local_samples
                    )

        # Gather and combine attention weights from all devices
        if self.world_size > 1:
            # All-reduce to combine sampling results
            dist.all_reduce(
                local_attention_weights, op=dist.ReduceOp.SUM, group=self.process_group
            )
            # Normalize by number of devices
            local_attention_weights = local_attention_weights / float(self.world_size)

        return local_attention_weights


class QuantumGradientSynchronizer:
    """
    Synchronizer for quantum attention gradients across distributed devices.

    Handles communication-efficient gradient aggregation for quantum parameters.
    """

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self.gradient_buffers = {}

    def register_quantum_parameters(self, model: nn.Module):
        """Register quantum parameters for gradient synchronization."""
        self.quantum_params = []

        for name, param in model.named_parameters():
            if "quantum" in name.lower() or "attention" in name.lower():
                self.quantum_params.append((name, param))
                # Create gradient buffer
                self.gradient_buffers[name] = torch.zeros_like(param)

    def synchronize_gradients(self, async_op: bool = False):
        """
        Synchronize quantum parameter gradients across devices.

        Uses optimized communication patterns for quantum gradients.
        """
        handles: List[Any] = []

        for name, param in self.quantum_params:
            if param.grad is not None:
                # Copy gradient to buffer
                self.gradient_buffers[name].copy_(param.grad)

                # All-reduce gradient (may return a handle when async)
                handle = dist.all_reduce(
                    self.gradient_buffers[name], op=dist.ReduceOp.SUM, async_op=async_op
                )

                if async_op:
                    handles.append(handle)
                else:
                    # Average gradient
                    self.gradient_buffers[name] /= float(self.world_size)
                    param.grad.copy_(self.gradient_buffers[name])

        # Wait for async operations
        if async_op:
            for handle in handles:
                handle.wait()

            # Apply averaged gradients after async ops complete
            for name, param in self.quantum_params:
                if param.grad is not None:
                    self.gradient_buffers[name] /= float(self.world_size)
                    param.grad.copy_(self.gradient_buffers[name])

    def apply_quantum_noise_reduction(self):
        """
        Apply noise reduction techniques to quantum gradients.

        Reduces noise introduced by quantum sampling in distributed setting.
        """
        for name, param in self.quantum_params:
            if param.grad is not None:
                # Apply gradient clipping specific to quantum parameters
                grad_norm = torch.norm(param.grad)
                if grad_norm > 1.0:  # Quantum-specific threshold
                    param.grad *= 0.5 / grad_norm

                # Add small amount of regularization
                param.grad += 1e-6 * param.data


class MultiGPUQuantumTransformer(nn.Module):
    """
    Multi-GPU quantum transformer with optimized distributed attention.

    Manages quantum attention computation across multiple GPUs with
    load balancing and efficient communication.
    """

    def __init__(self, config: Dict[str, Any], world_size: int, rank: int):
        super().__init__()

        self.config = config
        self.world_size = world_size
        self.rank = rank

        # Distributed quantum attention layers
        self.attention_layers = nn.ModuleList(
            [
                DistributedQuantumAttention(
                    embed_dim=config["hidden_size"],
                    num_heads=config["num_attention_heads"],
                    world_size=world_size,
                    rank=rank,
                    quantum_config=config.get("quantum_config", {}),
                )
                for _ in range(config["num_hidden_layers"])
            ]
        )

        # Other transformer components (embeddings, feed-forward, etc.)
        self.embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layer_norms = nn.ModuleList(
            [
                nn.LayerNorm(config["hidden_size"])
                for _ in range(config["num_hidden_layers"])
            ]
        )

        # Feed-forward networks
        self.feed_forwards = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config["hidden_size"], config["intermediate_size"]),
                    nn.GELU(),
                    nn.Linear(config["intermediate_size"], config["hidden_size"]),
                )
                for _ in range(config["num_hidden_layers"])
            ]
        )

        # Gradient synchronizer
        self.gradient_sync = QuantumGradientSynchronizer(world_size, rank)
        self.gradient_sync.register_quantum_parameters(self)

        # Load balancing for quantum operations
        self.load_balancer = QuantumLoadBalancer(world_size, rank)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with distributed quantum attention."""

        # Embeddings
        hidden_states = self.embeddings(input_ids)

        # Apply attention layers with load balancing
        for i, (attn_layer, norm_layer, ff_layer) in enumerate(
            zip(self.attention_layers, self.layer_norms, self.feed_forwards)
        ):
            # Load balancing decision
            if self.load_balancer.should_compute_layer(i):
                # Quantum attention with residual connection
                attn_output, attn_weights = attn_layer(
                    hidden_states,
                    hidden_states,
                    hidden_states,
                    key_padding_mask=attention_mask,
                )
                hidden_states = norm_layer(hidden_states + attn_output)

                # Feed-forward with residual connection
                ff_output = ff_layer(hidden_states)
                hidden_states = norm_layer(hidden_states + ff_output)
            else:
                # Skip computation on this device, will receive from others
                hidden_states = self.load_balancer.receive_layer_output(
                    i, hidden_states
                )

        return hidden_states

    def backward_hook(self):
        """Custom backward hook for quantum gradient synchronization."""
        self.gradient_sync.apply_quantum_noise_reduction()
        self.gradient_sync.synchronize_gradients(async_op=True)


class QuantumLoadBalancer:
    """
    Load balancer for distributed quantum computations.

    Dynamically distributes quantum attention layers across devices
    based on computational load and communication costs.
    """

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self.layer_assignments = {}
        self.computation_times = {}

    def should_compute_layer(self, layer_idx: int) -> bool:
        """
        Determine if this device should compute the given layer.

        Uses round-robin assignment by default, can be made adaptive.
        """
        _assigned_device = layer_idx % self.world_size
        return _assigned_device == self.rank

    def receive_layer_output(
        self, layer_idx: int, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Receive layer output from the device that computed it.

        Args:
            layer_idx: Index of the layer
            input_tensor: Input tensor (used for shape/device info)

        Returns:
            Layer output tensor
        """
        assigned_device = layer_idx % self.world_size

        if assigned_device == self.rank:
            # This device computed the layer
            return input_tensor
        else:
            # Receive from the assigned device
            output_tensor = torch.zeros_like(input_tensor)
            # broadcast in-place into output_tensor from the assigned device
            dist.broadcast(output_tensor, src=assigned_device)
            return output_tensor

    def update_computation_times(self, layer_idx: int, computation_time: float):
        """Update computation time statistics for load balancing."""
        self.computation_times[layer_idx] = computation_time

    def rebalance_layers(self) -> Dict[int, int]:
        """
        Rebalance layer assignments based on computation times.

        Returns:
            New layer assignment mapping {layer_idx: device_rank}
        """
        if not self.computation_times:
            # Use round-robin if no timing data; nothing to rebalance
            return {}

        # Sort layers by computation time (descending)
        sorted_layers = sorted(
            self.computation_times.items(), key=lambda x: x[1], reverse=True
        )

        # Greedy assignment to balance load
        device_loads = [0.0] * self.world_size
        new_assignments: Dict[int, int] = {}

        for layer_idx, comp_time in sorted_layers:
            # Assign to device with minimum current load
            min_device = min(range(self.world_size), key=lambda i: device_loads[i])
            new_assignments[layer_idx] = min_device
            device_loads[min_device] += comp_time

        self.layer_assignments = new_assignments
        return new_assignments


def setup_distributed_quantum_training(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl",
):
    """
    Setup distributed quantum transformer training.

    Args:
        rank: Current process rank
        world_size: Total number of processes
        master_addr: Master node address
        master_port: Master node port
        backend: Communication backend (nccl, gloo, mpi)
    """

    # Set environment variables
    import os

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    # Initialize process group
    dist.init_process_group(backend=backend, _rank=rank, _world_size=world_size)

    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    print("Initialized distributed quantum training: rank {rank}/{world_size}")


def cleanup_distributed_quantum_training():
    """Cleanup distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


class QuantumCommunicationOptimizer:
    """
    Optimizer for quantum attention communication patterns.

    Reduces communication overhead in distributed quantum computations.
    """

    def __init__(self, world_size: int):
        self.world_size = world_size
        self.communication_schedule = {}
        self.compression_enabled = True

    def optimize_attention_communication(
        self, attention_patterns: List[torch.Tensor], sparsity_threshold: float = 0.01
    ) -> List[torch.Tensor]:
        """
        Optimize communication of attention patterns.

        Uses sparsity and compression to reduce communication volume.
        """
        _optimized_patterns: List[torch.Tensor] = []

        for pattern in attention_patterns:
            if self.compression_enabled:
                # Apply sparsification
                _sparse_pattern = self._sparsify_attention(pattern, sparsity_threshold)

                # Compress using quantization
                _compressed_pattern = self._quantize_attention(_sparse_pattern)

                _optimized_patterns.append(_compressed_pattern)
            else:
                _optimized_patterns.append(pattern)

        return _optimized_patterns

    def _sparsify_attention(
        self, attention_matrix: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        """Apply sparsification to attention matrix."""
        # Zero out small attention weights
        _sparse_matrix = attention_matrix.clone()
        _sparse_matrix[_sparse_matrix < threshold] = 0.0

        # Renormalize rows
        _row_sums = _sparse_matrix.sum(dim=-1, keepdim=True)
        _sparse_matrix = _sparse_matrix / (_row_sums + 1e-8)

        return _sparse_matrix

    def _quantize_attention(self, attention_matrix: torch.Tensor) -> torch.Tensor:
        """Apply quantization to reduce communication size."""
        # Simple 8-bit quantization
        _min_val = attention_matrix.min()
        _max_val = attention_matrix.max()

        # Scale to [0, 255] and quantize
        _scaled = (attention_matrix - _min_val) / (_max_val - _min_val + 1e-8)
        _quantized = torch.round(_scaled * 255).byte()

        # Dequantize for computation
        _dequantized = _quantized.float() / 255.0 * (_max_val - _min_val) + _min_val

        return _dequantized


# Factory function for creating distributed quantum models
def create_distributed_quantum_transformer(
    config: Dict[str, Any], world_size: int, rank: int
) -> Any:
    """
    Create distributed quantum transformer model.

    Args:
        config: Model configuration
        world_size: Number of distributed processes
        rank: Current process rank

    Returns:
        Distributed quantum transformer model
    """

    # Build a distributed quantum transformer using the DistributedQuantumAttention
    _model = DistributedQuantumAttention(
        embed_dim=config.get("embed_dim", 768),
        num_heads=config.get("num_heads", 12),
        world_size=world_size,
        rank=rank,
        quantum_config=config.get("quantum_config", {}),
    )

    # Wrap with DistributedDataParallel for standard parameters
    if world_size > 1:
        _model = DistributedDataParallel(
            _model,
            device_ids=[rank] if torch.cuda.is_available() else None,
            find_unused_parameters=True,  # For quantum attention modules
        )

    return _model
