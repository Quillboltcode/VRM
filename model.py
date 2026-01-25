
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class RecursiveBlock(nn.Module):
    """
    A recursive block with a residual connection.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += residual
        out = self.gelu(out)
        return out

class HaltingUnit(nn.Module):
    """
    Halting unit which decides whether to continue the recursion.
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        out = self.pool(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return self.sigmoid(out)

class RecursiveFERModel(nn.Module):
    """
    Recursive Facial Expression Recognition model with Adaptive Computation Time.
    """
    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 64, max_steps: int = 10):
        super().__init__()
        self.max_steps = max_steps
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )

        # Recursive components
        self.recursive_block = RecursiveBlock(hidden_dim, hidden_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.halting_unit = HaltingUnit(hidden_dim)

        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # To store halting probabilities for loss calculation
        self.halting_probabilities = []

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = x.size(0)
        h = self.embedding(x)
        
        # Initial hidden state for GRU
        h_flat = h.mean(dim=[2, 3])
        
        # Initialize tensors for accumulation
        total_output = torch.zeros(batch_size, self.classifier.out_features, device=x.device)
        cumulative_halt_probs = torch.zeros(batch_size, 1, device=x.device)
        
        # To store halting probabilities for ponder loss
        self.halting_probabilities = []
        
        # Per-sample step counter
        step_counters = torch.zeros(batch_size, device=x.device)
        # Track which samples are still running
        still_running = torch.ones(batch_size, dtype=torch.bool, device=x.device)

        for step in range(self.max_steps):
            if not still_running.any():
                break

            # GRU Memory update for all samples
            pooled_h = h.mean(dim=[2, 3])
            h_flat = self.gru_cell(pooled_h, h_flat)

            # Reshape GRU output back to image-like tensor and apply recursive block
            h = self.recursive_block(h_flat.unsqueeze(2).unsqueeze(3).expand_as(h))

            # Calculate halting probability for all samples
            halt_p = self.halting_unit(h)

            # For the last step, force halt for remaining samples
            if step == self.max_steps - 1:
                halt_p[still_running] = 1.0
            
            # Store probabilities for ponder loss
            self.halting_probabilities.append(halt_p.detach())

            # Calculate the weight for this step's output
            step_weight = torch.zeros_like(halt_p)
            if still_running.any():
                remaining_prob = 1.0 - cumulative_halt_probs[still_running]
                step_weight[still_running] = torch.min(halt_p[still_running], remaining_prob)

            # Update total output for all samples (but only running ones contribute)
            running_features = h.mean(dim=[2, 3])
            classifier_output = self.classifier(running_features)
            
            # Add weighted output to total
            total_output += step_weight * classifier_output

            # Update which samples are still running
            newly_halted = (cumulative_halt_probs.squeeze() + halt_p.squeeze()) >= 1.0
            still_running = still_running & (~newly_halted)

            # Update cumulative halt probabilities
            cumulative_halt_probs += halt_p
            cumulative_halt_probs = torch.clamp(cumulative_halt_probs, 0.0, 1.0)
            
            # Increment step counters for running samples
            step_counters[still_running] += 1
            
        return total_output, step_counters