
import torch
import torch.nn as nn
from torch import Tensor
from typing import List

class PonderLoss(nn.Module):
    """
    Loss function for models with Adaptive Computation Time.
    It's a combination of a standard classification loss and a ponder cost.
    """
    def __init__(self, classification_loss: nn.Module, lambda_ponder: float):
        """
        Args:
            classification_loss: The base loss for the classification task (e.g., CrossEntropyLoss).
            lambda_ponder: The weight for the ponder cost.
        """
        super().__init__()
        self.classification_loss = classification_loss
        self.lambda_ponder = lambda_ponder

    def forward(self, logits: Tensor, targets: Tensor, num_steps: Tensor) -> Tensor:
        """
        Args:
            logits: The model's output logits.
            targets: The ground truth labels.
            num_steps: A tensor with the number of steps taken for each sample.

        Returns:
            The total loss, classification loss, and ponder cost.
        """
        # Classification loss
        class_loss = self.classification_loss(logits, targets)

        # Ponder cost is the average number of steps
        ponder_cost = num_steps.mean()

        total_loss = class_loss + self.lambda_ponder * ponder_cost
        return total_loss, class_loss, ponder_cost
