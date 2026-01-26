import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from torch.utils.checkpoint import checkpoint

class RecursiveBlock(nn.Module):
    """
    A simple residual convolutional block that preserves spatial dimensions.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels)
        )
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.net(x)
        return self.act(out + residual)

class RecursiveFER(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 64, max_steps: int = 6, use_gradient_checkpointing: bool = False):
        super().__init__()
        self.max_steps = max_steps
        self.num_classes = num_classes
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # 1. Input Stem (Project Image -> Hidden Dim)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU()
        )

        # 2. The Loop Body
        self.recursive_block = RecursiveBlock(hidden_dim)

        # 3. Heads
        # Classifier: Maps hidden features to Class Logits
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Halting Head: Maps hidden features to a "Stop" logit
        self.halt_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, x, labels=None, step_weights=None):
        """
        Args:
            x: Input images [Batch, Channels, Height, Width]
            labels: Optional ground truth [Batch]
        
        Returns:
            Training: total_loss, (task_loss, halt_loss)
            Inference: final_logits, steps_taken, halt_probs
        """
        # Embed input: [b, c, h, w] -> [b, hidden, h, w]
        h = self.stem(x)
        
        logits_list = []
        halt_logits_list = []

        # --- 1. Recursive Unrolling ---
        for step in range(self.max_steps):
            # Use gradient checkpointing for the recursive block if enabled
            if self.use_gradient_checkpointing and self.training:
                def create_recursive_forward(block):
                    def recursive_forward(x):
                        return block(x)
                    return recursive_forward
                
                h = checkpoint(create_recursive_forward(self.recursive_block), h, use_reentrant=False)
            else:
                # Recurse (Preserve spatial dims)
                h = self.recursive_block(h)
            
            # Pool for prediction: [b, hidden, h, w] -> [b, hidden]
            pooled = reduce(h, 'b c h w -> b c', 'mean')
            
            # Predict
            logits_list.append(self.classifier(pooled))
            halt_logits_list.append(self.halt_head(h))

        # Stack outputs into a time sequence
        # [b, steps, classes]
        logits_seq = torch.stack(logits_list, dim=1)
        # [b, steps] (Remove the singleton last dim from Linear(..., 1))
        halt_logits_seq = torch.stack(halt_logits_list, dim=1).squeeze(-1)

        # --- 2. Training Logic (Supervised Halting) ---
        if labels is not None:
            # A. Task Loss (CrossEntropy across ALL steps)
            # Repeat labels to match the sequence length: [b] -> [b*steps]
            labels_seq = repeat(labels, 'b -> (b s)', s=self.max_steps)
            # B. Halting Loss (Binary Cross Entropy)
            # Find which steps were actually correct
            preds_seq = logits_seq.argmax(dim=-1) # [b, s]
            # Flatten logits to match labels: [b, s, c] -> [b*s, c]
            logits_flat = rearrange(logits_seq, 'b s c -> (b s) c')
            # --- Calculate individual losses ---
            # reduction='none' gives us a loss for every single step of every image
            loss_matrix = F.cross_entropy(logits_flat, labels_seq, reduction='none')
            # Reshape back to [Batch, Steps]
            loss_matrix = rearrange(loss_matrix, '(b s) -> b s', s=self.max_steps)

            if step_weights is not None:
                # Apply weights: broadcast [s] weights across [b, s] matrix
                # We normalize weights so their sum equals the number of steps 
                # (keeps the loss scale consistent)
                step_weights = step_weights * (self.max_steps / step_weights.sum())
                loss_matrix = loss_matrix * step_weights
            task_loss = loss_matrix.mean()


            
            # Expand labels for comparison: [b] -> [b, s]
            labels_expanded = repeat(labels, 'b -> b s', s=self.max_steps)
            
            # Target is 1.0 if prediction is correct, 0.0 otherwise
            is_correct = (preds_seq == labels_expanded).float()
            
            halt_loss = F.binary_cross_entropy_with_logits(halt_logits_seq, is_correct)

            # Weighted sum (Task is usually harder, so we weight it higher)
            total_loss = task_loss + (0.5 * halt_loss)
            return total_loss, (task_loss, halt_loss)

        # --- 3. Inference Logic (Adaptive Stopping) ---
        else:
            # Calculate probabilities
            halt_probs = torch.sigmoid(halt_logits_seq) # [b, s]

            # Determine where to stop (Prob > 0.5)
            should_stop = (halt_probs > 0.5).long()  # Convert to long for argmax
            
            # Force stop at the last step if never stopped
            force_stop = torch.ones_like(should_stop[:, :1]) # [b, 1]
            should_stop = torch.cat([should_stop, force_stop], dim=1) # [b, s+1]
            
            # Find the FIRST index where stop is True
            stop_indices = should_stop.argmax(dim=1) 
            
            # Clamp to valid range (0 to max_steps-1)
            stop_indices = stop_indices.clamp(max=self.max_steps - 1)

            # Gather the specific answer from the sequence
            # We need to select one 'step' per batch from logits_seq [b, s, c]
            
            # einops doesn't replace torch.gather well for index selection, 
            # so we use standard PyTorch gather here.
            # Reshape indices: [b] -> [b, 1, 1] -> [b, 1, c]
            gather_indices = stop_indices.view(-1, 1, 1).expand(-1, 1, self.num_classes)
            
            final_logits = logits_seq.gather(1, gather_indices).squeeze(1)
            
            # Steps taken (1-based count for metrics)
            steps_taken = stop_indices.float() + 1.0

            return final_logits, steps_taken, halt_probs