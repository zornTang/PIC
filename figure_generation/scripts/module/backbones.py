import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


def masked_mean(sequence_tensor: torch.Tensor, start_padding_indices: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean of variable-length sequences while ignoring padded positions.
    """
    batch_size, seq_len, input_dim = sequence_tensor.shape
    pooled = torch.zeros(batch_size, input_dim, dtype=sequence_tensor.dtype, device=sequence_tensor.device)
    for i, start_idx in enumerate(start_padding_indices):
        if start_idx != -1:
            if start_idx == 0:
                pooled[i] = torch.zeros(input_dim, device=sequence_tensor.device, dtype=sequence_tensor.dtype)
            else:
                pooled[i] = torch.mean(sequence_tensor[i, :start_idx, :], dim=0)
        else:
            pooled[i] = torch.mean(sequence_tensor[i, :, :], dim=0)
    return pooled


class AttentionBackbone(nn.Module):
    def __init__(self, input_shape: int, num_heads: int, attn_drop: float, device: torch.device) -> None:
        super().__init__()
        if input_shape % num_heads != 0:
            raise ValueError(
                f"input_shape ({input_shape}) must be divisible by num_heads ({num_heads}) for MultiheadAttention."
            )
        self.device = device
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=input_shape,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(input_shape)

    def forward(
        self, x: torch.Tensor, start_padding_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, seq_len, _ = x.shape
        key_padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=x.device)
        for i, start_padding_idx in enumerate(start_padding_indices):
            if start_padding_idx != -1:
                key_padding_mask[i, start_padding_idx:] = True
        attn_output, attention_weights = self.multihead_attention(
            x, x, x, key_padding_mask=key_padding_mask
        )
        output = x + self.layer_norm(attn_output)
        pooled = masked_mean(output, start_padding_indices)
        return pooled, {"attention_weights": attention_weights}


class CNNBackbone(nn.Module):
    def __init__(
        self,
        input_shape: int,
        cnn_channels: int,
        cnn_kernel_size: int,
        cnn_layers: int,
        cnn_drop: float,
    ) -> None:
        super().__init__()
        layers = []
        in_channels = input_shape
        for _ in range(max(1, cnn_layers)):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=cnn_channels,
                    kernel_size=cnn_kernel_size,
                    padding=cnn_kernel_size // 2,
                )
            )
            layers.append(nn.BatchNorm1d(cnn_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cnn_drop))
            in_channels = cnn_channels
        layers.append(nn.Conv1d(in_channels=in_channels, out_channels=input_shape, kernel_size=1))
        self.conv_stack = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, start_padding_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # x: (batch, seq, channels) -> conv expects (batch, channels, seq)
        conv_input = x.transpose(1, 2)
        conv_output = self.conv_stack(conv_input).transpose(1, 2)
        pooled = masked_mean(conv_output, start_padding_indices)
        return pooled, {}


class AvgPoolBackbone(nn.Module):
    def forward(
        self, x: torch.Tensor, start_padding_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pooled = masked_mean(x, start_padding_indices)
        return pooled, {}
