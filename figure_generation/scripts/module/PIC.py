import torch
import torch.nn as nn

from .backbones import AttentionBackbone, CNNBackbone, AvgPoolBackbone


class PIC(nn.Module):
    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        hidden_units,
        attn_drop,
        linear_drop,
        device,
        model_variant: str = "attention",
        num_heads: int = 1,
        cnn_channels: int = 256,
        cnn_kernel_size: int = 5,
        cnn_layers: int = 2,
        cnn_drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.device = device
        self.model_variant = model_variant

        if model_variant == "attention":
            self.backbone = AttentionBackbone(
                input_shape=input_shape,
                num_heads=num_heads,
                attn_drop=attn_drop,
                device=device,
            )
        elif model_variant == "cnn":
            self.backbone = CNNBackbone(
                input_shape=input_shape,
                cnn_channels=cnn_channels,
                cnn_kernel_size=cnn_kernel_size,
                cnn_layers=cnn_layers,
                cnn_drop=cnn_drop,
            )
        elif model_variant == "avgpool":
            self.backbone = AvgPoolBackbone()
        else:
            raise ValueError(f"Unsupported model_variant: {model_variant}")

        hidden_units = int(hidden_units)
        self.generator = nn.Sequential(
            nn.BatchNorm1d(input_shape, device=self.device, dtype=torch.float32),
            nn.Linear(
                in_features=input_shape,
                out_features=hidden_units * 2,
                device=self.device,
                dtype=torch.float32,
            ),
            nn.ReLU(),
            nn.Dropout(linear_drop),
            nn.BatchNorm1d(hidden_units * 2, device=self.device, dtype=torch.float32),
            nn.Linear(
                in_features=hidden_units * 2,
                out_features=hidden_units,
                device=self.device,
                dtype=torch.float32,
            ),
            nn.ReLU(),
            nn.Dropout(linear_drop),
            nn.Linear(
                in_features=hidden_units,
                out_features=output_shape,
                device=self.device,
                dtype=torch.float32,
            ),
        )

    def forward(self, x, start_padding_indices, get_attention: bool = False):
        pooled_feature, extras = self.backbone(x, start_padding_indices)
        logits = self.generator(pooled_feature)
        if get_attention:
            if self.model_variant != "attention":
                raise ValueError("get_attention=True is only supported for attention variant.")
            return logits, extras.get("attention_weights")
        return logits
