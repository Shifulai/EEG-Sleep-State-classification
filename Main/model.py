from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNFeatureExtractor(nn.Module):
    """Extract local temporal patterns from one EEG epoch."""

    def __init__(self, in_channels: int, hidden_channels: int, embedding_dim: int) -> None:
        super().__init__()
        c1 = hidden_channels // 2
        c2 = hidden_channels

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, c1, kernel_size=7, stride=2),
            nn.MaxPool1d(kernel_size=2),
            ConvBlock(c1, c2, kernel_size=5, stride=1),
            nn.MaxPool1d(kernel_size=2),
            ConvBlock(c2, c2, kernel_size=3, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(c2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, L]
        features = self.encoder(x)
        pooled = self.pool(features).squeeze(-1)
        return self.proj(pooled)


class CNNTransformerClassifier(nn.Module):
    """Sleep staging model: EEG -> CNN -> Transformer Encoder -> classifier."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embedding_dim: int = 256,
        cnn_hidden_channels: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 9,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

        self.feature_extractor = CNNFeatureExtractor(
            in_channels=in_channels,
            hidden_channels=cnn_hidden_channels,
            embedding_dim=embedding_dim,
        )

        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, L]
        batch_size, seq_len, channels, samples = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
            )

        x = x.reshape(batch_size * seq_len, channels, samples)
        token_features = self.feature_extractor(x)  # [B*T, D]
        token_features = token_features.reshape(batch_size, seq_len, -1)  # [B, T, D]

        token_features = token_features + self.pos_embedding[:, :seq_len, :]
        encoded = self.transformer(token_features)

        center_idx = seq_len // 2
        center_token = encoded[:, center_idx, :]
        return self.classifier(center_token)


def build_model(cfg) -> CNNTransformerClassifier:
    return CNNTransformerClassifier(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        embedding_dim=cfg.model.embedding_dim,
        cnn_hidden_channels=cfg.model.cnn_hidden_channels,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
        max_seq_len=cfg.model.max_seq_len,
    )
