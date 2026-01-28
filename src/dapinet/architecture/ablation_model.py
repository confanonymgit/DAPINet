"""Ablation variants of DoubleInvariantTransformer."""

from enum import Enum

import torch
import torch.nn as nn

from .config import Config


class AblationVariant(str, Enum):
    """Supported ablation variants."""

    DACE_SIMPLE = "dace_simple"
    POOL_MEAN = "pool_mean"
    NO_COL_ATTN = "no_col_attn"
    POSITIONAL = "positional"


class SimpleStatsEncoder(nn.Module):
    """Compute simple per-column stats and project to d_model."""

    def __init__(self, d_model: int, stats_dim: int = 5):
        super().__init__()
        self.projection = nn.Linear(stats_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, row_mask: torch.Tensor | None) -> torch.Tensor:
        """Return (B, C, D) stats embedding."""
        B, R, C = x.shape

        if row_mask is not None:
            mask = (~row_mask).float().unsqueeze(-1)  # (B, R, 1)
            x_masked = x * mask
            valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1, 1)

            col_mean = x_masked.sum(dim=1) / valid_counts.squeeze(1)  # (B, C)

            diff_sq = ((x - col_mean.unsqueeze(1)) * mask) ** 2
            col_std = (diff_sq.sum(dim=1) / valid_counts.squeeze(1)).sqrt()

            x_for_minmax = x.clone()
            x_for_minmax[row_mask.unsqueeze(-1).expand(-1, -1, C)] = float("inf")
            col_min = x_for_minmax.min(dim=1).values

            x_for_max = x.clone()
            x_for_max[row_mask.unsqueeze(-1).expand(-1, -1, C)] = float("-inf")
            col_max = x_for_max.max(dim=1).values

            col_median = x_masked.median(dim=1).values
        else:
            col_mean = x.mean(dim=1)
            col_std = x.std(dim=1)
            col_min = x.min(dim=1).values
            col_max = x.max(dim=1).values
            col_median = x.median(dim=1).values

        stats = torch.stack([col_mean, col_std, col_min, col_max, col_median], dim=-1)

        return self.norm(self.projection(stats))  # (B, C, D)


class MeanPooling(nn.Module):
    """Mean pooling over a sequence."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Mean-pool with optional padding mask."""
        if key_padding_mask is not None:
            mask = (~key_padding_mask).float().unsqueeze(-1)  # (B, S, 1)
            x_masked = x * mask
            pooled = x_masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        return self.norm(pooled)


class AblationDoubleInvariantTransformer(nn.Module):
    """DoubleInvariantTransformer with ablation switches."""

    def __init__(self, variant: AblationVariant):
        super().__init__()

        self.variant = variant
        d_model = Config.D_MODEL
        num_heads = Config.N_HEAD
        num_bins = Config.NUM_BINS
        dropout = Config.DROPOUT
        num_classes = Config.NUM_CLASSES

        if self.variant == AblationVariant.DACE_SIMPLE:
            self.col_identity = SimpleStatsEncoder(d_model, stats_dim=5)
        else:
            from .model import DistributionAwareIdentifier

            self.col_identity = DistributionAwareIdentifier(d_model, num_bins)

        self.val_encoder = nn.Linear(1, d_model)

        if self.variant == AblationVariant.POSITIONAL:
            self.col_pos_embedding = nn.Embedding(Config.MAX_COLS, d_model)
        else:
            self.col_pos_embedding = None

        if self.variant == AblationVariant.NO_COL_ATTN:
            self.col_transformer = None
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.col_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=1, enable_nested_tensor=False
            )

        self.col_projection = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())

        row_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.row_transformer = nn.TransformerEncoder(
            row_encoder_layer, num_layers=Config.N_LAYERS, enable_nested_tensor=False
        )

        if self.variant == AblationVariant.POOL_MEAN:
            self.row_pooler = MeanPooling(d_model)
        else:
            from .model import PoolingByMultiheadAttention

            self.row_pooler = PoolingByMultiheadAttention(d_model, num_heads)

        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        row_mask: torch.Tensor | None = None,
        col_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional masks."""
        B, R, C = x.shape
        device = x.device
        D = Config.D_MODEL

        if row_mask is None:
            row_mask = torch.zeros((B, R), dtype=torch.bool, device=device)
        if col_mask is None:
            col_mask = torch.zeros((B, C), dtype=torch.bool, device=device)

        col_ids = self.col_identity(x, row_mask)  # (B, C, D)

        val_emb = self.val_encoder(x.unsqueeze(-1))  # (B, R, C, D)

        col_ids_expanded = col_ids.unsqueeze(1).expand(-1, R, -1, -1)  # (B, R, C, D)
        cell_embeddings = val_emb + col_ids_expanded

        if self.col_pos_embedding is not None:
            col_positions = torch.arange(C, device=device)
            pos_emb = self.col_pos_embedding(col_positions)  # (C, D)
            cell_embeddings = cell_embeddings + pos_emb.unsqueeze(0).unsqueeze(0)

        if self.col_transformer is not None:
            cell_flat = cell_embeddings.view(B * R, C, D)

            if col_mask is not None:
                col_mask_expanded = col_mask.unsqueeze(1).expand(-1, R, -1).reshape(B * R, C)
            else:
                col_mask_expanded = None

            chunk_size = 2048
            attended_list = []
            for i in range(0, B * R, chunk_size):
                end = min(i + chunk_size, B * R)
                chunk_input = cell_flat[i:end]
                chunk_mask = col_mask_expanded[i:end] if col_mask_expanded is not None else None
                chunk_out = self.col_transformer(chunk_input, src_key_padding_mask=chunk_mask)
                attended_list.append(chunk_out)

            attended = torch.cat(attended_list, dim=0).view(B, R, C, D)
        else:
            attended = cell_embeddings

        if col_mask is not None:
            mask = (~col_mask).float().unsqueeze(1).unsqueeze(-1)  # (B, 1, C, 1)
            attended = attended * mask
            counts = mask.sum(dim=2).clamp(min=1)
        else:
            counts = C

        row_vectors = attended.sum(dim=2) / counts  # (B, R, D)
        row_vectors = self.col_projection(row_vectors)

        encoded_rows = self.row_transformer(row_vectors, src_key_padding_mask=row_mask)

        table_vector = self.row_pooler(encoded_rows, key_padding_mask=row_mask)

        return torch.sigmoid(self.classifier(table_vector))


def create_ablation_model(variant_name: str) -> AblationDoubleInvariantTransformer:
    """Create ablation model from variant name."""
    variant = AblationVariant(variant_name)
    return AblationDoubleInvariantTransformer(variant)
