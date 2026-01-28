import torch
import torch.nn as nn

from .config import Config


class DistributionAwareIdentifier(nn.Module):
    """DACE column identity encoder from quantiles."""

    def __init__(self, d_model, num_bins=10):
        super().__init__()

        self.num_bins = num_bins

        self.stat_encoder = nn.Sequential(
            nn.Linear(num_bins, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x, row_mask):
        B, R, C = x.shape

        x_clean = x.clone()

        if row_mask is not None:
            mask_expanded = row_mask.unsqueeze(-1).expand(-1, -1, C)
            x_clean[mask_expanded] = float("inf")

            valid_counts = (~row_mask).sum(dim=1).float()
        else:
            valid_counts = torch.full((B,), R, device=x.device, dtype=torch.float)

        x_sorted, _ = torch.sort(x_clean, dim=1)

        fracs = torch.linspace(0, 1, self.num_bins, device=x.device)  # (num_bins,)

        max_idx = valid_counts - 1
        indices_float = fracs.unsqueeze(0) * max_idx.unsqueeze(1)  # (B, num_bins)

        indices = indices_float.long().clamp(min=0, max=R - 1)  # (B, num_bins)

        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, C)  # (B, num_bins, C)
        quantiles = torch.gather(x_sorted, dim=1, index=indices_expanded)  # (B, num_bins, C)

        quantiles = quantiles.permute(0, 2, 1)

        return self.stat_encoder(quantiles)  # (Batch, Cols, D_Model)


class PoolingByMultiheadAttention(nn.Module):
    """PMA pooling over a set using a learned seed."""

    def __init__(self, d_model, num_heads=4):
        super().__init__()

        self.seed_vector = nn.Parameter(torch.randn(1, 1, d_model))

        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        B = x.shape[0]

        query = self.seed_vector.repeat(B, 1, 1)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.clone()

        attended, _ = self.mha(query, x, x, key_padding_mask=key_padding_mask)

        return self.norm(attended.squeeze(1))  # (Batch, D_Model)


class ColumnSetEmbedding(nn.Module):
    """Encode columns into per-row vectors with DACE and attention."""

    def __init__(self, d_model, num_bins=10):
        super().__init__()

        self.dace = DistributionAwareIdentifier(d_model, num_bins)

        self.val_encoder = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=Config.N_HEAD,
            dim_feedforward=d_model * 4,
            dropout=Config.DROPOUT,
            batch_first=True,
            norm_first=True,
        )
        self.col_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=1, enable_nested_tensor=False
        )

        self.projection = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())

    def forward(self, x, row_mask, col_mask):
        B, R, C = x.shape
        D = self.dace.stat_encoder[0].out_features  # Get d_model from DACE

        dace_ids = self.dace(x, row_mask)

        val_emb = self.val_encoder(x.unsqueeze(-1))

        dace_expanded = dace_ids.unsqueeze(1).expand(-1, R, -1, -1)
        cell_embeddings = val_emb + dace_expanded  # (B, R, C, D)

        cell_flat = cell_embeddings.view(B * R, C, D)

        if col_mask is not None:
            col_mask_expanded = col_mask.unsqueeze(1).expand(-1, R, -1).reshape(B * R, C).clone()
        else:
            col_mask_expanded = None

        chunk_size = 2048
        attended_list = []

        total_rows = cell_flat.size(0)
        for i in range(0, total_rows, chunk_size):
            end = min(i + chunk_size, total_rows)

            chunk_input = cell_flat[i:end]

            if col_mask_expanded is not None:
                chunk_mask = col_mask_expanded[i:end]
            else:
                chunk_mask = None

            chunk_out = self.col_transformer(chunk_input, src_key_padding_mask=chunk_mask)
            attended_list.append(chunk_out)

        attended = torch.cat(attended_list, dim=0)

        attended = attended.view(B, R, C, D)

        if col_mask is not None:
            mask = (~col_mask).float().unsqueeze(1).unsqueeze(-1)  # (B, 1, C, 1)
            attended = attended * mask
            counts = mask.sum(dim=2)
        else:
            counts = C

        row_vectors = attended.sum(dim=2) / counts

        return self.projection(row_vectors)


class DoubleInvariantTransformer(nn.Module):
    """Model: columns -> rows -> table -> scores."""

    def __init__(self):
        super().__init__()

        d_model = Config.D_MODEL

        num_classes = Config.NUM_CLASSES

        num_bins = Config.NUM_BINS

        num_heads = Config.N_HEAD

        self.col_processor = ColumnSetEmbedding(d_model, num_bins)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=Config.DROPOUT,
            batch_first=True,
            norm_first=True,
        )

        self.row_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.N_LAYERS, enable_nested_tensor=False
        )

        self.row_pooler = PoolingByMultiheadAttention(d_model, num_heads)

        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, row_mask=None, col_mask=None):
        B, R, C = x.shape

        device = x.device

        if row_mask is None:
            row_mask = torch.zeros((B, R), dtype=torch.bool, device=device)

        if col_mask is None:
            col_mask = torch.zeros((B, C), dtype=torch.bool, device=device)

        row_vectors = self.col_processor(x, row_mask, col_mask)  # (B, R, D)

        encoded_rows = self.row_transformer(row_vectors, src_key_padding_mask=row_mask)

        table_vector = self.row_pooler(encoded_rows, key_padding_mask=row_mask)  # (B, D)

        return torch.sigmoid(self.classifier(table_vector))
