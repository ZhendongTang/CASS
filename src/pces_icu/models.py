from __future__ import annotations

from dataclasses import dataclass


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for model training. Install requirements.txt on the GPU server.") from exc
    return torch


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    hidden_size: int = 192
    layers: int = 4
    heads: int = 6
    dropout: float = 0.1
    max_length: int = 512


def build_event_transformer(config: TransformerConfig):
    torch = require_torch()
    nn = torch.nn

    class EventTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.concept = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
            self.position = nn.Embedding(config.max_length, config.hidden_size)
            self.numeric = nn.Sequential(nn.Linear(3, config.hidden_size), nn.LayerNorm(config.hidden_size))
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size, nhead=config.heads, dim_feedforward=config.hidden_size * 4,
                dropout=config.dropout, batch_first=True, norm_first=True, activation="gelu"
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
            self.norm = nn.LayerNorm(config.hidden_size)
            self.head = nn.Linear(config.hidden_size, 1)

        def forward(self, concept_ids, numeric_features, attention_mask):
            positions = torch.arange(concept_ids.shape[1], device=concept_ids.device).unsqueeze(0)
            hidden = self.concept(concept_ids) + self.position(positions) + self.numeric(numeric_features)
            hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask.bool())
            masked = hidden.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)
            pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
            return self.head(self.norm(pooled)).squeeze(-1)

    return EventTransformer()


def symmetric_bernoulli_kl(logits_a, logits_b):
    torch = require_torch()
    pa = torch.sigmoid(logits_a).clamp(1e-6, 1 - 1e-6)
    pb = torch.sigmoid(logits_b).clamp(1e-6, 1 - 1e-6)
    kl_ab = pa * torch.log(pa / pb) + (1 - pa) * torch.log((1 - pa) / (1 - pb))
    kl_ba = pb * torch.log(pb / pa) + (1 - pb) * torch.log((1 - pb) / (1 - pa))
    return 0.5 * (kl_ab + kl_ba).mean()

