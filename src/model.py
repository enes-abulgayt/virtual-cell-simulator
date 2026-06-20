"""PyTorch model for virtual cell perturbation simulation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class VirtualCellModel(nn.Module):
    """Predict perturbation-induced expression deltas.

    The model encodes the control cell, combines that representation with a
    learned perturbation embedding, and predicts a gene-wise expression delta.
    """

    def __init__(
        self,
        num_genes: int,
        num_perturbations: int,
        embedding_dim: int = 32,
        encoder_dim: int = 128,
        hidden_dim: int = 192,
        dropout: float = 0.10,
        num_perturbation_groups: int = 0,
        perturbation_group_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.num_perturbations = num_perturbations
        self.num_perturbation_groups = num_perturbation_groups

        self.control_encoder = nn.Sequential(
            nn.Linear(num_genes, encoder_dim),
            nn.GELU(),
            nn.LayerNorm(encoder_dim),
            nn.Dropout(dropout),
        )
        self.perturbation_embedding = nn.Embedding(
            num_perturbations + 1,
            embedding_dim,
        )
        fusion_input_dim = encoder_dim + embedding_dim
        if num_perturbation_groups > 0:
            if perturbation_group_ids is None:
                raise ValueError(
                    "perturbation_group_ids are required when group embeddings "
                    "are enabled."
                )
            group_ids = torch.as_tensor(perturbation_group_ids, dtype=torch.long)
            if group_ids.numel() != num_perturbations + 1:
                raise ValueError(
                    "perturbation_group_ids must include control and every "
                    "perturbation."
                )
            self.register_buffer("perturbation_group_ids", group_ids)
            self.group_embedding = nn.Embedding(
                num_perturbation_groups + 1,
                embedding_dim,
            )
            fusion_input_dim += embedding_dim
        else:
            self.group_embedding = None

        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(hidden_dim, num_genes)

        nn.init.normal_(self.perturbation_embedding.weight, mean=0.0, std=0.05)
        if self.group_embedding is not None:
            nn.init.normal_(self.group_embedding.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self,
        normalized_control_expression: torch.Tensor,
        perturbation_id: torch.Tensor,
        control_expression: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict delta and perturbed expression for a batch of cells."""

        control_features = self.control_encoder(normalized_control_expression)
        perturbation_features = self.perturbation_embedding(perturbation_id)
        feature_parts = [control_features, perturbation_features]
        if self.group_embedding is not None:
            group_ids = self.perturbation_group_ids[perturbation_id]
            feature_parts.append(self.group_embedding(group_ids))
        joint_features = torch.cat(feature_parts, dim=-1)
        predicted_delta = self.delta_head(self.fusion_network(joint_features))

        if control_expression is None:
            control_expression = normalized_control_expression
        predicted_expression = control_expression + predicted_delta
        return {
            "predicted_delta": predicted_delta,
            "predicted_expression": predicted_expression,
        }
