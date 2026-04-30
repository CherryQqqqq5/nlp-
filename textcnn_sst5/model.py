from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_classes: int = 5,
        num_filters: int = 100,
        kernel_sizes: Iterable[int] = (3, 4, 5),
        dropout: float = 0.5,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embedding: bool = False,
    ):
        super().__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=freeze_embedding,
                padding_idx=pad_idx,
            )
            if pretrained_embeddings.size(1) != embed_dim:
                raise ValueError(
                    f"embed_dim ({embed_dim}) must match pretrained dimension ({pretrained_embeddings.size(1)})"
                )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(tuple(kernel_sizes)), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)  # [B, L, E]
        x = x.transpose(1, 2)  # [B, E, L]

        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x))  # [B, F, L-k+1]
            h = F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2)  # [B, F]
            pooled.append(h)

        x = torch.cat(pooled, dim=1)
        x = self.dropout(x)
        return self.fc(x)
