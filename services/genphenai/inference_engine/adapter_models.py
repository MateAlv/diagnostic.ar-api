from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _is_single_graph(batch: Optional[torch.Tensor]) -> bool:
    if batch is None or batch.numel() == 0:
        return True
    return bool((batch == batch[0]).all())


class GCNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        if num_nodes == 0:
            return x

        src = edge_index[0].long()
        dst = edge_index[1].long()
        loop = torch.arange(num_nodes, device=x.device, dtype=torch.long)
        src = torch.cat([src, loop], dim=0)
        dst = torch.cat([dst, loop], dim=0)

        deg = torch.zeros(num_nodes, dtype=x.dtype, device=x.device)
        deg.index_add_(0, dst, torch.ones(dst.shape[0], dtype=x.dtype, device=x.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        xw = F.linear(x, self.weight)
        out = torch.zeros_like(xw)
        out.index_add_(0, dst, xw[src] * norm.unsqueeze(-1))
        if self.bias is not None:
            out = out + self.bias
        return out


class GINLayer(nn.Module):
    def __init__(self, mlp: nn.Module, eps_init: float = 0.0, train_eps: bool = True):
        super().__init__()
        self.mlp = mlp
        eps = torch.tensor([eps_init], dtype=torch.float32)
        if train_eps:
            self.eps = nn.Parameter(eps)
        else:
            self.register_buffer("eps", eps)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = edge_index[0].long()
        dst = edge_index[1].long()
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, x[src])
        out = (1.0 + self.eps) * x + aggregated
        return self.mlp(out)


class AttentionPool(nn.Module):
    def __init__(self, gate_nn: nn.Module):
        super().__init__()
        self.gate_nn = gate_nn

    def _pool_single(self, x: torch.Tensor) -> torch.Tensor:
        score = self.gate_nn(x).squeeze(-1)
        weights = torch.softmax(score, dim=0).unsqueeze(-1)
        return (weights * x).sum(dim=0, keepdim=True)

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if _is_single_graph(batch):
            return self._pool_single(x)

        assert batch is not None
        pooled = []
        unique_batches = torch.unique(batch, sorted=True)
        for batch_id in unique_batches.tolist():
            mask = batch == batch_id
            pooled.append(self._pool_single(x[mask]).squeeze(0))
        return torch.stack(pooled, dim=0)


class GenePhenAIv2_0Adapter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout_prob: float = 0.25,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.conv1 = GCNLayer(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNLayer(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNLayer(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.conv4 = GCNLayer(hidden_channels, hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels)

        gate_nn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.attention_pool = AttentionPool(gate_nn=gate_nn)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.dropout(F.relu(self.bn1(self.conv1(x, edge_index))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x, edge_index))))
        x = self.dropout(F.relu(self.bn3(self.conv3(x, edge_index))))
        x = self.dropout(F.relu(self.bn4(self.conv4(x, edge_index))))

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        pooled = self.dropout(self.attention_pool(x, batch))
        return self.classifier(pooled)


class GenePhenAIv2_5Adapter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int = 4,
        dropout_prob: float = 0.25,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.conv1 = GCNLayer(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNLayer(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNLayer(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.mha_pool = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=False,
        )
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def _pool_single(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.unsqueeze(1)
        attn_out, _ = self.mha_pool(seq, seq, seq)
        return attn_out.mean(dim=0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.dropout(F.relu(self.bn1(self.conv1(x, edge_index))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x, edge_index))))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))

        if batch is None or _is_single_graph(batch):
            pooled = self._pool_single(x)
        else:
            pooled_list = []
            unique_batches = torch.unique(batch, sorted=True)
            for batch_id in unique_batches.tolist():
                pooled_list.append(self._pool_single(x[batch == batch_id]).squeeze(0))
            pooled = torch.stack(pooled_list, dim=0)

        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class GenePhenAIv3_0Adapter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout_prob: float = 0.30,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        mlp3 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.conv1 = GINLayer(mlp1, train_eps=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GINLayer(mlp2, train_eps=True)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GINLayer(mlp3, train_eps=True)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        gate_nn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.attention_pool = AttentionPool(gate_nn=gate_nn)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.dropout(F.relu(self.bn1(self.conv1(x, edge_index))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x, edge_index))))
        x = self.dropout(F.relu(self.bn3(self.conv3(x, edge_index))))

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        pooled = self.dropout(self.attention_pool(x, batch))
        return self.classifier(pooled)


def build_adapter_model(
    model_version: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_heads: int,
) -> nn.Module:
    if model_version == "2.0":
        return GenePhenAIv2_0Adapter(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
        )
    if model_version == "2.5":
        return GenePhenAIv2_5Adapter(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_heads=num_heads,
        )
    if model_version == "3.0":
        return GenePhenAIv3_0Adapter(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
        )
    raise ValueError(f"Unsupported model_version: {model_version}")


def convert_checkpoint_state_dict(
    raw_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in raw_state_dict.items():
        new_key = key

        # PyG GCNConv uses convX.lin.weight; adapters use convX.weight.
        new_key = new_key.replace(".lin.weight", ".weight")

        # PyG BatchNorm stores parameters under bnX.module.* in many versions.
        new_key = new_key.replace(".module.", ".")

        # PyG GINConv uses convX.nn.*; adapters use convX.mlp.*.
        new_key = new_key.replace(".nn.", ".mlp.")

        converted[new_key] = value
    return converted
