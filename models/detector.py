"""
Leak detector for single-pipe abrupt leaks (softmax multi-class):

Pipeline:
  residual sensors (B, L_det, S) + time feats (B, L_det, 9)
    -> SharedSensorGRUEncoder produces per-sensor embeddings (B, S, d_s)
    -> Scatter to full graph nodes + sensor mask
    -> Spatial GCN layers over full WDN topology
    -> EdgeHead logits for each pipe in pipe_ids_in_order
    -> NoLeakHead logit
    -> concat => logits (B, num_pipes+1)

Important:
- sensor_node_ids must match CSV columns (node IDs in .inp)
- pipe_ids_in_order MUST match dataset pipe ordering (class indices)
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from .utils import WDNGraph, build_wdn_graph_from_inp


class SharedSensorGRUEncoder(nn.Module):
    """
    Encodes each sensor's residual time series using a shared GRU.

    Inputs:
      r:      (B, L, S)
      tfeat:  (B, L, 9) aligned to r

    Output:
      h_s:    (B, S, d)
    """
    def __init__(
        self,
        time_dim: int = 9,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_time: bool = True,
    ) -> None:
        super().__init__()
        self.use_time = bool(use_time)
        self.hidden_size = int(hidden_size)
        in_dim = 1 + (time_dim if self.use_time else 0)

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, r: torch.Tensor, tfeat: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, S = r.shape
        rr = r.transpose(1, 2).contiguous().view(B * S, L, 1)  # (B*S, L, 1)
        if self.use_time:
            if tfeat is None:
                raise ValueError("tfeat required when use_time=True")
            tf = tfeat.unsqueeze(1).repeat(1, S, 1, 1).contiguous().view(B * S, L, -1)  # (B*S, L, 9)
            inp = torch.cat([rr, tf], dim=-1)  # (B*S, L, 10)
        else:
            inp = rr

        out, _ = self.gru(inp)
        h_last = out[:, -1, :]  # (B*S, d)
        return h_last.view(B, S, -1)


class EdgeHead(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_u: torch.Tensor, h_v: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([h_u, h_v, (h_u - h_v).abs()], dim=-1)  # (B,P,3D)
        return self.mlp(feat).squeeze(-1)  # (B,P)


class NoLeakHead(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.mlp(pooled).squeeze(-1)  # (B,)


def _batchify_edge_index(edge_index_single: torch.Tensor, num_nodes: int, batch_size: int) -> torch.Tensor:
    """
    Build a disjoint-union edge_index for B identical graphs by offsetting node indices.
    edge_index_single: (2, E)
    return: (2, E*B)
    """
    E = edge_index_single.size(1)
    edge_index = edge_index_single.repeat(1, batch_size)  # (2, E*B)
    offset = (torch.arange(batch_size, device=edge_index_single.device).repeat_interleave(E) * num_nodes)  # (E*B,)
    return edge_index + offset.unsqueeze(0)


class LeakDetector(nn.Module):
    """
    residual: (B, L_det, S)
    tfeat:    (B, L_det, 9)
    logits:   (B, num_pipes+1)
    """
    def __init__(
        self,
        inp_path: str | Path,
        sensor_node_ids: Sequence[str],
        pipe_ids_in_order: Sequence[str],
        sensor_hidden: int = 64,
        node_hidden: int = 64,
        gnn_layers: int = 2,
        dropout: float = 0.1,
        use_time: bool = True,
        include_links: Sequence[str] = ("PIPES", "PUMPS", "VALVES"),
    ) -> None:
        super().__init__()

        self.graph: WDNGraph = build_wdn_graph_from_inp(
            inp_path=inp_path,
            sensor_node_ids=sensor_node_ids,
            pipe_ids_in_order=pipe_ids_in_order,
            include_links=include_links,
            add_self_loops=False,
            make_undirected=True,
        )

        self.node_names = self.graph.node_names
        self.node_to_idx = self.graph.node_to_idx
        
        self.pipe_ids = self.graph.pipe_ids
        self.pipe_to_idx = self.graph.pipe_to_idx
        self.pipe_ends = torch.tensor(self.graph.pipe_ends, dtype=torch.long)  # (P,2)
        self.edge_index_single = self.graph.edge_index  # (2,E)

        self.sensor_node_ids = list(sensor_node_ids)
        self.sensor_node_idx = torch.tensor([self.node_to_idx[n] for n in self.sensor_node_ids], dtype=torch.long)

        # layers and operations
        self.sensor_encoder = SharedSensorGRUEncoder(hidden_size=sensor_hidden, use_time=use_time)
        
        self.sensor_to_node = nn.Linear(sensor_hidden + 1, node_hidden) # node init: (sensor_hidden + mask) -> node_hidden

        self.convs = nn.ModuleList(
            [GCNConv(node_hidden, node_hidden, add_self_loops=True, normalize=True) for _ in range(gnn_layers)]
        )
        self.dropout = nn.Dropout(dropout)

        self.edge_head = EdgeHead(node_hidden, hidden_dim=128, dropout=dropout)
        self.noleak_head = NoLeakHead(node_hidden, hidden_dim=128, dropout=dropout)

    def forward(self, residual: torch.Tensor, tfeat: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, S = residual.shape
        device = residual.device
        N = len(self.node_names)

        # Encode sensors -> (B,S,d_s)
        h_s = self.sensor_encoder(residual, tfeat)

        # Init node features: zeros then scatter sensor embeddings
        h0 = torch.zeros(B, N, h_s.shape[-1], device=device, dtype=residual.dtype)
        idx = self.sensor_node_idx.to(device)
        h0[:, idx, :] = h_s

        # Add sensor mask, project to node_hidden
        mask = torch.zeros(N, 1, device=device, dtype=residual.dtype)
        mask[idx, 0] = 1.0
        mask = mask.unsqueeze(0).expand(B, -1, -1)  # (B,N,1)

        h = torch.cat([h0, mask], dim=-1)  # (B,N,d_s+1)
        h = F.relu(self.sensor_to_node(h))  # (B,N,node_hidden)
        h = self.dropout(h)

        # PyG expects (num_nodes_total, feat_dim) and a single edge_index (2, E_total)
        x = h.reshape(B * N, -1)  # (B*N, D)

        edge_index_single = self.edge_index_single.to(device)
        edge_index = _batchify_edge_index(edge_index_single, num_nodes=N, batch_size=B)
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        # Back to (B,N,D)
        h_nodes = x.view(B, N, -1)
        # Pipe logits
        ends = self.pipe_ends.to(device)
        u = ends[:, 0]
        v = ends[:, 1]
        h_u = h_nodes[:, u, :]
        h_v = h_nodes[:, v, :]
        pipe_logits = self.edge_head(h_u, h_v)  # (B,P)

        # No-leak logit with global pooling
        batch = torch.arange(B, device=device).repeat_interleave(N)  # (B*N,)
        pooled = global_mean_pool(x, batch)  # (B,D)
        noleak_logit = self.noleak_head(pooled).unsqueeze(-1)  # (B,1)

        return torch.cat([pipe_logits, noleak_logit], dim=-1)  # (B,P+1)