"""
Manifold-Constrained Hyper-Connections (mHC).

Maintains n parallel residual streams instead of one. Learnable routing
matrices A (pre-sublayer) and B (post-sublayer) mix information between
streams. A and B are initialized to identity so that at init, stream 0
behaves like a standard residual connection and streams 1..n-1 are inert.

Manifold constraint: A and B are projected onto O(n) after each optimizer
step via Newton-Schulz retraction.
"""
import torch
import torch.nn as nn


class HyperConnection(nn.Module):
    def __init__(self, n_streams: int):
        super().__init__()
        self.n_streams = n_streams
        self.A = nn.Parameter(torch.eye(n_streams))
        self.B = nn.Parameter(torch.eye(n_streams))

    def route_in(self, S: torch.Tensor):
        """
        Mix streams and extract sublayer input.

        Args:
            S: (batch, seq, n_streams, d_model)
        Returns:
            S_routed: mixed state (batch, seq, n_streams, d_model)
            h: sublayer input from stream 0 (batch, seq, d_model)
        """
        S_routed = torch.einsum("ij,...jd->...id", self.A, S)
        h = S_routed[..., 0, :]
        return S_routed, h

    def route_out(self, S_routed: torch.Tensor, y: torch.Tensor):
        """
        Distribute sublayer output back into streams.

        Args:
            S_routed: mixed state from route_in (batch, seq, n_streams, d_model)
            y: sublayer output (batch, seq, d_model)
        Returns:
            S_new: updated state (batch, seq, n_streams, d_model)
        """
        update = torch.zeros_like(S_routed)
        update[..., 0, :] = y
        S_new = torch.einsum("ij,...jd->...id", self.B, S_routed + update)
        return S_new
