"""
Manifold-Constrained Hyper-Connections (mHC).

Maintains n parallel residual streams instead of one. A sublayer reads one
input vector h from the streams, runs, and its output y is written back into
the streams.

Two learnable mechanisms control the flow:

1. **Mixing matrices A (pre-sublayer) and B (post-sublayer).** A mixes streams
   before the read; B mixes streams after the write. Optional — controlled by
   `mix`. When `constrain` is True (the manifold constraint) each matrix is
   produced from its free parameter via Sinkhorn normalization, projecting onto
   the Birkhoff polytope (doubly-stochastic matrices); when False the raw
   parameter matrix is used directly (the unconstrained ablation). The logits are
   initialized to a scaled identity so the effective matrices start at ~identity,
   i.e. mHC begins at the standard residual baseline.

2. **Stream selection** — how the sublayer input is read from the streams and how
   its output is written back. Three modes:
     - "static":   read stream 0, write to stream 0 (fixed index convention).
     - "learnable": read/write are softmax-weighted combinations over streams,
                    driven by learnable per-stream logit vectors (constant across
                    tokens). Initialized to favour stream 0 so it starts at the
                    static configuration and can move from there.
     - "per_token": same as "learnable" but the softmax weights are produced from
                    the token's hidden state by a small linear gate, so selection
                    is data-dependent (varies per position). Gate initialized to be
                    data-independent and favour stream 0 at start.

Note on "static vs learnable": even in "static" mode the read is
(A @ S)[0] = sum_j A[0, j] S_j and the write is distributed as column 0 of B, so
A/B already make the effective read/write partly learnable. The selection modes
make that choice explicit and decouple it from the index-0 convention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _slot0_logits(n_streams: int, val: float = 4.0) -> torch.Tensor:
    """Logit vector whose softmax concentrates on stream 0 (≈ static read/write)."""
    v = torch.zeros(n_streams)
    v[0] = val
    return v


class HyperConnection(nn.Module):
    # Diagonal scale for the mixing-logit init. Sinkhorn(eye * scale) -> ~identity
    # so the effective A/B start at (near) identity and mHC begins at the standard
    # residual baseline. At 1.0 the init is far from identity (A[0,0]~0.48), which
    # would start every mHC run from a degraded state vs vanilla.
    MIX_INIT_SCALE = 4.0

    def __init__(
        self,
        n_streams: int,
        selection: str = "static",
        d_model: int = None,
        mix: bool = True,
        constrain: bool = True,
    ):
        super().__init__()
        assert selection in ("static", "learnable", "per_token")
        self.n_streams = n_streams
        self.selection = selection
        self.mix = mix
        self.constrain = constrain

        if mix:
            self.A_logits = nn.Parameter(torch.eye(n_streams) * self.MIX_INIT_SCALE)
            self.B_logits = nn.Parameter(torch.eye(n_streams) * self.MIX_INIT_SCALE)

        if selection == "learnable":
            self.read_logits = nn.Parameter(_slot0_logits(n_streams))
            self.write_logits = nn.Parameter(_slot0_logits(n_streams))
        elif selection == "per_token":
            assert d_model is not None, "per_token selection needs d_model"
            self.read_gate = nn.Linear(d_model, n_streams, bias=True)
            self.write_gate = nn.Linear(d_model, n_streams, bias=True)
            # Start data-independent and favouring stream 0 (≈ static).
            nn.init.zeros_(self.read_gate.weight)
            nn.init.zeros_(self.write_gate.weight)
            with torch.no_grad():
                self.read_gate.bias.copy_(_slot0_logits(n_streams))
                self.write_gate.bias.copy_(_slot0_logits(n_streams))

    def sinkhorn(self, logits: torch.Tensor, n_iters: int = 20):
        """
        Sinkhorn normalization: project logits onto the Birkhoff polytope.

        Args:
            logits: (n_streams, n_streams) unnormalized routing matrix
            n_iters: number of Sinkhorn iterations
        Returns:
            A doubly stochastic matrix (n_streams, n_streams)
        """
        A = torch.exp(logits)
        for _ in range(n_iters):
            A = A / A.sum(dim=1, keepdim=True)  # Row normalization
            A = A / A.sum(dim=0, keepdim=True)  # Column normalization
        return A

    def mix_matrix(self, logits: torch.Tensor) -> torch.Tensor:
        """Effective mixing matrix: Sinkhorn-projected if constrained, else raw."""
        return self.sinkhorn(logits) if self.constrain else logits

    def _read(self, S_routed: torch.Tensor) -> torch.Tensor:
        """Extract sublayer input h (batch, seq, d_model) from streams."""
        if self.selection == "static":
            return S_routed[..., 0, :]
        if self.selection == "learnable":
            w = F.softmax(self.read_logits, dim=0)            # (n_streams,)
            return torch.einsum("i,...id->...d", w, S_routed)
        # per_token
        summary = S_routed.mean(dim=-2)                       # (batch, seq, d_model)
        w = F.softmax(self.read_gate(summary), dim=-1)        # (batch, seq, n_streams)
        return torch.einsum("...i,...id->...d", w, S_routed)

    def _write(self, S_routed: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Build the per-stream update tensor for sublayer output y."""
        if self.selection == "static":
            update = torch.zeros_like(S_routed)
            update[..., 0, :] = y
            return update
        if self.selection == "learnable":
            w = F.softmax(self.write_logits, dim=0)           # (n_streams,)
            return torch.einsum("i,...d->...id", w, y)
        # per_token
        summary = S_routed.mean(dim=-2)
        w = F.softmax(self.write_gate(summary), dim=-1)       # (batch, seq, n_streams)
        return torch.einsum("...i,...d->...id", w, y)

    def route_in(self, S: torch.Tensor):
        """
        Mix streams (optional) and extract sublayer input.

        Args:
            S: (batch, seq, n_streams, d_model)
        Returns:
            S_routed: mixed state (batch, seq, n_streams, d_model)
            h: sublayer input (batch, seq, d_model)
        """
        if self.mix:
            A = self.mix_matrix(self.A_logits)
            S_routed = torch.einsum("ij,...jd->...id", A, S)
        else:
            S_routed = S
        h = self._read(S_routed)
        return S_routed, h

    def route_out(self, S_routed: torch.Tensor, y: torch.Tensor):
        """
        Write sublayer output back into streams and mix (optional).

        Args:
            S_routed: mixed state from route_in (batch, seq, n_streams, d_model)
            y: sublayer output (batch, seq, d_model)
        Returns:
            S_new: updated state (batch, seq, n_streams, d_model)
        """
        state = S_routed + self._write(S_routed, y)
        if self.mix:
            B = self.mix_matrix(self.B_logits)
            return torch.einsum("ij,...jd->...id", B, state)
        return state
