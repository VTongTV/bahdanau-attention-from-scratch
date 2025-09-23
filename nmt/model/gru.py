"""gated hidden unit of cho 2014, built from scratch."""

import torch
import torch.nn as nn

from nmt.model.params import gaussian, orthogonal, zero
from nmt.config import WEIGHT_INIT_STD


class GRUCell(nn.Module):
    """gru with update and reset gates. context input is optional."""

    def __init__(self, input_size: int, hidden_size: int, context_size: int = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.w = nn.Linear(input_size, hidden_size, bias=False)
        self.u = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_z = nn.Linear(input_size, hidden_size, bias=False)
        self.u_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_r = nn.Linear(input_size, hidden_size, bias=False)
        self.u_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        self.c: nn.Linear | None = None
        self.c_z: nn.Linear | None = None
        self.c_r: nn.Linear | None = None
        if context_size > 0:
            self.c = nn.Linear(context_size, hidden_size, bias=False)
            self.c_z = nn.Linear(context_size, hidden_size, bias=False)
            self.c_r = nn.Linear(context_size, hidden_size, bias=False)

    def forward(self, x, state, context=None):
        """one step. x (batch, input). state (batch, hidden)."""
        z = self.w_z(x) + self.u_z(state) + self.b_z
        r = self.w_r(x) + self.u_r(state) + self.b_r
        h_in = self.w(x) + self.b
        if context is not None and self.c is not None:
            z = z + self.c_z(context)
            r = r + self.c_r(context)
            h_in = h_in + self.c(context)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        h_tilde = torch.tanh(h_in + self.u(state * r))
        return (1 - z) * state + z * h_tilde

    def init_parameters(self) -> None:
        """orthogonal recurrent matrices. gaussian input and context."""
        for u in (self.u, self.u_z, self.u_r):
            orthogonal(u.weight)
        for lin in (self.w, self.w_z, self.w_r, self.c, self.c_z, self.c_r):
            if lin is not None:
                gaussian(lin.weight, WEIGHT_INIT_STD)
        for b in (self.b, self.b_z, self.b_r):
            zero(b)


class ContextFreeCell(GRUCell):
    """gru without context terms, for the encoder."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__(input_size, hidden_size, context_size=0)