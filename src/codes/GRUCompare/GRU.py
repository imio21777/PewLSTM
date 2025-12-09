import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init
from torch import Tensor
import math

class standard_GRU(nn.Module):
    """
    Standard GRU implementation without periodic history or weather gating.
    This serves as a baseline for comparison with PewGRU.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super(standard_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Reset gate parameters
        self.w_rx = Parameter(Tensor(hidden_size, input_size))
        self.w_rh = Parameter(Tensor(hidden_size, hidden_size))
        self.b_r = Parameter(Tensor(hidden_size, 1))

        # Update gate parameters
        self.w_zx = Parameter(Tensor(hidden_size, input_size))
        self.w_zh = Parameter(Tensor(hidden_size, hidden_size))
        self.b_z = Parameter(Tensor(hidden_size, 1))

        # Candidate hidden state parameters
        self.w_hx = Parameter(Tensor(hidden_size, input_size))
        self.w_hh = Parameter(Tensor(hidden_size, hidden_size))
        self.b_h = Parameter(Tensor(hidden_size, 1))

        self.reset_weights()

    def reset_weights(self):
        """Initialize weights using uniform distribution"""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, x_input):
        """
        Forward pass through GRU
        Args:
            x_input: [batch_size, seq_size, input_size]
        Returns:
            h_output: [batch_size, seq_size, hidden_size]
        """
        batch_size, seq_size, input_dim = x_input.size()

        h_output = torch.zeros(batch_size, seq_size, self.hidden_size)

        for b in range(batch_size):
            h_t = torch.zeros(1, self.hidden_size).t()

            for t in range(seq_size):
                x = x_input[b, t, :].unsqueeze(0).t()  # [input_dim, 1]

                # Reset gate
                r = torch.sigmoid(self.w_rx @ x + self.w_rh @ h_t + self.b_r)

                # Update gate
                z = torch.sigmoid(self.w_zx @ x + self.w_zh @ h_t + self.b_z)

                # Candidate hidden state
                h_tilde = torch.tanh(self.w_hx @ x + self.w_hh @ (r * h_t) + self.b_h)

                # New hidden state
                h_next = (1 - z) * h_t + z * h_tilde

                h_output[b, t] = h_next.t().squeeze(0)
                h_t = h_next

        return h_output
