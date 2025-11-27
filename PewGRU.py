import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init
from torch import Tensor
import math

class pew_GRU(nn.Module):
    """
    Periodic GRU with Weather-Aware Gating Mechanism.
    This is the GRU version of PewLSTM, incorporating:
    1. Periodic history features (day, week, month)
    2. Weather-aware gating mechanism
    3. Historical observation gating (h_o)
    """
    def __init__(self, input_size: int, hidden_size: int, weather_size: int):
        super(pew_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weather_size = weather_size

        # Reset gate parameters
        self.w_rx = Parameter(Tensor(hidden_size, input_size))
        self.w_rh = Parameter(Tensor(hidden_size, hidden_size))
        self.w_re = Parameter(Tensor(hidden_size, hidden_size))  # weather gating
        self.b_r = Parameter(Tensor(hidden_size, 1))

        # Update gate parameters
        self.w_zx = Parameter(Tensor(hidden_size, input_size))
        self.w_zh = Parameter(Tensor(hidden_size, hidden_size))
        self.w_ze = Parameter(Tensor(hidden_size, hidden_size))  # weather gating
        self.b_z = Parameter(Tensor(hidden_size, 1))

        # Candidate hidden state parameters
        self.w_hx = Parameter(Tensor(hidden_size, input_size))
        self.w_hh = Parameter(Tensor(hidden_size, hidden_size))
        self.b_h = Parameter(Tensor(hidden_size, 1))

        # Historical observation (h_o) parameters
        self.w_d = Parameter(Tensor(hidden_size, input_size))  # day
        self.w_w = Parameter(Tensor(hidden_size, input_size))  # week
        self.w_m = Parameter(Tensor(hidden_size, input_size))  # month
        self.w_t = Parameter(Tensor(hidden_size, hidden_size))  # current hidden

        # Weather embedding parameters
        self.w_e = Parameter(Tensor(hidden_size, weather_size))
        self.b_e = Parameter(Tensor(hidden_size, 1))

        self.reset_weights()

    def reset_weights(self):
        """Initialize weights using uniform distribution"""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, x_input, x_weather):
        """
        Forward pass through PewGRU
        Args:
            x_input: [batch_size, seq_size, input_size]
            x_weather: [batch_size, seq_size, weather_size]
        Returns:
            h_output: [batch_size, seq_size, hidden_size]
        """
        batch_size, seq_size, input_dim = x_input.size()

        h_output = torch.zeros(batch_size, seq_size, self.hidden_size)

        for b in range(batch_size):
            h_t = torch.zeros(1, self.hidden_size).t()

            for t in range(24):  # Fixed to 24 hours per day
                # Extract periodic history features
                # Day period (previous day, same hour)
                if b < 1:
                    h_d = torch.zeros(1, self.input_size).t()
                else:
                    h_d = x_input[b-1, t, :].unsqueeze(0).t()

                # Week period (previous week, same hour)
                if b < 7:
                    h_w = torch.zeros(1, self.input_size).t()
                else:
                    h_w = x_input[b-7, t, :].unsqueeze(0).t()

                # Month period (30 days ago, same hour)
                if b < 30:
                    h_m = torch.zeros(1, self.input_size).t()
                else:
                    h_m = x_input[b-30, t, :].unsqueeze(0).t()

                # Current input and weather
                x = x_input[b, t, :].unsqueeze(0).t()  # [input_dim, 1]
                weather_t = x_weather[b, t, :].unsqueeze(0).t()  # [weather_dim, 1]

                # Compute historical observation gating (h_o)
                # This replaces the traditional h_{t-1} in GRU gates
                h_o = torch.sigmoid(self.w_d @ h_d + self.w_w @ h_w + 
                                   self.w_m @ h_m + self.w_t @ h_t)

                # Compute weather embedding
                e_t = torch.sigmoid(self.w_e @ weather_t + self.b_e)

                # Reset gate (with h_o and weather gating)
                r = torch.sigmoid(self.w_rx @ x + self.w_rh @ h_o + 
                                 self.w_re @ e_t + self.b_r)

                # Update gate (with h_o and weather gating)
                z = torch.sigmoid(self.w_zx @ x + self.w_zh @ h_o + 
                                 self.w_ze @ e_t + self.b_z)

                # Candidate hidden state (with reset-gated h_o)
                h_tilde = torch.tanh(self.w_hx @ x + self.w_hh @ (r * h_o) + self.b_h)

                # New hidden state
                h_next = (1 - z) * h_o + z * h_tilde

                h_output[b, t] = h_next.t().squeeze(0)
                h_t = h_next

        return h_output
