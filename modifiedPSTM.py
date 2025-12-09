import math
import torch
import torch.nn as nn
from torch.nn import Parameter, init
from torch import Tensor
import xlrd
import numpy as np
from math import sqrt
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import math
import random as rd
import calendar
from torch.autograd import Variable
from sklearn.preprocessing import minmax_scale 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import csv

class pew_LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        weather_size: int,
        use_periodic: bool = True,
        use_weather: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weather_size = weather_size
        self.use_periodic = use_periodic
        self.use_weather = use_weather

        self.w_ix = Parameter(Tensor(hidden_size, input_size))
        self.w_ih = Parameter(Tensor(hidden_size, hidden_size))
        self.w_ie = Parameter(Tensor(hidden_size, hidden_size))
        self.b_i = Parameter(Tensor(hidden_size, 1))

        self.w_fx = Parameter(Tensor(hidden_size, input_size))
        self.w_fo = Parameter(Tensor(hidden_size, hidden_size))
        self.w_fe = Parameter(Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(Tensor(hidden_size, 1))

        self.w_ox = Parameter(Tensor(hidden_size, input_size))
        self.w_oh = Parameter(Tensor(hidden_size, hidden_size))
        self.w_oe = Parameter(Tensor(hidden_size, hidden_size))
        self.b_o = Parameter(Tensor(hidden_size, 1))

        self.w_gx = Parameter(Tensor(hidden_size, input_size))
        self.w_gh = Parameter(Tensor(hidden_size, hidden_size))
        self.b_g = Parameter(Tensor(hidden_size, 1))

        self.w_d = Parameter(Tensor(hidden_size, input_size))
        self.w_w = Parameter(Tensor(hidden_size, input_size))
        self.w_m = Parameter(Tensor(hidden_size, input_size))
        self.w_t = Parameter(Tensor(hidden_size, hidden_size))
        self.w_e = Parameter(Tensor(hidden_size, weather_size))
        self.b_e = Parameter(Tensor(hidden_size, 1))

        self.reset_weights()

    def reset_weights(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, x_input: torch.Tensor, x_weather: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_size, _ = x_input.size()
        device = x_input.device

        h_output = torch.zeros(batch_size, seq_size, self.hidden_size, device=device)
        c_output = torch.zeros(batch_size, seq_size, self.hidden_size, device=device)

        for b in range(batch_size):
            h_t = torch.zeros(self.hidden_size, 1, device=device)
            c_t = torch.zeros(self.hidden_size, 1, device=device)

            for t in range(seq_size):
                if self.use_periodic and b >= 1:
                    h_d = x_input[b - 1, t, :].unsqueeze(1)
                else:
                    h_d = torch.zeros(self.input_size, 1, device=device)

                if self.use_periodic and b >= 7:
                    h_w = x_input[b - 7, t, :].unsqueeze(1)
                else:
                    h_w = torch.zeros(self.input_size, 1, device=device)

                if self.use_periodic and b >= 30:
                    h_m = x_input[b - 30, t, :].unsqueeze(1)
                else:
                    h_m = torch.zeros(self.input_size, 1, device=device)

                x = x_input[b, t, :].unsqueeze(1)
                if self.use_weather:
                    weather_t = x_weather[b, t, :].unsqueeze(1)
                else:
                    weather_t = torch.zeros(self.weather_size, 1, device=device)

                h_o = torch.sigmoid(
                    self.w_d @ h_d + self.w_w @ h_w + self.w_m @ h_m + self.w_t @ h_t + self.b_e
                )
                e_t = torch.sigmoid(self.w_e @ weather_t + self.b_e)

                i = torch.sigmoid(self.w_ix @ x + self.w_ih @ h_o + self.w_ie @ e_t + self.b_i)
                g = torch.tanh(self.w_gx @ x + self.w_gh @ h_o + self.b_g)
                f = torch.sigmoid(self.w_fx @ x + self.w_fo @ h_o + self.w_fe @ e_t + self.b_f)
                o = torch.sigmoid(self.w_ox @ x + self.w_oh @ h_o + self.w_oe @ e_t + self.b_o)

                c_next = f * c_t + i * g
                h_next = o * torch.tanh(c_next)

                h_output[b, t] = h_next.squeeze(1)
                c_output[b, t] = c_next.squeeze(1)

                h_t = h_next
                c_t = c_next

        return h_output, c_output
