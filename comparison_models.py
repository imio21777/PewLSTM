import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init
from torch import Tensor
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- Simple LSTM ---
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(SimpleLSTM, self).__init__()
        # input_size should be 1 for "only parking history"
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, x):
        # x: [batch, seq, input_size]
        # For SimpleLSTM, we might need to slice x if it contains weather data
        if x.size(2) > 1:
            x = x[:, :, -1].unsqueeze(2) # Take only the last column (parking count)
            
        out, _ = self.lstm(x)
        # Flatten output for regression: (batch * seq, hidden)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out).view(-1) # (batch * seq)
        return out

# --- RandomForest ---
class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

    def fit(self, x, y):
        # x: [batch, seq, features] -> flatten to [batch * seq, features] ? 
        # Wait, PewLSTM predicts sequence. 
        # In main.py: 
        # train_x: [days, 24, 5]
        # train_y: [days * 24]
        # The LSTM outputs (days * 24) predictions.
        # For RF, we probably want to predict each hour based on its context.
        # But PewLSTM uses a sequence-to-sequence approach (many-to-many).
        # If we treat each hour as a sample:
        # Input for hour t: features at t? Or window?
        # The LSTM sees the whole sequence.
        # To be simple and comparable, let's reshape inputs to (N, features)
        # where N = total hours.
        # But LSTM has state. RF doesn't.
        # If we just use current hour features, RF is weak.
        # If we use window, we need to restructure data.
        # However, the prompt asks to use the provided data split.
        # train_x is (days, 24, 5).
        # We can flatten this to (days * 24, 5) and use the 5 features of the current hour to predict.
        # Or we can use the whole day sequence to predict the whole day sequence?
        # RF usually predicts one value.
        # Let's assume we predict each hour independently based on its features (and maybe previous hour if we had it, but the data structure is (days, 24, 5)).
        # Actually, the LSTM uses h_d, h_w, h_m which are derived from indices.
        # RF won't have access to h_d, h_w, h_m unless we engineer them.
        # Given the constraints, I will flatten the input to (Total_Hours, 5) and predict (Total_Hours).
        # This is a "Regression Method" baseline.
        
        # Reshape x: (batch, seq, feat) -> (batch * seq, feat)
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = y.reshape(-1)
        self.model.fit(x_flat, y_flat)

    def predict(self, x):
        x_flat = x.reshape(-1, x.shape[-1])
        return self.model.predict(x_flat)

# --- Ablation PewLSTM ---
# Copied and modified from modifiedPSTM.py
class AblationPewLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, weather_size: int, 
                 use_periodic=True, use_weather=True):
        super(AblationPewLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weather_size = weather_size
        self.use_periodic = use_periodic
        self.use_weather = use_weather

        # input gate
        self.w_ix = Parameter(Tensor(hidden_size, input_size))
        self.w_ih = Parameter(Tensor(hidden_size, hidden_size))
        self.w_ie = Parameter(Tensor(hidden_size, hidden_size)) 
        self.b_i = Parameter(Tensor(hidden_size, 1))

        # forget gate
        self.w_fx = Parameter(Tensor(hidden_size, input_size))
        self.w_fo = Parameter(Tensor(hidden_size, hidden_size))
        self.w_fe = Parameter(Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(Tensor(hidden_size, 1))

        # output gate
        self.w_ox = Parameter(Tensor(hidden_size, input_size))
        self.w_oh = Parameter(Tensor(hidden_size, hidden_size))
        self.w_oe = Parameter(Tensor(hidden_size, hidden_size))
        self.b_o = Parameter(Tensor(hidden_size, 1))
        
        # cell
        self.w_gx = Parameter(Tensor(hidden_size, input_size))
        self.w_gh = Parameter(Tensor(hidden_size, hidden_size))
        self.b_g = Parameter(Tensor(hidden_size, 1))

        # ho
        self.w_d = Parameter(Tensor(hidden_size, input_size))
        self.w_w = Parameter(Tensor(hidden_size, input_size))
        self.w_m = Parameter(Tensor(hidden_size, input_size))
        self.w_t = Parameter(Tensor(hidden_size, hidden_size))
        self.w_e = Parameter(Tensor(hidden_size, weather_size))
        self.b_e = Parameter(Tensor(hidden_size, 1)) 

        self.reset_weights()

    def reset_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, x_input, x_weather):
        batch_size, seq_size, input_dim = x_input.size()

        h_output = torch.zeros(batch_size, seq_size, self.hidden_size).to(x_input.device)
        c_output = torch.zeros(batch_size, seq_size, self.hidden_size).to(x_input.device)

        for b in range(batch_size):
            h_t = torch.zeros(1, self.hidden_size).t().to(x_input.device)
            c_t = torch.zeros(1, self.hidden_size).t().to(x_input.device)

            for t in range(24):
                # Periodic components
                if self.use_periodic:
                    # day
                    if b < 1:
                        h_d = torch.zeros(1, self.input_size).t().to(x_input.device)
                    else:
                        h_d = x_input[b-1, t, :].unsqueeze(0).t()
                    
                    # week
                    if b < 7:
                        h_w = torch.zeros(1, self.input_size).t().to(x_input.device)
                    else:
                        h_w = x_input[b-7, t, :].unsqueeze(0).t()

                    # month
                    if b < 30:
                        h_m = torch.zeros(1, self.input_size).t().to(x_input.device)
                    else:
                        h_m = x_input[b-30, t, :].unsqueeze(0).t()
                else:
                    # If not using periodic, these terms should not contribute
                    # We can set them to zero, but we also need to ensure h_o calculation is correct
                    # In the formula: h_o = sigmoid(w_d@h_d + ... + w_t@h_t)
                    # If we set h_d, h_w, h_m to zero, the weights w_d, w_w, w_m still exist but result is 0.
                    h_d = torch.zeros(1, self.input_size).t().to(x_input.device)
                    h_w = torch.zeros(1, self.input_size).t().to(x_input.device)
                    h_m = torch.zeros(1, self.input_size).t().to(x_input.device)

                x = x_input[b, t, :].unsqueeze(0).t()  # [input_dim, 1]
                weather_t = x_weather[b, t, :].unsqueeze(0).t()  # [weather_dim, 1]
                
                # Calculate h_o
                # Note: modifiedPSTM.py uses:
                # h_o = torch.sigmoid(self.w_d @ h_d + self.w_w @ h_w + self.w_t @ h_t +
                #                     self.w_m @ h_m + self.w_t @ h_t)
                # Wait, self.w_t @ h_t is added TWICE in the original code?
                # "self.w_t @ h_t + ... + self.w_t @ h_t"
                # Let's check the file content I read earlier.
                # Line 109: self.w_d @ h_d + self.w_w @ h_w + self.w_t @ h_t + self.w_m @ h_m + self.w_t @ h_t
                # Yes, it seems to be added twice or it's a typo in original code. I will preserve it.
                
                if self.use_periodic:
                     h_o = torch.sigmoid(self.w_d @ h_d + self.w_w @ h_w + self.w_t @ h_t +
                                    self.w_m @ h_m + self.w_t @ h_t)
                else:
                    # If w/o periodic, we should probably just use h_t or similar?
                    # The paper says "w/o Periodic".
                    # If we just zero out h_d, h_w, h_m, h_o becomes sigmoid(2 * w_t @ h_t).
                    # This seems like a fair ablation if we want to keep the structure.
                    h_o = torch.sigmoid(self.w_d @ h_d + self.w_w @ h_w + self.w_t @ h_t +
                                    self.w_m @ h_m + self.w_t @ h_t)

                # Weather component
                if self.use_weather:
                    e_t = torch.sigmoid(self.w_e @ weather_t + self.b_e)
                else:
                    # If w/o weather, e_t should be zero?
                    # Or we can just not add it to the gates.
                    # If e_t is 0, w_ie @ e_t is 0.
                    e_t = torch.zeros(self.hidden_size, 1).to(x_input.device)

                # input gate
                i = torch.sigmoid(self.w_ix @ x + self.w_ih @ h_o +
                                    self.w_ie @ e_t + self.b_i)
                # cell
                g = torch.tanh(self.w_gx @ x + self.w_gh @ h_o
                                + self.b_g)
                # forget gate
                f = torch.sigmoid(self.w_fx @ x + self.w_fo @ h_o +
                                self.w_fe @ e_t + self.b_f)
                
                # output gate
                # modifiedPSTM.py uses h_o in output gate: self.w_oh @ h_o (Wait, let me check diff)
                # Diff said: < o = ... self.w_oh @ h_t ... --- > o = ... self.w_oh @ h_o ...
                # So modifiedPSTM uses h_o.
                o = torch.sigmoid(self.w_ox @ x + self.w_oh @ h_o +
                                    self.w_oe @ e_t + self.b_o)

                c_next = f * c_t + i * g
                h_next = o * torch.tanh(c_next)

                h_output[b, t] = h_next.t().squeeze(0)
                c_output[b, t] = c_next.t().squeeze(0)

                h_t = h_next
                c_t = c_next

        return (h_output, c_output)

class AblationPewLSTM(nn.Module):
    def __init__(self, hidden_dim=1, use_periodic=True, use_weather=True):
        super(AblationPewLSTM, self).__init__()
        self.lstm1 = AblationPewLSTMCell(1, hidden_dim, 4, use_periodic, use_weather)
        self.lstm2 = AblationPewLSTMCell(hidden_dim, hidden_dim, 4, use_periodic, use_weather)
        self.fc = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc.weight)
        self.hidden_dim = hidden_dim

    def forward(self, input):  # [batch_size, seq_size, weather_size + input_dim]
        x_weather = input[:, :, :-1]  # [batch_size, seq_size, weather_size]
        x_input = input[:, :, -1].unsqueeze(2)  # [batch_size, seq_size, input_dim]

        h1, c1 = self.lstm1(x_input, x_weather)
        h2, c2 = self.lstm2(h1, x_weather)
        out = h2.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out).view(-1)
        return out
