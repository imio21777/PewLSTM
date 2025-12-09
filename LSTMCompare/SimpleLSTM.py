import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
import math
from torch.nn import init

class SimpleLSTM(nn.Module):
    """
    Standard LSTM for parking prediction (History only)
    """
    def __init__(self, input_size, hidden_size):
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Standard LSTM parameters
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
    def forward(self, x, h_c=None):
        """
        Args:
            x: [batch_size, seq_len, input_size]
            h_c: (h_0, c_0)
        Returns:
            h_n, c_n (last step hidden state and cell state)
        """
        # We only need the output for the sequence to match PewLSTM's interface which returns (h, c) sequence
        # But looking at overall.py: 
        # h1, c1 = self.lstm1(x_input)
        # h2, c2 = self.lstm2(h1)
        # It expects the full sequence of hidden states.
        
        output, (h_n, c_n) = self.lstm(x, h_c)
        
        # PewLSTM returns (h_output, c_output) where h_output is [batch, seq, hidden]
        # nn.LSTM returns output as [batch, seq, hidden]
        # We need to return c_output as well to be compatible with the second layer call in overall.py
        # However, nn.LSTM doesn't return the full sequence of cell states easily unless we use a custom loop or just return the last one?
        # Wait, overall.py does: h2, c2 = self.lstm2(h1)
        # So the first layer must return a sequence of hidden states 'h1' to be input to the second layer.
        # The second layer's 'h2' is then viewed: out = h2.contiguous().view(-1, HIDDEN_DIM)
        # So we definitely need the full sequence of hidden states.
        # We don't strictly need the full sequence of cell states for the next layer, but the interface might expect it.
        # Let's check PewLSTM.py again. It returns (h_output, c_output).
        # For SimpleLSTM, we can just return (output, None) if c_output isn't used by the next layer.
        # In overall.py:
        # h1, c1 = self.lstm1(x_input)
        # h2, c2 = self.lstm2(h1)
        # out = h2...
        # c1 and c2 are unused. So we can return None for the second element.
        
        return output, None
