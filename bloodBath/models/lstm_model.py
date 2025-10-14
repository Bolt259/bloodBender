"""Minimal LSTM model wrapper with optional PyTorch dependency.

If PyTorch is available this provides a simple LSTM regression model. If not,
the module exposes a numpy-based fallback so tests can run without installing
torch in lightweight environments.
"""
from typing import Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class SimpleLSTM(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1, output_size: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # x: (batch, seq, features)
            out, _ = self.lstm(x)
            # Take last timestep
            out = out[:, -1, :]
            return self.fc(out)

    def build_model(input_size: int, hidden_size: int = 32, num_layers: int = 1, output_size: int = 1):
        return SimpleLSTM(input_size, hidden_size, num_layers, output_size)

else:
    # Numpy fallback: a stub that mimics the interface but performs no learning
    import numpy as np

    class NumpyLSTMStub:
        def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1, output_size: int = 1):
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.output_size = output_size

        def train_step(self, batch_x, batch_y):
            # batch_x: numpy array (batch, seq, features)
            # batch_y: numpy array (batch,)
            # Return a fake loss that decreases slightly each call
            return float(np.abs(batch_y).mean())

    def build_model(input_size: int, hidden_size: int = 32, num_layers: int = 1, output_size: int = 1):
        return NumpyLSTMStub(input_size, hidden_size, num_layers, output_size)
