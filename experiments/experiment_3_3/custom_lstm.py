import torch
import torch.nn as nn
import math


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended_output = torch.sum(attention_weights * hidden_states, dim=1)
        return attended_output, attention_weights


class CustomLSTMCell(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.dropout = nn.Dropout(dropout)

        # Layer normalization components
        self.layer_norm_1 = nn.LayerNorm(hidden_sz)
        self.layer_norm_2 = nn.LayerNorm(hidden_sz)

        # Gate parameters with Xavier/Glorot initialization
        self.gates = nn.Linear(input_sz + hidden_sz, 4 * hidden_sz)
        nn.init.xavier_uniform_(self.gates.weight)

        # Residual connection parameters
        self.residual_proj = nn.Linear(input_sz, hidden_sz) if input_sz != hidden_sz else None

    def forward(self, x_t, h_t, c_t):
        # Combined input and hidden state
        combined = torch.cat((x_t, h_t), dim=1)

        # Apply dropout to the input
        combined = self.dropout(combined)

        # Calculate gates with layer normalization
        gates = self.gates(combined)
        i_t, f_t, g_t, o_t = gates.chunk(4, dim=1)

        # Apply activations
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)

        # Cell state update with layer normalization
        c_t = f_t * c_t + i_t * g_t
        c_t = self.layer_norm_1(c_t)

        # Hidden state update
        h_t = o_t * torch.tanh(c_t)
        h_t = self.layer_norm_2(h_t)

        # Residual connection
        if self.residual_proj is not None:
            h_t = h_t + self.residual_proj(x_t)

        return h_t, c_t


class EnhancedLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.num_layers = num_layers

        # Multiple LSTM layers
        self.lstm_cells = nn.ModuleList([
            CustomLSTMCell(
                input_sz if i == 0 else hidden_sz,
                hidden_sz,
                dropout
            ) for i in range(num_layers)
        ])

        # Attention mechanism
        self.attention = AttentionLayer(hidden_sz)

        # Skip connections
        self.skip_connections = nn.ModuleList([
            nn.Linear(input_sz, hidden_sz) for _ in range(num_layers)
        ])

    def forward(self, x):
        bs, seq_sz, _ = x.shape
        hidden_states = []

        # Initialize states for all layers
        h_t = [torch.zeros(bs, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        c_t = [torch.zeros(bs, self.hidden_size).to(x.device) for _ in range(self.num_layers)]

        for t in range(seq_sz):
            x_t = x[:, t, :]
            layer_input = x_t

            for layer in range(self.num_layers):
                # Process through LSTM cell
                h_t[layer], c_t[layer] = self.lstm_cells[layer](
                    layer_input, h_t[layer], c_t[layer]
                )

                # Skip connection
                skip = self.skip_connections[layer](x_t)
                h_t[layer] = h_t[layer] + skip

                # Prepare input for next layer
                layer_input = h_t[layer]

            hidden_states.append(h_t[-1].unsqueeze(1))

        # Combine all hidden states
        hidden_seq = torch.cat(hidden_states, dim=1)

        # Apply attention
        attended_output, attention_weights = self.attention(hidden_seq)

        return attended_output, hidden_seq, attention_weights


class AdvancedPricePredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = EnhancedLSTM(input_size, hidden_size, num_layers, dropout)

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        attended_output, hidden_seq, attention_weights = self.lstm(x)
        predictions = self.prediction_head(attended_output)
        return predictions, attention_weights
