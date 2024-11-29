import torch
import torch.nn as nn
import torch.nn.functional as F
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size = x.size(0)

        Q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, V).transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.num_heads * self.head_dim)
        return self.out(out)


class GatedLinearUnit(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size * 2)

    def forward(self, x):
        x = self.linear(x)
        x, gates = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gates)


class TemporalConvNet(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = x.transpose(1, 2)
        return self.layer_norm(x)


class EnhancedLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.num_layers = num_layers

        # Input embedding layer
        self.input_embedding = nn.Linear(input_sz, hidden_sz)

        # Original LSTM components
        self.lstm_cells = nn.ModuleList([
            CustomLSTMCell(
                hidden_sz,  # Now all layers use hidden_sz
                hidden_sz,
                dropout
            ) for i in range(num_layers)
        ])

        # Original attention mechanism
        self.attention = AttentionLayer(hidden_sz)

        # Additional advanced components
        self.positional_encoding = PositionalEncoding(hidden_sz)
        self.self_attention = MultiHeadSelfAttention(hidden_sz)
        self.glu = GatedLinearUnit(hidden_sz, hidden_sz)
        self.temporal_conv = TemporalConvNet(hidden_sz, hidden_sz)

        # Layer normalization and skip connections
        self.layer_norm = nn.LayerNorm(hidden_sz)
        # Modified skip connections to use hidden_sz
        self.skip_connections = nn.ModuleList([
            nn.Linear(hidden_sz, hidden_sz) for _ in range(num_layers)
        ])

    def forward(self, x):
        bs, seq_sz, _ = x.shape

        # Project input to hidden size
        x = self.input_embedding(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        hidden_states = []

        # Initialize states
        h_t = [torch.zeros(bs, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        c_t = [torch.zeros(bs, self.hidden_size).to(x.device) for _ in range(self.num_layers)]

        for t in range(seq_sz):
            x_t = x[:, t, :]  # Now x_t is already in hidden_size dimension
            layer_input = x_t

            for layer in range(self.num_layers):
                # LSTM processing
                h_t[layer], c_t[layer] = self.lstm_cells[layer](
                    layer_input, h_t[layer], c_t[layer]
                )

                # Enhanced processing
                h_t[layer] = self.glu(h_t[layer])

                # Skip connection
                skip = self.skip_connections[layer](layer_input)
                h_t[layer] = h_t[layer] + skip

                layer_input = h_t[layer]

            hidden_states.append(h_t[-1].unsqueeze(1))

        # Combine hidden states
        hidden_seq = torch.cat(hidden_states, dim=1)

        # Apply additional processing
        hidden_seq = self.self_attention(hidden_seq) + hidden_seq
        hidden_seq = self.temporal_conv(hidden_seq) + hidden_seq
        hidden_seq = self.layer_norm(hidden_seq)

        # Apply original attention mechanism
        attended_output, attention_weights = self.attention(hidden_seq)

        return attended_output, hidden_seq, attention_weights


class AdvancedPricePredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = EnhancedLSTM(input_size, hidden_size, num_layers, dropout)

        # Enhanced prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        attended_output, hidden_seq, attention_weights = self.lstm(x)
        predictions = self.prediction_head(attended_output)
        return predictions, attention_weights
