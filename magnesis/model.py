import torch.nn as nn


class GeomagneticModel(nn.Module):
    def __init__(
        self,
        n_features,
        lstm_hidden_size,
        lstm_num_layers,
        lstm_dropout,
        dst_attention_heads,
        ae_attention_heads,
    ):
        super().__init__()

        self.dst_lstm = nn.LSTM(
            n_features,
            lstm_hidden_size,
            lstm_num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.ae_lstm = nn.LSTM(
            n_features,
            lstm_hidden_size,
            lstm_num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )

        self.dst_attention = nn.MultiheadAttention(
            lstm_hidden_size,
            dst_attention_heads,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.ae_attention = nn.MultiheadAttention(
            lstm_hidden_size, ae_attention_heads, dropout=lstm_dropout, batch_first=True
        )

        self.dst_attention_norm = nn.LayerNorm(lstm_hidden_size)
        self.ae_attention_norm = nn.LayerNorm(lstm_hidden_size)

        self.dropout = nn.Dropout(lstm_dropout)
        heads_hidden_size = lstm_hidden_size

        self.dst_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, heads_hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(lstm_dropout),
            nn.Linear(heads_hidden_size, 1),
        )

        self.ae_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, heads_hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(lstm_dropout),
            nn.Linear(heads_hidden_size, 1),
        )

    def forward(self, x):
        dst_out, _ = self.dst_lstm(x)
        dst_query = dst_out[:, -1:, :]
        dst_attn_out, dst_attn_w = self.dst_attention(dst_query, dst_out, dst_out)
        dst_features = self.dst_attention_norm(
            dst_query.squeeze(1) + dst_attn_out.squeeze(1)
        )
        dst_features = self.dropout(dst_features)
        dst = self.dst_head(dst_features)

        ae_out, _ = self.ae_lstm(x)
        ae_query = ae_out[:, -1:, :]
        ae_attn_out, ae_attn_w = self.ae_attention(ae_query, ae_out, ae_out)
        ae_features = self.ae_attention_norm(
            ae_query.squeeze(1) + ae_attn_out.squeeze(1)
        )
        ae_features = self.dropout(ae_features)
        ae = self.ae_head(ae_features)

        return dst, ae, (dst_attn_w, ae_attn_w)
