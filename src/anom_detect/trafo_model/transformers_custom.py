"""A from-scratch Transformer encoder-decoder used by ``TransformerAE``."""
import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        """Add positional embeddings to (batch, seq_len, d_model)."""
        return x + self.pos_embedding[:, : x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dimension, num_attention_heads, dropout):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert hidden_dimension % num_attention_heads == 0
        self.positional_encoding = LearnablePositionalEncoding(d_model=512, max_len=1024)
        self.hidden_dimension = hidden_dimension
        self.num_attention_heads = num_attention_heads
        self.head_dimension = hidden_dimension // num_attention_heads

        self.W_q = nn.Linear(hidden_dimension, hidden_dimension)
        self.W_k = nn.Linear(hidden_dimension, hidden_dimension)
        self.W_v = nn.Linear(hidden_dimension, hidden_dimension)
        self.W_o = nn.Linear(hidden_dimension, hidden_dimension)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dimension])).to(device)

    def split_heads(self, item, batch_size):
        """(B, len, hidden) -> (B, heads, len, head_dim)."""
        item = item.view(batch_size, -1, self.num_attention_heads, self.head_dimension)
        return item.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        Q = Q + self.positional_encoding(Q)
        K = K + self.positional_encoding(K)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Q x K.T / sqrt(d_k) -> (B, heads, qlen, klen)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e15)
        attention = torch.softmax(energy, dim=-1)

        attention_scored_value = torch.matmul(self.dropout(attention), V)
        attention_scored_value = attention_scored_value.permute(0, 2, 1, 3).contiguous()
        attention_scored_value = attention_scored_value.view(batch_size, -1, self.hidden_dimension)

        return self.W_o(attention_scored_value), attention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output, _ = self.cross_attn(x, enc_output, enc_output)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm3(x + self.dropout(ff_output))


class Transformer(nn.Module):
    def __init__(self, d_model=256, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """``src``/``tgt`` are (batch, seq_len, d_model)."""
        enc_output = src
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        dec_output = tgt
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, tgt_mask, src_mask)

        return self.fc_out(dec_output)
