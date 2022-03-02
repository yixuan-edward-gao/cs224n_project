import layers
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Modified from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, embed_size, drop_prob=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=drop_prob)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        pe = torch.zeros(1, max_len, embed_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # (batch_size, seq_len, embed_size)
        x = x + self.pe[:, :x.size(0), :]
        return self.dropout(x)


class DepthwiseSeparableConv(nn.Module):
    """
    Implemented based on the description in
    https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
    """
    def __init__(self, kernel_size, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                        kernel_size=kernel_size, padding=kernel_size // 2, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # x is (batch_size, in_channels, seq_len, emb_size)
        x = self.depthwise_conv(x)  # (batch_size, in_channels, seq_len, emb_size)
        x = self.pointwise_conv(x)  # (batch_size, out_channels, seq_len, emb_size)
        return torch.relu(x)


class EncoderConvBlock(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, emb_size):
        super(EncoderConvBlock, self).__init__()
        self.conv = DepthwiseSeparableConv(kernel_size, in_channels, out_channels)
        self.layer_norm = nn.LayerNorm(emb_size)

    def forward(self, x):
        return x + self.conv(self.layer_norm(x))


class SelfAttention(nn.Module):
    def __init__(self, input_size, n_heads, drop_prob):
        super(SelfAttention, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.attn = nn.MultiheadAttention(input_size, n_heads, batch_first=True, dropout=drop_prob)

    def forward(self, x):
        res = x
        mask = (torch.eye(x.shape[1]) == 1).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        x = self.layer_norm(x)
        return res + self.attn(x, x, x, attn_mask=mask)


class EncoderBlock(nn.Module):
    def __init__(self, input_size, output_size, num_conv, kernel_size, n_heads, drop_prob):
        super(EncoderBlock, self).__init__()
        self.dropout = nn.Dropout(drop_prob)

        self.pos_embed = PositionalEncoding(input_size, drop_prob)
        self.init_proj = nn.Linear(input_size, output_size, bias=False)
        self.conv = nn.Sequential(*[EncoderConvBlock(kernel_size, output_size, output_size) for _ in range(num_conv)])
        self.conv_squeeze = nn.Conv2d(in_channels=output_size, out_channels=1, kernel_size=1)
        self.attn = SelfAttention(output_size, n_heads, drop_prob)
        self.ffn = nn.Sequential(nn.LayerNorm(output_size),
                                 nn.Linear(output_size, 4 * output_size),
                                 nn.GELU(),
                                 nn.Linear(4 * output_size, output_size),
                                 self.dropout)

        self.output_size = output_size

    def forward(self, x):
        # x is (batch_size, seq_len, input_size)
        x = self.pos_embed(x)   # (batch_size, seq_len, input_size)
        x = self.init_proj(x)   # (batch_size, seq_len, output_size)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, output_size)
        x = x.expand(-1, self.output_size, -1, -1)  # (batch_size, output_size, seq_len, output_size)
        x = self.conv(x)    # (batch_size, output_size, seq_len, output_size)
        x = self.conv_squeeze(x).squeeze()    # (batch_size, seq_len, output_size)
        x = self.dropout(x)
        x = self.attn(x)    # (batch_size, seq_len, output_size)
        return x + self.ffn(x)  # (batch_size, seq_len, output_size)


class QANet(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, n_heads, encoder_size, drop_prob):
        super(QANet, self).__init__()
        self.word_emb = layers.WordEmbedding(word_vectors=word_vectors,
                                             hidden_size=hidden_size,
                                             drop_prob=drop_prob)
        self.char_emb = layers.CharEmbedding(char_vectors=char_vectors,
                                             hidden_size=hidden_size,
                                             drop_prob=drop_prob,
                                             kernel_height=3)
        self.highway = layers.HighwayEncoder(num_layers=2, hidden_size=2 * hidden_size)
        self.highway_proj = nn.Linear(2 * hidden_size, hidden_size)
        self.encoder = EncoderBlock(hidden_size, encoder_size, 4, 7, n_heads, drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs, *args):
        cw_emb = self.word_emb(cw_idxs)  # (batch_size, c_len, hidden_size)
        qw_emb = self.word_emb(qw_idxs)  # (batch_size, q_len, hidden_size)

        cc_emb = self.char_emb(cc_idxs)  # (batch_size, c_len, hidden_size)
        qc_emb = self.char_emb(qc_idxs)  # (batch_size, q_len, hidden_size)

        c_emb_cat = torch.cat((cw_emb, cc_emb), 2)  # (batch_size, c_len, 2 * hidden_size)
        q_emb_cat = torch.cat((qw_emb, qc_emb), 2)  # (batch_size, q_len, 2 * hidden_size)

        c_emb = self.highway_proj(self.highway(c_emb_cat))  # (batch_size, c_len, hidden_size)
        q_emb = self.highway_proj(self.highway(q_emb_cat))  # (batch_size, q_len, hidden_size)

        c_enc = self.encoder(c_emb)     # (batch_size, c_len, encoder_size)
        q_enc = self.encoder(q_emb)     # (batch_size, c_len, encoder_size)
