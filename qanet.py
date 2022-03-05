import layers
import torch
import torch.nn as nn
import math
from util import masked_softmax


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)


class PositionalEncoding(nn.Module):
    """
    Modified from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, embed_size, drop_prob=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=drop_prob)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        pe = torch.zeros(1, embed_size, max_len)
        pe[0, 0::2, :] = torch.sin(position * div_term).transpose(0, 1)
        pe[0, 1::2, :] = torch.cos(position * div_term).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # (batch_size, embed_size, seq_len)
        x = x + self.pe[:, :, :x.size(2)]
        return self.dropout(x)


class DepthwiseSeparableConv(nn.Module):
    """
    Implemented based on the description in
    https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim=1, bias=True):
        super(DepthwiseSeparableConv, self).__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                            kernel_size=kernel_size, groups=in_channels,
                                            padding=kernel_size // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=1, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                            kernel_size=kernel_size, groups=in_channels,
                                            padding=kernel_size // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=1, bias=bias)
        else:
            raise ValueError(f'dimension of {dim} invalid')

        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class QANetEmbedding(nn.Module):
    def __init__(self, word_vectors, char_vectors, drop_prob):
        super(QANetEmbedding, self).__init__()

        word_emb_size = word_vectors.shape[-1]
        char_emb_size = char_vectors.shape[-1]

        self.embed_size = word_emb_size + char_emb_size
        self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_vectors))
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_vectors))

        self.conv = DepthwiseSeparableConv(char_emb_size, char_emb_size, 5, dim=2)
        self.highway = layers.HighwayEncoder(2, word_emb_size + char_emb_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, w_idxs, c_idxs):
        w_emb = self.word_emb(w_idxs)  # (batch_size, seq_len, word_emb_size)
        c_emb = self.char_emb(c_idxs)  # (batch_size, seq_len, max char per word, char_emb_size)

        c_emb = c_emb.permute(0, 3, 1, 2)   # (batch_size, char_emb_size, seq_len, max char per word)
        c_emb = self.conv(c_emb)    # (batch_size, char_emb_size, seq_len, max char per word)
        c_emb = torch.relu(c_emb)
        c_emb, _ = torch.max(c_emb, dim=-1) # (batch_size, char_emb_size, seq_len)
        w_emb = self.dropout(w_emb)
        c_emb = self.dropout(c_emb)
        c_emb = c_emb.transpose(1, 2)   # (batch_size, seq_len, char_emb_size)
        emb = torch.cat([w_emb, c_emb], dim=2)  # (batch_size, seq_len, emb_size)
        emb = self.highway(emb)
        return emb  # (batch_size, seq_len, emb_size)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_size, n_head, drop_prob):
        super(MultiHeadSelfAttention, self).__init__()
        dim_per_head = input_size // n_head
        self.query = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(input_size, input_size)
        self.scale = 1 / math.sqrt(dim_per_head)
        self.input_size = input_size
        self.dim_per_head = dim_per_head
        self.n_head = n_head

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.size()   # (batch_size, seq_len, emb_size)
        k = self.key(x).view(batch_size, seq_len, self.n_head, self.dim_per_head)
        q = self.query(x).view(batch_size, seq_len, self.n_head, self.dim_per_head)
        v = self.value(x).view(batch_size, seq_len, self.n_head, self.dim_per_head)
        q = q.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_head, seq_len, self.dim_per_head)
        k = k.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_head, seq_len, self.dim_per_head)
        v = v.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_head, seq_len, self.dim_per_head)
        mask = mask.unsqueeze(1).expand(-1, seq_len, -1).repeat(self.n_head, 1, 1)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale     # (batch_size * n_head, seq_len, seq_len)
        attn = mask_logits(attn, mask)
        attn = torch.softmax(attn, dim=2)   # (batch_size * n_head, seq_len, seq_len)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v)    # (batch_size * n_head, seq_len, dim_per_head)
        out = out.view(self.n_head, batch_size, seq_len, self.dim_per_head).permute(1, 2, 0, 3).contiguous()
        out = out.view(batch_size, seq_len, self.input_size)
        out = self.fc(out)  # (batch_size, seq_len, input_size)
        out = self.dropout(out)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, num_conv, num_channel, kernel_size, input_size, n_head, drop_prob):
        super(EncoderBlock, self).__init__()

        self.convs = nn.ModuleList([DepthwiseSeparableConv(num_channel, num_channel, kernel_size) for _ in range(num_conv)])
        self.self_att = MultiHeadSelfAttention(input_size, n_head, drop_prob)
        self.fc = nn.Linear(num_channel, num_channel)
        self.pos = PositionalEncoding(input_size)
        self.layer_norm_conv = nn.ModuleList([nn.LayerNorm(input_size) for _ in range(num_conv)])
        self.layer_norm_att = nn.LayerNorm(input_size)
        self.layer_norm_fc = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, mask):
        # x is (batch_size, num_channel, seq_len, input_size)
        out = self.pos(x).transpose(1, 2)
        res = out

        for i, conv in enumerate(self.convs):
            out = self.layer_norm_conv[i](out)
            out = out.transpose(1, 2)
            out = conv(out)
            out = torch.relu(out)
            out = out.transpose(1, 2)
            out = out + res
            if i % 2 == 0:
                out = self.dropout(out)
            res = out

        out = self.layer_norm_att(out)
        out = self.self_att(out, mask)
        out = out + res
        out = self.dropout(out)

        res = out
        out = self.layer_norm_fc(out)
        out = self.fc(out)
        out = torch.relu(out)
        out = out + res
        out = self.dropout(out)
        return out


class EncoderConvBlock(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, emb_size):
        super(EncoderConvBlock, self).__init__()
        self.conv = DepthwiseSeparableConv(kernel_size, in_channels, out_channels)
        self.layer_norm = nn.LayerNorm(emb_size)

    def forward(self, x):
        return x + self.conv(self.layer_norm(x))


class QANetOutput(nn.Module):
    def __init__(self, input_size):
        super(QANetOutput, self).__init__()
        self.W1 = nn.Linear(input_size * 2, 1)
        self.W2 = nn.Linear(input_size * 2, 1)

    def forward(self, m0, m1, m2, mask):
        logits_1 = self.W1(torch.cat([m0, m1], 2))
        logits_2 = self.W2(torch.cat([m0, m2], 2))

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


class QANet(nn.Module):
    def __init__(self, word_vectors, char_vectors, n_heads, encoder_size, drop_prob):
        super(QANet, self).__init__()
        self.emb = QANetEmbedding(word_vectors, char_vectors, drop_prob)
        embed_size = self.emb.embed_size

        self.c_conv = DepthwiseSeparableConv(embed_size, encoder_size, 5)
        self.q_conv = DepthwiseSeparableConv(embed_size, encoder_size, 5)

        self.emb_enc = EncoderBlock(num_conv=4, num_channel=encoder_size, kernel_size=7,
                                    input_size=encoder_size, n_head=n_heads, drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(encoder_size, drop_prob)
        self.att_squeeze = DepthwiseSeparableConv(encoder_size * 4, encoder_size, 5)

        self.modeling = nn.ModuleList([EncoderBlock(num_conv=2, num_channel=encoder_size,
                                                    kernel_size=7, input_size=encoder_size,
                                                    n_head=n_heads, drop_prob=drop_prob) for _ in range(5)])

        self.out = QANetOutput(encoder_size)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs, *args):
        c_mask = (torch.zeros_like(cw_idxs) != cw_idxs).float()     # (batch_size, p_len)
        q_mask = (torch.zeros_like(qw_idxs) != qw_idxs).float()     # (batch_size, q_len)

        c_emb = self.emb(cw_idxs, cc_idxs).transpose(1, 2)     # (batch_size, emb_size, p_len)
        q_emb = self.emb(qw_idxs, qc_idxs).transpose(1, 2)     # (batch_size, emb_size, q_len)

        c_conv = self.c_conv(c_emb)     # (batch_size, encoder_size, p_len)
        q_conv = self.q_conv(q_emb)     # (batch_size, encoder_size, q_len)

        c_enc = self.emb_enc(c_conv, c_mask)  # (batch_size, p_len, encoder_size)
        q_enc = self.emb_enc(q_conv, q_mask)  # (batch_size, q_len, encoder_size)

        x = self.att(c_enc, q_enc, c_mask, q_mask)  # (batch_size, p_len, 4 * encoder_size)
        x = x.transpose(1, 2)
        m0 = self.att_squeeze(x)    # (batch_size, encoder_size, p_len)

        for enc in self.modeling:
            m0 = enc(m0, c_mask).transpose(1, 2)
        m1 = m0
        for enc in self.modeling:
            m1 = enc(m1, c_mask).transpose(1, 2)
        m2 = m1
        for enc in self.modeling:
            m2 = enc(m2, c_mask).transpose(1, 2)

        m0, m1, m2 = [m.transpose(1, 2) for m in (m0, m1, m2)]
        return self.out(m0, m1, m2, c_mask)
