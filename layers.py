"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


class WordEmbedding(nn.Module):
    """
    Word embedding layer used by BiDAF, without highway network.

    To be concatenated with character embedding before being fed to highway network.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(WordEmbedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)

        return emb


class CharEmbedding(nn.Module):
    """
    Character embedding layer used by BiDAF.

    Args:
        char_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
        kernel_height (int): height of kernel
    """
    def __init__(self, char_vectors, hidden_size, drop_prob, kernel_height):
        # hard coded constants...
        max_word_length = 16
        word_vec_size = 300
        char_vec_size = 64

        super(CharEmbedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(char_vectors)
        self.conv = nn.Conv2d(in_channels=1, out_channels=word_vec_size,
                              kernel_size=(kernel_height, char_vec_size), stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=(max_word_length - kernel_height + 1, 1), stride=1)
        self.proj = nn.Linear(word_vec_size, hidden_size, bias=False)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, max_word_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)

        batch_size, seq_len, word_len, embed_size = emb.size()

        emb = emb.view(-1, 1, word_len, embed_size)
        emb = self.conv(emb)    # (batch_size, word_vec_size, max_word_length - kernel_height + 1, 1)
        emb = F.relu(emb)
        emb = self.maxpool(emb) # (batch_size, word_vec_size, 1, 1)
        emb = emb.squeeze() # (batch_size, word_vec_size)
        emb = self.proj(emb)
        emb = emb.view(batch_size, seq_len, -1) # (batch_size, seq_len, hidden_size)

        return emb


class AnswerPointer(nn.Module):
    """
    Answer pointer layer to serve as output.
    See https://arxiv.org/abs/1608.07905 for details.
    """
    def __init__(self, hidden_size, input_size):
        super(AnswerPointer, self).__init__()
        self.V = nn.Linear(input_size, hidden_size, bias=False)
        self.W = nn.Linear(hidden_size, hidden_size)
        self.initial_hidden = nn.parameter.Parameter(torch.zeros(hidden_size))
        self.initial_cell = nn.parameter.Parameter(torch.zeros(hidden_size))
        self.proj = nn.Linear(hidden_size, 1)
        self.lstm = nn.LSTMCell(input_size, hidden_size)

    def forward(self, att, mod, mask):
        H = torch.cat([att, mod], 2)    # (batch_size, seq_len, input_size)
        batch_size = H.size()[0]

        F_start = torch.tanh(torch.add(self.V(H), self.W(self.initial_hidden))) # (batch_size, seq_len, hidden_size)
        beta_start = masked_softmax(self.proj(F_start).squeeze(), mask, log_softmax=True) # (batch_size, seq_len)

        # (batch_size, hidden_size)
        h_end, _ = self.lstm(torch.bmm(torch.transpose(H, 1, 2), torch.unsqueeze(beta_start, 2)).squeeze(),
                             (self.initial_hidden.expand((batch_size, -1)), self.initial_cell.expand((batch_size, -1))))
        F_end = torch.tanh(torch.add(self.V(H), self.W(h_end).unsqueeze(1)))    # (batch_size, seq_len, hidden_size)
        beta_end = masked_softmax(self.proj(F_end).squeeze(), mask, log_softmax=True)   # (batch_size, seq_len)

        return beta_start, beta_end


class MultiplicativeSelfAttention(nn.Module):
    """
    xxx
    """
    def __init__(self, input_size, drop_prob):
        super(MultiplicativeSelfAttention, self).__init__()
        self.W = nn.Linear(input_size, input_size, bias=False)
        self.dropout = nn.Dropout(drop_prob)
        self.proj = nn.Linear(input_size, input_size)

    def forward(self, x):
        c = calc_multiplicative_attention(self.W, x)       # (batch_size, seq_len, input_size)
        c = torch.relu(self.proj(c))
        return self.dropout(x + c)    # (batch_size, seq_len, input_size)


class GatedMultiplicativeSelfAttention(nn.Module):
    """
    xxx
    """

    def __init__(self, input_size, hidden_size, drop_prob):
        super(GatedMultiplicativeSelfAttention, self).__init__()
        self.W = nn.Linear(input_size, input_size, bias=False)
        self.dropout = nn.Dropout(drop_prob)
        self.gate = nn.Sequential(nn.Linear(2 * input_size, 2 * input_size, bias=False),
                                  nn.Sigmoid())
        self.rnn = nn.GRU(input_size=2 * input_size,
                          hidden_size=hidden_size,
                          bidirectional=True,
                          batch_first=True)
        self.proj = nn.Linear(2 * hidden_size, input_size)

    def forward(self, x):
        c = calc_multiplicative_attention(self.W, x)    # (batch_size, seq_len, input_size)
        h = self.dropout(calc_gated_attention(x, c, self.gate, self.rnn, self.proj))
        return x + h  # (batch_size, seq_len, input_size)


class TransformerSelfAttention(nn.Module):
    def __init__(self, input_size, num_heads, drop_prob):
        super(TransformerSelfAttention, self).__init__()
        assert(input_size % num_heads == 0)

        self.key = nn.Linear(input_size, input_size, bias=False)
        self.query = nn.Linear(input_size, input_size, bias=False)
        self.value = nn.Linear(input_size, input_size, bias=False)

        self.dropout = nn.Dropout(drop_prob)

        self.proj = nn.Linear(input_size, input_size)

        self.ln1 = nn.LayerNorm(input_size)
        self.ln2 = nn.LayerNorm(input_size)
        self.mlp = nn.Sequential(nn.Linear(input_size, 4 * input_size),
                                 nn.GELU(),
                                 nn.Linear(4 * input_size, input_size),
                                 self.dropout)

        self.n_heads = num_heads

    def forward(self, x):
        x = self.ln1(x)

        batch_size, seq_len, input_size = x.size()

        k = self.key(x).view(batch_size, seq_len, self.n_heads, input_size // self.n_heads).transpose(1, 2)
        q = self.query(x).view(batch_size, seq_len, self.n_heads, input_size // self.n_heads).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.n_heads, input_size // self.n_heads).transpose(1, 2)

        mask = (torch.eye(seq_len) == 1).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att.masked_fill_(mask, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, input_size)
        y = self.dropout(self.proj(y))

        x = x + y
        return x + self.mlp(self.ln2(x))


class ConditionalBiDAFOutput(BiDAFOutput):
    """
    xxx

    Args:
        att_size: size of attention layer output
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(ConditionalBiDAFOutput, self).__init__(hidden_size, drop_prob)
        self.proj = nn.Linear(4 * hidden_size, 2 * hidden_size)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        mod_2 = self.proj(torch.cat([mod, mod_2], 2))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


# helper functions
def calc_multiplicative_attention(weight, x):
    batch_size, seq_len, input_size = x.size()
    Wx = weight(x)  # (batch_size, seq_len, input_size)
    x_copy = torch.transpose(x, 1, 2)  # (batch_size, input_size, seq_len)

    s = torch.bmm(Wx, x_copy)  # (batch_size, seq_len, seq_len)
    mask = (torch.eye(seq_len) == 1).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    s.masked_fill_(mask, float('-inf'))  # mask out similarity between the same tokens
    a = torch.softmax(s, dim=2)  # (batch_size, seq_len, seq_len)
    c = torch.bmm(a, x)  # (batch_size, seq_len, input_size)

    return c


def calc_gated_attention(x, c, gate, rnn, proj=None):
    rnn_in = torch.cat([x, c], dim=2)       # (batch_size, seq_len, 2 * input_size)
    rnn_in = rnn_in * gate(rnn_in)      # (batch_size, seq_len, 2 * input_size)
    h, _ = rnn(rnn_in)  # (batch_size, seq_len, 2 * hidden_size)
    return h if proj is None else proj(h)   # (batch_size, seq_len, input_size)
