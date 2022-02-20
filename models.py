"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, *args):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class BiDAFCharEmbed(nn.Module):
    """
    BiDAF model plus character embeddings.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Character level embedding:
    Using pretrained character vectors, each word is represented as a 2D
    image of size (vector size x word length).
    This 2D image is processed by a single convolutional layer and then
    max-pooled to generate a 1D vector.
    This 1D vector is concatenated to the projected word embedding vector
    and then further processed by the highway encoder.


    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAFCharEmbed, self).__init__()
        self.word_emb = layers.WordEmbedding(word_vectors=word_vectors,
                                             hidden_size=hidden_size,
                                             drop_prob=drop_prob)

        self.char_emb = layers.CharEmbedding(char_vectors=char_vectors,
                                             hidden_size=hidden_size,
                                             drop_prob=drop_prob,
                                             kernel_height=3)

        self.highway = layers.HighwayEncoder(num_layers=2, hidden_size=2 * hidden_size)

        # shrink the dimension of the output of highway encoder by a factor of 2 to match the rest of the model
        self.highway_proj = nn.Linear(2 * hidden_size, hidden_size)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs, *args):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        cw_emb = self.word_emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        qw_emb = self.word_emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        cc_emb = self.char_emb(cc_idxs)         # (batch_size, c_len, hidden_size)
        qc_emb = self.char_emb(qc_idxs)         # (batch_size, q_len, hidden_size)

        c_emb_cat = torch.cat((cw_emb, cc_emb), 2)  # (batch_size, c_len, 2 * hidden_size)
        q_emb_cat = torch.cat((qw_emb, qc_emb), 2)  # (batch_size, q_len, 2 * hidden_size)

        c_emb = self.highway_proj(self.highway(c_emb_cat))  # (batch_size, c_len, hidden_size)
        q_emb = self.highway_proj(self.highway(q_emb_cat))  # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc, c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class BiDAFAnsPtr(nn.Module):
    """
    The baseline BiDAF model for SQuAD, where the output layer is
    replaced by an Answer Pointer layer.
    See https://arxiv.org/abs/1608.07905 for details.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAFAnsPtr, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.AnswerPointer(hidden_size=hidden_size,
                                        input_size=10 * hidden_size)

    def forward(self, cw_idxs, qw_idxs, *args):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class BiDAFSelfAttention(nn.Module):
    """
    Modified BiDAF model with an additional self-attention layer, as presented in R-Net.
    See https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf for details.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
        att_dim (int): dimension of self attention layer
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0., att_dim=20):
        super(BiDAFSelfAttention, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.self_att = layers.SelfAttention(input_size=2 * hidden_size,
                                             hidden_size=hidden_size,
                                             att_dim=att_dim,
                                             drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=12 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.ExtendedBiDAFOutput(att_size=12 * hidden_size,
                                              hidden_size=hidden_size,
                                              drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, *args):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)
        self_att = self.self_att(c_enc)   # (batch_size, c_len, 2 * hidden_size)
        att = torch.cat([att, self_att, c_enc * self_att], 2)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


def init_model(name, split, **kwargs):
    name = name.lower()
    if name =='bidaf':
        return BiDAF(word_vectors=kwargs['word_vectors'],
                     hidden_size=kwargs['hidden_size'],
                     drop_prob=kwargs['drop_prob'] if split == 'train' else 0)
    elif name =='char_emb':
        return BiDAFCharEmbed(word_vectors=kwargs['word_vectors'],
                              char_vectors=kwargs['char_vectors'],
                              hidden_size=kwargs['hidden_size'],
                              drop_prob=kwargs['drop_prob'] if split == 'train' else 0)
    elif name == 'ansptr':
        return BiDAFAnsPtr(word_vectors=kwargs['word_vectors'],
                           hidden_size=kwargs['hidden_size'],
                           drop_prob=kwargs['drop_prob'] if split == 'train' else 0)
    elif name == 'self_att':
        return BiDAFSelfAttention(word_vectors=kwargs['word_vectors'],
                                  hidden_size=kwargs['hidden_size'],
                                  drop_prob=kwargs['drop_prob'] if split == 'train' else 0)

    raise ValueError(f'No model named {name}')
