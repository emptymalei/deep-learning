# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# # Transformer Explainer

import math

import matplotlib.pyplot as plt

# +
import numpy as np
import scipy as sp
import torch
from torch import nn

# -

# ## Examples

x = np.array([1, 3])

qk = np.outer(x, x)
sp.special.expit(qk)

sp.special.expit(qk).dot(x)

sp.special.expit(np.outer([1, 1], [1, 1]))

# ## The A Matrix
#
# Some examples of the A matrix.

x = np.array([1, 10])


def compare_a(a, x):
    return a, sp.special.expit(a), sp.special.expit(a).dot(x)


a_1 = np.array([[1, 0], [0, 1]])
compare_a(a_1, x)

a_2 = np.array([[1, -10], [-10, 1]])
compare_a(a_2, x)

a_3 = np.array([[-10, 1], [1, -10]])
compare_a(a_3, x)

# ## Visualize
#
# We plot out the q,k vectors and the corresponding attention.

# +
atten_similarity_q = np.array([[1], [10]])
atten_similarity_k = np.array([[1], [10]])

atten_similarity_sim = sp.special.expit(
    np.outer(atten_similarity_q, atten_similarity_k)
)


for a in [atten_similarity_q, atten_similarity_k, atten_similarity_sim]:
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.matshow(a)

    for (i, j), z in np.ndenumerate(a):
        ax.text(
            j,
            i,
            "{:0.1f}".format(z),
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.3"),
            fontsize="xx-large",
        )

    ax.set_axis_off()
# -

# ## Positional Encoding

import math


# +
class PositionalEncodingSimple:
    """Positional encoding for our transformer
    written in numpy.

    :param d_model: hidden dimension of the encoder
    :param max_len: maximum length of our positional
        encoder. The encoder can not encode sequence
        length longer than max_len.
    """

    def __init__(self, d_model: int, max_len: int = 100):
        position = np.expand_dims(np.arange(max_len), axis=1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = np.zeros((max_len, d_model))
        self.pe[:, 0::2] = np.sin(position * div_term)
        self.pe[:, 1::2] = np.cos(position * div_term)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: input to be encoded
            with shape
            `(batch_size, sequence_length, embedding_dim)`
        """
        return self.pe[: x.shape[1]]


pes = PositionalEncodingSimple(d_model=50)
x_pes_in = np.ones((1, 10, 1))

x_pes_out = pes(x=x_pes_in)

_, ax = plt.subplots(figsize=(10, 6.18))

ax.matshow(x_pes_out, cmap="cividis")
ax.set_xlabel("Embedding")
ax.set_ylabel("Temporal")

# +
_, ax = plt.subplots()

ax.plot(x_pes_out[-1, :])
ax.plot(x_pes_out[0, :])


# -

# Positional encoding in nixtla


class PositionalEmbedding(nn.Module):
    def __init__(self, hidden_size, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, hidden_size).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0) / hidden_size)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


pe = PositionalEmbedding(hidden_size=192, max_len=20)

plt.plot((torch.arange(0, 192, 2).float() * -(math.log(10000.0) / 192)).exp().numpy())


# Token Embedding


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, hidden_size):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=hidden_size,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


x_te_in = torch.ones((1, 10, 1))

# +
te = TokenEmbedding(c_in=1, hidden_size=4)

x_te = te(x_te_in)

x_te.shape

# +
te_pe = PositionalEmbedding(hidden_size=4, max_len=20)

te_pe(x)
# -


from neuralforecast.common._modules import DataEmbedding


# +
class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    """
    FullAttention
    """

    def __init__(
        self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, hidden_size, n_head, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (hidden_size // n_head)
        d_values = d_values or (hidden_size // n_head)

        self.inner_attention = attention
        self.query_projection = nn.Linear(hidden_size, d_keys * n_head)
        self.key_projection = nn.Linear(hidden_size, d_keys * n_head)
        self.value_projection = nn.Linear(hidden_size, d_values * n_head)
        self.out_projection = nn.Linear(d_values * n_head, hidden_size)
        self.n_head = n_head

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_head

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        # out = out.view(B, L, -1)

        # return self.out_projection(out), attn
        return out, attn


# +
enc_in = 1
hidden_size = 4

enc_embedding = DataEmbedding(
    c_in=enc_in, exog_input_size=0, hidden_size=hidden_size, pos_embedding=True
)
# -

x = torch.ones(size=(1, 10, 1))  # batch size: 1, history length 10, variables 1

x_embedded = enc_embedding(x)
x_embedded.shape

attention = FullAttention(mask_flag=False, output_attention=True)
attention_layer = AttentionLayer(
    attention,
    hidden_size=hidden_size,
    n_head=1,
)

attention_layer.query_projection(x_embedded).view(1, 10, 1, -1).shape

# +
al_out, al_att = attention_layer(x_embedded, x_embedded, x_embedded, attn_mask=False)

al_out.shape

# +
queries = attention_layer.query_projection(x_embedded).view(1, 10, 1, -1)
keys = attention_layer.key_projection(x_embedded).view(1, 10, 1, -1)

torch.einsum("blhe,bshe->bhls", queries, keys).shape
# -

queries.shape


class DataEmbedding_inverted(nn.Module):
    """
    DataEmbedding_inverted
    """

    def __init__(self, c_in, hidden_size, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate hidden_size]
        return self.dropout(x)


# +
i_de = DataEmbedding_inverted(10, hidden_size)

i_enc_out = i_de(x, None)

i_enc_out.shape

# -
