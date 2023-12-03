from torch import nn
import torch
from torch.nn import functional as F
import numpy as np


def make_alibi_masks(B, N, n_heads, device=torch.device('cpu')):
    inf_mask = torch.ones((1, N, N), device=device) * -torch.inf
    idx_mask = torch.arange(N, device=device).unsqueeze(0).repeat(N, 1)
    add = -torch.arange(N, device=device).unsqueeze(1)
    base_mask = torch.triu(inf_mask, 1) + torch.tril(idx_mask + add)
    base_mult = 2 ** (-8 / n_heads)
    masks = base_mask.repeat(n_heads, 1, 1)
    masks *= (base_mult ** torch.arange(1, n_heads + 1, device=device).reshape(-1, 1, 1))
    return masks.repeat(B, 1, 1)


class MyTransformerLayer(nn.Module):
    '''
    Modified from https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
    '''
    def __init__(self, embed_dim, n_heads, ffw_size, dropout, kqdim=None) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, n_heads,# kdim=kqdim, 
                                       dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, ffw_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffw_size, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.activation = nn.GELU()
    
    def attention(self, X, mask, is_causal):
        with torch.backends.cuda.sdp_kernel(
                        enable_flash=True,
                        enable_math=False,
                        enable_mem_efficient=False,
                    ):
            X, out = self.mha(X, X, X, attn_mask=mask, is_causal=is_causal, need_weights=False)
        return self.dropout(X)
    
    def ffw(self, X):
        X = self.dropout(self.activation(self.linear1(X)))
        X = self.dropout(self.linear2(X))
        return X

    def forward(self, X, mask, is_causal=True):
        X = X + self.attention(self.norm1(X), mask, is_causal)
        X = X + self.ffw(self.norm2(X))
        return X


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len: int = 5000):
        super().__init__()
        T = torch.tensor(10000.)

        idx_sin = (torch.arange(embed_dim) / embed_dim)[None, :]
        idx_cos = ((torch.arange(embed_dim) - 1) / embed_dim)[None, :]
        pos = torch.arange(max_len)[:, None]

        sin_pe = torch.sin(pos / T.pow(idx_sin))
        cos_pe = torch.cos(pos / T.pow(idx_cos))

        pe  = sin_pe * (torch.arange(embed_dim) % 2 == 0)[None, :]
        pe += cos_pe * (torch.arange(embed_dim) % 2 == 1)[None, :]
        pe = pe.unsqueeze(0)
        # here should be a tensor of size (1, max_len, embed_dim), dummy dimension is needed for proper addition

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]


class SimplerDimplerModel(nn.Module):
    def __init__(self, vocab_size, n_layers=4, embed_dim=128, n_heads=8, ffw_size=1024, \
                 dropout=0.1, kqdim=64):
        super().__init__()
        self.embeds = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, n_heads, ffw_size, dropout, \
                                       batch_first=True, activation="gelu"),
            n_layers
        )
        self.head = nn.Linear(embed_dim, vocab_size)
        self.n_heads = n_heads
    
    def forward(self, X, lengths, dtype=None):
        X = self.embeds(X)
        mask = make_alibi_masks(X.shape[0], lengths.max().item(), 
                                self.n_heads, device=X.device)
        if dtype is not None:
            mask = mask.to(dtype)
        X = self.transformer(X, mask, is_causal=True)
        return self.head(X)

if __name__ == "__main__":
    mask = make_alibi_masks(1, 3, 2)
    print(mask.shape)
    for n in range(mask.shape[0]):
        print(f'n is {n}')
        print(mask[n])
        print()