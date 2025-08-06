#!/usr/bin/env python3

import numpy as np
from scipy.special import softmax
from scipy.stats import norm, truncnorm


def gelu(x):
    return x*norm.cdf(x)


def gelu_prime(x):
    return x*norm.pdf(x) + norm.cdf(x)


class Transformer:
    def __init__(self,
                 n_vocab: int,
                 d_inner: int,
                 d_mlp: int,
                 n_blocks: int,
                 n_heads: int,
                 max_length: int,
                 epsilon):
        self.max_length = max_length
        self.epsilon = epsilon

        self.n_vocab = n_vocab
        self.n_blocks = n_blocks

        self.d_inner = d_inner
        self.d_mlp = d_mlp

        self.n_heads = n_heads
        
        self.w_embedding = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_vocab, self.d_embed))
        self.b_unembedding = np.zeros((1, self.n_vocab))

        self.w_q = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_blocks,
                                                                 self.n_heads,
                                                                 self.d_embed,
                                                                 self.d_inner))
        self.b_q = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_blocks,
                                                                 self.n_heads,
                                                                 1,
                                                                 self.d_inner))

        self.w_k = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_blocks,
                                                                 self.n_heads,
                                                                 self.d_embed,
                                                                 self.d_inner))
        self.b_k = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_blocks,
                                                                 self.n_heads,
                                                                 1,
                                                                 self.d_inner))

        self.w_v = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_blocks,
                                                                 self.n_heads,
                                                                 self.d_embed,
                                                                 self.d_inner))
        self.b_v = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_blocks,
                                                                 self.n_heads,
                                                                 1,
                                                                 self.d_inner))

        self.w_up = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_blocks,
                                                                  self.d_embed,
                                                                  self.d_mlp))
        self.b_up = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_blocks,
                                                                  1,
                                                                  self.d_mlp))

        self.w_down = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_blocks,
                                                                    self.d_mlp,
                                                                    self.d_embed))
        self.b_down = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_blocks,
                                                                    1,
                                                                    self.d_embed))

        self.preattention_gamma = np.ones((self.n_blocks, 1, self.d_embed))
        self.premlp_gamma = np.ones((self.n_blocks, 1, self.d_embed))
        self.final_gamma = np.ones((1, self.d_embed))

        self.preattention_beta = np.zeros((self.n_blocks, 1, self.d_embed))
        self.premlp_beta = np.zeros((self.n_blocks, 1, self.d_embed))
        self.final_beta = np.zeros((1, self.d_embed))

    @property
    def d_embed(self) -> int:
        return self.n_heads*self.d_inner
    
    def positional_encoding(self, n: int) -> np.ndarray:
        c = self.max_length
        d = self.d_embed
        
        p = np.empty((n, d))
        
        for i in range(n):
            p[i, :] = i
        
        # not the same j as in the textbook
        for j in range(0, d, 2):
            p[:, j] = np.sin(p[:, j]/(c**(j/d)))
        for j in range(1, d, 2):
            p[:, j] = np.sin(p[:, j]/(c**((j-1)/d)))
        
        return p
    
    def dropout(self, x: np.ndarray) -> np.ndarray:
        return x

    def layernorm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        zero_mean = x-np.mean(x, 0)
        adjusted_std = np.sqrt(np.var(x, 0)+self.epsilon)
        normalised = zero_mean/adjusted_std
        return gamma*normalised + beta

    def multiheaded_selfattention(self, x: np.ndarray, i: int) -> np.ndarray:
        inputs = np.repeat([x], self.n_heads, 0)
        
        queries = inputs@self.w_q[i] + self.b_q[i]
        keys = inputs@self.w_k[i] + self.b_k[i]
        values = inputs@self.w_v[i] + self.b_v[i]

        grids = queries@np.transpose(keys, (0, 2, 1))
        masked_grids = np.tril(grids)
        adjusted_grids = self.dropout(softmax(masked_grids/np.sqrt(self.d_inner), axis=2))
        patterns = adjusted_grids@values

        return np.concatenate(patterns, axis=1)

    def attention_subblock(self, x: np.ndarray, i: int) -> np.ndarray:
        raw = x
        normed = self.layernorm(x, self.preattention_gamma[i], self.preattention_beta[i])
        results = self.multiheaded_selfattention(normed, i)
        return self.dropout(results)+raw

    def mlp_subblock(self, x: np.ndarray, i: int) -> np.ndarray:
        normed = self.layernorm(x, self.premlp_gamma[i], self.premlp_beta[i])
        z = normed
        a = normed@self.w_up[i] + self.b_up[i]
        z = gelu(a)
        a = z@self.w_down[i] + self.b_down[i]
        return self.dropout(a)+normed

    def block(self, x: np.ndarray, i: int) -> np.ndarray:
        return self.mlp_subblock(self.attention_subblock(x, i), i)

    def apply(self, x: np.ndarray) -> np.ndarray:
        z = x@self.w_embedding + self.positional_encoding(len(x))
        
        z = self.dropout(z)
        
        for i in range(self.n_blocks):
            z = self.block(z, i)

        z = self.layernorm(z, self.final_gamma, self.final_beta)

        z = z@self.w_embedding.T + self.b_unembedding

        z = softmax(z, axis=1)

        return z
