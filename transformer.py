#!/usr/bin/env python3

import numpy as np

import numpy.typing as npt

from scipy.special import softmax
from scipy.stats import norm, truncnorm


def gelu(x):
    return x*norm.cdf(x)


def gelu_prime(x):
    return x*norm.pdf(x) + norm.cdf(x)


def it(x):
    # inner transpose
    return np.transpose(x, (0, 2, 1))


class Transformer:
    def __init__(self,
                 n_vocab: int,
                 d_k: int,
                 d_mlp: int,
                 n_blocks: int,
                 n_heads: int,
                 max_length: int,
                 epsilon):
        self.max_length = max_length
        self.epsilon = epsilon

        self.n_vocab = n_vocab
        self.n_blocks = n_blocks

        self.d_k = d_k
        self.d_mlp = d_mlp

        self.n_heads = n_heads
        
        self.w_embedding = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_vocab, self.d_embed))
        
        self.w_unembedding = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.d_embed, self.n_vocab))
        self.b_unembedding = np.zeros((1, self.n_vocab))

        self.w_q = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_blocks,
                                                                 self.n_heads,
                                                                 self.d_embed,
                                                                 self.d_k))
        self.b_q = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_blocks,
                                                                 self.n_heads,
                                                                 1,
                                                                 self.d_k))

        self.w_k = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_blocks,
                                                                 self.n_heads,
                                                                 self.d_embed,
                                                                 self.d_k))
        self.b_k = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_blocks,
                                                                 self.n_heads,
                                                                 1,
                                                                 self.d_k))

        self.w_v = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_blocks,
                                                                 self.n_heads,
                                                                 self.d_embed,
                                                                 self.d_embed))
        self.b_v = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_blocks,
                                                                 self.n_heads,
                                                                 1,
                                                                 self.d_embed))

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

        self.pre_attention_gamma = np.ones((self.n_blocks, 1, self.d_embed))
        self.pre_mlp_gamma = np.ones((self.n_blocks, 1, self.d_embed))
        self.final_gamma = np.ones((1, self.d_embed))

        self.pre_attention_beta = np.zeros((self.n_blocks, 1, self.d_embed))
        self.pre_mlp_beta = np.zeros((self.n_blocks, 1, self.d_embed))
        self.final_beta = np.zeros((1, self.d_embed))

    @property
    def d_embed(self) -> int:
        return self.n_heads*self.d_k
    
    def positional_encoding(self, n: int) -> npt.NDArray[np.floating]:
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
    
    def dropout(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        return x

    def layernorm(self,
                  x: npt.NDArray[np.floating],
                  gamma: npt.NDArray[np.floating],
                  beta: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        zero_mean = x-np.mean(x, 0)
        adjusted_std = np.sqrt(np.var(x, 0)+self.epsilon)
        normalised = zero_mean/adjusted_std
        return gamma*normalised + beta

    def mask(self, ps: npt.NDArray[np.floating]):
        # works on one or many grids

        return ps - np.triu(np.inf*np.ones_like(ps), k=1)

    def multiheaded_selfattention(self, x: npt.NDArray[np.floating], i: int) -> npt.NDArray[np.floating]:
        n = len(x)
        
        self.pre_attention[i] = np.repeat([x], self.n_heads, 0)
        
        queries = self.pre_attention[i]@self.w_q[i] + self.b_q[i]
        keys = self.pre_attention[i]@self.w_k[i] + self.b_k[i]
        values = self.pre_attention[i]@self.w_v[i] + self.b_v[i]

        patterns = queries@np.transpose(keys, (0, 2, 1))
        masked_patterns = self.mask(patterns)
        self.adjusted_patterns[i] = self.dropout(softmax(masked_patterns/np.sqrt(self.d_k), axis=2))
        grids = self.adjusted_patterns[i]@values

        return np.sum(grids, 0)

    def attention_subblock(self, x: npt.NDArray[np.floating], i: int) -> npt.NDArray[np.floating]:
        raw = x
        normed = self.layernorm(x, self.pre_attention_gamma[i], self.pre_attention_beta[i])
        results = self.multiheaded_selfattention(normed, i)
        return self.dropout(results)+raw

    def mlp_subblock(self, x: npt.NDArray[np.floating], i: int) -> npt.NDArray[np.floating]:
        self.pre_mlp[i] = self.layernorm(x, self.pre_mlp_gamma[i], self.pre_mlp_beta[i])
        self.pre_gelu[i] = self.pre_mlp[i]@self.w_up[i] + self.b_up[i]
        self.post_gelu[i] = gelu(self.pre_gelu[i])
        a = self.post_gelu[i]@self.w_down[i] + self.b_down[i]
        return self.dropout(a)+self.pre_mlp[i]

    def block(self, x: npt.NDArray[np.floating], i: int) -> npt.NDArray[np.floating]:
        self.post_attention[i] = self.attention_subblock(x, i)
        return self.mlp_subblock(self.post_attention[i], i)

    def apply(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        n = len(x)
        
        embeddings = x@self.w_embedding + self.positional_encoding(len(x))

        self.z = np.empty((self.n_blocks+1, n, self.d_embed))

        self.pre_mlp = np.empty((self.n_blocks, n, self.d_embed))
        self.pre_gelu = np.empty((self.n_blocks, n, self.d_mlp))
        self.post_gelu = np.empty((self.n_blocks, n, self.d_mlp))
        self.post_attention = np.empty((self.n_blocks, n, self.d_embed))
        self.adjusted_patterns = np.empty((self.n_blocks, self.n_heads, n, n))
        self.pre_attention = np.empty((self.n_blocks, self.n_heads, n, self.d_embed))
        
        self.z[0] = self.dropout(embeddings)

        for i in range(self.n_blocks):
            self.z[i+1] = self.block(self.z[i], i)

        self.normalised = self.layernorm(self.z[-1], self.final_gamma, self.final_beta)

        logits = self.normalised@self.w_unembedding + self.b_unembedding

        self.probs = softmax(logits, axis=1)

        return self.probs

    def backprop(self, x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        n = len(x)
        self.apply(x)

        diffs = self.probs - y
        
        self.grad_b_unembedding = np.sum(diffs, axis=0, keepdims=True)
        self.grad_w_unembedding = self.normalised.T@diffs

        self.grad_final_beta = np.sum(diffs@self.w_unembedding.T, axis=0, keepdims=True)
        self.grad_final_gamma = np.sum(self.w_unembedding@diffs.T@self.z[-1], axis=0, keepdims=True)

        self.grad_b_down = np.empty((n, self.n_blocks, 1, self.d_embed))
        self.grad_w_down = np.empty((n, self.n_blocks, self.d_mlp, self.d_embed))

        self.grad_b_up = np.empty((n, self.n_blocks, 1, self.d_mlp))
        self.grad_w_up = np.empty((n, self.n_blocks, self.d_embed, self.d_mlp))

        self.grad_pre_mlp_beta = np.empty((n, self.n_blocks, 1, self.d_embed))
        self.grad_pre_mlp_gamma = np.empty((n, self.n_blocks, 1, self.d_embed))

        self.grad_w_v = np.empty((n, self.n_blocks, self.n_heads, self.d_embed, self.d_embed))
        self.grad_w_k = np.empty((n, self.n_blocks, self.n_heads, self.d_embed, self.d_k))
        self.grad_w_q = np.empty((n, self.n_blocks, self.n_heads, self.d_embed, self.d_k))

        m = np.empty((n, n))
        s = np.empty((self.n_blocks, self.n_heads, n, n, n))

        for i in range(n):
            u0 = diffs[i, np.newaxis]@self.w_unembedding.T@np.diag(self.final_gamma[0])
            
            self.grad_b_down[i, -1] = u0
            self.grad_w_down[i, -1] = self.post_gelu[-1, i, np.newaxis].T@u0

            u1 = u0@self.w_down[-1].T@np.diag(gelu_prime(self.pre_gelu[-1, i]))

            self.grad_b_up[i, -1] = u1
            self.grad_w_up[i, -1] = self.pre_mlp[-1, i, np.newaxis].T@u1

            u2 = u0 + u1@self.w_up[-1].T

            self.grad_pre_mlp_beta[i, -1] = u2
            self.grad_pre_mlp_gamma[i, -1] = np.sum(u2.T@self.post_attention[-1, i, np.newaxis])

            u3 = u2@np.diag(self.pre_mlp_gamma[-1, 0])

            self.grad_w_v[i, -1] = it(self.adjusted_patterns[-1, :, np.newaxis, i]@self.pre_attention[-1])@u3

            u4 = u3@it(self.w_v[-1])
            
            self.grad_w_k[i, -1] = (1/np.sqrt(self.d_k))*it(self.pre_attention[-1])@m@s[-1, :, i]@self.pre_attention[-1]@u3.T@self.pre_attention[-1, :, i, np.newaxis]@self.w_q[-1]
            self.grad_w_q[i, -1] = (1/np.sqrt(self.d_k))*it(self.pre_attention[-1, :, i, np.newaxis])@u3@it(self.pre_attention[-1])@s[-1, :, i]@m.T@self.pre_attention[-1]@self.w_k[-1]

            

        for (v, g) in [(self.b_unembedding, self.grad_b_unembedding),
                       (self.w_unembedding, self.grad_w_unembedding),
                       (self.final_beta, self.grad_final_beta),
                       (self.final_gamma, self.grad_final_gamma)]:
            assert v.shape == g.shape
