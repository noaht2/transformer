#!/usr/bin/env python3

import numpy as np

import numpy.typing as npt

from scipy.special import softmax
from scipy.stats import norm, truncnorm


def gelu(x):
    return x*norm.cdf(x)


def gelu_prime(x):
    return x*norm.pdf(x) + norm.cdf(x)


def it(x: np.ndarray) -> np.ndarray:
    # inner transpose
    return np.transpose(x, (0, 2, 1))


def diag2(x: np.ndarray) -> np.ndarray:
    # diag2(x)[i] = np.diag(np.diag(x[i]))
    assert len(x.shape) == 3
    assert x.shape[1] == x.shape[2]
    return x*np.eye(x.shape[1])[np.newaxis].repeat(x.shape[0], axis=0)


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

        self.n_flavours = 4  # values, gradients, m, s
        
        self.w_embedding = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_flavours, self.n_vocab, self.d_embed))
        
        self.w_unembedding = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_flavours, self.d_embed, self.n_vocab))
        self.b_unembedding = np.zeros((self.n_flavours, 1, self.n_vocab))

        self.w_q = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_flavours,
                                                                 self.n_blocks,
                                                                 self.n_heads,
                                                                 self.d_embed,
                                                                 self.d_k))
        self.b_q = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_flavours,
                                                                 self.n_blocks,
                                                                 self.n_heads,
                                                                 1,
                                                                 self.d_k))

        self.w_k = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_flavours,
                                                                 self.n_blocks,
                                                                 self.n_heads,
                                                                 self.d_embed,
                                                                 self.d_k))
        self.b_k = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_flavours,
                                                                 self.n_blocks,
                                                                 self.n_heads,
                                                                 1,
                                                                 self.d_k))

        self.w_v = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_flavours,
                                                                 self.n_blocks,
                                                                 self.n_heads,
                                                                 self.d_embed,
                                                                 self.d_embed))
        self.b_v = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_flavours,
                                                                 self.n_blocks,
                                                                 self.n_heads,
                                                                 1,
                                                                 self.d_embed))

        self.w_up = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_flavours,
                                                                  self.n_blocks,
                                                                  self.d_embed,
                                                                  self.d_mlp))
        self.b_up = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_flavours,
                                                                  self.n_blocks,
                                                                  1,
                                                                  self.d_mlp))

        self.w_down = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_flavours,
                                                                    self.n_blocks,
                                                                    self.d_mlp,
                                                                    self.d_embed))
        self.b_down = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=(self.n_flavours,
                                                                    self.n_blocks,
                                                                    1,
                                                                    self.d_embed))

        self.pre_attention_gamma = np.ones((self.n_flavours, self.n_blocks, 1, self.d_embed))
        self.pre_mlp_gamma = np.ones((self.n_flavours, self.n_blocks, 1, self.d_embed))
        self.final_gamma = np.ones((self.n_flavours, 1, self.d_embed))

        self.pre_attention_beta = np.zeros((self.n_flavours, self.n_blocks, 1, self.d_embed))
        self.pre_mlp_beta = np.zeros((self.n_flavours, self.n_blocks, 1, self.d_embed))
        self.final_beta = np.zeros((self.n_flavours, 1, self.d_embed))

        self.params = [self.w_embedding,
                       self.w_unembedding,
                       self.b_unembedding,
                       self.w_q,
                       # self.b_q,
                       self.w_k,
                       # self.b_k,
                       self.w_v,
                       # self.b_v,
                       self.w_up,
                       # self.b_up,
                       self.w_down,
                       # self.b_down,
                       self.pre_attention_gamma,
                       self.pre_mlp_beta,
                       self.final_beta]

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

    def mask(self, ps: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        # works on one or many grids

        return ps - np.triu(np.inf*np.ones_like(ps), k=1)

    def multiheaded_selfattention(self, x: npt.NDArray[np.floating], i: int) -> npt.NDArray[np.floating]:
        n = len(x)
        
        self.pre_attention[i] = np.repeat([x], self.n_heads, 0)
        
        queries = self.pre_attention[i]@self.w_q[0, i] + self.b_q[0, i]
        self.keys[i] = self.pre_attention[i]@self.w_k[0, i] + self.b_k[0, i]
        values = self.pre_attention[i]@self.w_v[0, i] + self.b_v[0, i]

        self.patterns[i] = queries@np.transpose(self.keys[i], (0, 2, 1))
        self.masked_patterns[i] = self.mask(self.patterns[i])/np.sqrt(self.d_k)
        self.adjusted_patterns[i] = self.dropout(softmax(self.masked_patterns[i], axis=2))
        grids = self.adjusted_patterns[i]@values

        return np.sum(grids, 0)

    def attention_subblock(self, x: npt.NDArray[np.floating], i: int) -> npt.NDArray[np.floating]:
        raw = x
        normed = self.layernorm(x, self.pre_attention_gamma[0, i], self.pre_attention_beta[0, i])
        results = self.multiheaded_selfattention(normed, i)
        return self.dropout(results) + raw

    def mlp_subblock(self, x: npt.NDArray[np.floating], i: int) -> npt.NDArray[np.floating]:
        self.pre_mlp[i] = self.layernorm(x, self.pre_mlp_gamma[0, i], self.pre_mlp_beta[0, i])
        self.pre_gelu[i] = self.pre_mlp[i]@self.w_up[0, i] + self.b_up[0, i]
        self.post_gelu[i] = gelu(self.pre_gelu[i])
        a = self.post_gelu[i]@self.w_down[0, i] + self.b_down[0, i]
        return self.dropout(a) + self.pre_mlp[i]

    def block(self, x: npt.NDArray[np.floating], i: int) -> npt.NDArray[np.floating]:
        self.post_attention[i] = self.attention_subblock(x, i)
        return self.mlp_subblock(self.post_attention[i], i)

    def apply(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        self.x = x
        n = len(self.x)
        
        embeddings = x@self.w_embedding[0] + self.positional_encoding(n)

        self.z = np.empty((self.n_blocks+1, n, self.d_embed))

        self.pre_mlp = np.empty((self.n_blocks, n, self.d_embed))
        self.pre_gelu = np.empty((self.n_blocks, n, self.d_mlp))
        self.post_gelu = np.empty((self.n_blocks, n, self.d_mlp))
        self.pre_attention = np.empty((self.n_blocks, self.n_heads, n, self.d_embed))
        self.keys = np.empty((self.n_blocks, self.n_heads, n, self.d_k))
        self.masked_patterns = np.empty((self.n_blocks, self.n_heads, n, n))
        self.patterns = np.empty((self.n_blocks, self.n_heads, n, n))
        self.adjusted_patterns = np.empty((self.n_blocks, self.n_heads, n, n))
        self.post_attention = np.empty((self.n_blocks, n, self.d_embed))
        
        self.z[0] = self.dropout(embeddings)

        for i in range(self.n_blocks):
            self.z[i+1] = self.block(self.z[i], i)

        self.normalised = self.layernorm(self.z[-1], self.final_gamma[0], self.final_beta[0])

        self.logits = self.normalised@self.w_unembedding[0] + self.b_unembedding[0]

        self.probs = softmax(self.logits, axis=1)

        return self.probs

    def backprop(self, data: npt.NDArray[np.floating]) -> None:
        x = data[:-1]
        y = data[1:]

        n = len(x)
        self.apply(x)

        diffs = self.probs - y
        
        self.b_unembedding[1] = np.sum(diffs, axis=0, keepdims=True)
        self.w_unembedding[1] = self.normalised.T@diffs

        self.final_beta[1] = np.sum(diffs@self.w_unembedding[0].T, axis=0, keepdims=True)
        self.final_gamma[1] = np.sum(self.w_unembedding[0]@diffs.T@self.z[-1], axis=0, keepdims=True)

        self.b_down[1] = np.zeros((self.n_blocks, 1, self.d_embed))
        self.w_down[1] = np.zeros((self.n_blocks, self.d_mlp, self.d_embed))

        self.b_up[1] = np.zeros((self.n_blocks, 1, self.d_mlp))
        self.w_up[1] = np.zeros((self.n_blocks, self.d_embed, self.d_mlp))

        self.pre_mlp_beta[1] = np.zeros((self.n_blocks, 1, self.d_embed))
        self.pre_mlp_gamma[1] = np.zeros((self.n_blocks, 1, self.d_embed))

        self.w_v[1] = np.zeros((self.n_blocks, self.n_heads, self.d_embed, self.d_embed))
        self.w_k[1] = np.zeros((self.n_blocks, self.n_heads, self.d_embed, self.d_k))
        self.w_q[1] = np.zeros((self.n_blocks, self.n_heads, self.d_embed, self.d_k))

        self.through_attention = np.zeros((n, n, self.n_heads, self.d_embed, self.d_embed))
        self.from_attention = np.zeros((n, n, self.n_heads, 1, self.d_embed))

        self.pre_attention_beta[1] = np.zeros((self.n_blocks, 1, self.d_embed))
        self.pre_attention_gamma[1] = np.zeros((self.n_blocks, 1, self.d_embed))

        self.w_embedding[1] = np.empty((self.n_vocab, self.d_embed))

        s = np.empty((self.n_blocks, self.n_heads, n, n, n))

        u0 = diffs@self.w_unembedding[0].T@np.diag(self.final_gamma[0].reshape((self.d_embed,)))
        u1 = np.empty((n, self.d_mlp))
        u2 = np.empty((n, self.d_embed))
        u3 = np.empty((n, self.d_embed))
        u4 = np.empty((n, self.n_heads, 1, self.d_embed))

        for l in range(self.n_blocks-1, -1, -1):
            for i in range(n):
                m = np.eye(n)
                m[(i+1):, :] = 0
                
                s = np.empty((self.n_blocks, self.n_heads, n, n, n))
                s[l, :, i] = -self.adjusted_patterns[l, :, i, :, np.newaxis]@self.adjusted_patterns[l, :, np.newaxis, i]
                s[l, :, i] += self.adjusted_patterns[l, :, i, :, np.newaxis]*np.eye(n)
                
            
                self.b_down[1, l] += u0[i]/n
                self.w_down[1, l] += self.post_gelu[l, i, np.newaxis].T@u0[i, np.newaxis]/n

                u1[i] = u0[i, np.newaxis]@self.w_down[0, l].T@np.diag(gelu_prime(self.pre_gelu[l, i]))

                self.b_up[1, l] += u1[i, np.newaxis]/n
                self.w_up[1, l] += self.pre_mlp[l, i, np.newaxis].T@u1[i, np.newaxis]/n

                u2[i] = u0[i, np.newaxis] + u1[i, np.newaxis]@self.w_up[0, l].T
                
                self.pre_mlp_beta[1, l] += u2[i, np.newaxis]/n
                self.pre_mlp_gamma[1, l] += np.sum(u2[i, :, np.newaxis]@self.post_attention[l, i, np.newaxis], axis=0, keepdims=True)/n

                u3[i] = u2[i, np.newaxis]@np.diag(self.pre_mlp_gamma[0, l].reshape((self.d_embed,)))
                
                self.w_v[1, l] += it(self.adjusted_patterns[l, :, np.newaxis, i]@self.pre_attention[l])@u3[i, np.newaxis]/n
                
                u4[i] = u3[i, np.newaxis]@it(self.w_v[0, l])
                
                self.w_k[1, l] += (1/np.sqrt(self.d_k))*it(self.pre_attention[l])@m@s[l, :, i]@self.pre_attention[l]@u3[i, :, np.newaxis]@self.pre_attention[l, :, i, np.newaxis]@self.w_q[0, l]/n
                self.w_q[1, l] += (1/np.sqrt(self.d_k))*it(self.pre_attention[l, :, i, np.newaxis])@u3[i, np.newaxis]@it(self.pre_attention[l])@s[l, :, i]@m.T@self.pre_attention[l]@self.w_k[0, l]/n
                
                for j in range(n):
                    self.through_attention[i, j] = self.adjusted_patterns[l, :, i, j, np.newaxis, np.newaxis]*np.eye(self.d_embed)
                    if i == j:
                        w = self.pre_attention[l]
                        a = self.w_k[0, l]@it(self.w_q[0, l])
                        r = w[:, i, np.newaxis]@diag2(a)
                        x = w@a
                        x[:, i, np.newaxis] = r
                    else:
                        r = self.pre_attention[l, :, i, np.newaxis]@self.w_k[0, l]@it(self.w_q[0, l])
                        x = np.zeros((self.n_heads, n, self.d_embed))
                        x[:, j, np.newaxis] = r
                    y = s[l, :, i, np.newaxis, j]@m.T
                    z = y@(x/np.sqrt(self.d_k))
                    self.through_attention[i, j] += z.repeat(self.d_embed, axis=1)
                self.from_attention[i] = u4[i]@self.through_attention[i]

            u5 = self.from_attention.sum(axis=(0, 2, 3))

            for i in range(n):
                self.pre_attention_beta[1, l] += u5[i, np.newaxis]/n
                self.pre_attention_gamma[1, l] += np.sum(u5[i, :, np.newaxis]@self.z[l-1, i, np.newaxis], axis=0, keepdims=True)/n

            u0 = u5+u3

        for i in range(n):
            self.w_embedding[1] += self.x[i, :, np.newaxis]@u0[i, np.newaxis]/n

    def train(self, data, runs) -> None:
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 0.00001
        rho = 0.001

        self.apply(data[0, :-1])
        print(self.probs)

        for i in range(runs):
            if (i % (runs/10)) == 0:
                print(i)
            self.backprop(data[0])
            for w in self.params:
                w[0] -= rho*w[1]

        self.apply(data[0, :-1])
        print(self.probs)
        print()
