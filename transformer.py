#!/usr/bin/env python3

import numbers
import typing

import numpy as np

import numpy.typing as npt

from scipy.special import softmax
from scipy.stats import norm, truncnorm


def gelu(x: npt.ArrayLike) -> npt.ArrayLike:
    return x*norm.cdf(x)


def gelu_prime(x: npt.ArrayLike) -> npt.ArrayLike:
    return x*norm.pdf(x) + norm.cdf(x)


def it(x: np.ndarray) -> np.ndarray:
    # inner transpose
    assert len(x.shape) == 3
    return x.transpose(0, 2, 1)


def diag2(x: np.ndarray) -> np.ndarray:
    # diag2(x)[i] = np.diag(np.diag(x[i]))
    assert len(x.shape) == 3
    assert x.shape[1] == x.shape[2]
    return x*np.eye(x.shape[1])[np.newaxis].repeat(x.shape[0], axis=0)


def check(*xs: np.ndarray) -> None:
    for x in xs:
        if np.any(np.isnan(x)):
            raise ValueError(x.shape)


class Transformer:
    def __init__(self,
                 n_vocab: typing.SupportsIndex,
                 d_k: int,
                 d_v: int,
                 d_mlp: typing.SupportsIndex,
                 n_blocks: typing.SupportsIndex,
                 n_heads: int,
                 max_length: int,
                 epsilon: float):
        self.max_length = max_length
        self.epsilon = epsilon

        self.n_vocab = n_vocab
        self.n_blocks = n_blocks

        self.d_k = d_k
        self.d_v = d_v
        self.d_mlp = d_mlp

        self.n_heads = n_heads

        self.n_flavours = 4  # values, gradients, m, s
        
        self.w_embedding = np.nan*np.ones((self.n_flavours, self.n_vocab, self.d_embed))
        
        self.w_unembedding = np.nan*np.ones((self.n_flavours, self.d_embed, self.n_vocab))
        self.b_unembedding = np.nan*np.ones((self.n_flavours, 1, self.n_vocab))

        self.w_q = np.nan*np.ones((self.n_flavours,
                                   self.n_blocks,
                                   self.n_heads,
                                   self.d_embed,
                                   self.d_k))
        self.b_q = np.nan*np.ones((self.n_flavours,
                                   self.n_blocks,
                                   self.n_heads,
                                   1,
                                   self.d_k))

        self.w_k = np.nan*np.ones((self.n_flavours,
                                   self.n_blocks,
                                   self.n_heads,
                                   self.d_embed,
                                   self.d_k))
        self.b_k = np.nan*np.ones((self.n_flavours,
                                   self.n_blocks,
                                   self.n_heads,
                                   1,
                                   self.d_k))

        self.w_v = np.nan*np.ones((self.n_flavours,
                                   self.n_blocks,
                                   self.n_heads,
                                   self.d_embed,
                                   self.d_v))
        self.b_v = np.nan*np.ones((self.n_flavours,
                                   self.n_blocks,
                                   self.n_heads,
                                   1,
                                   self.d_v))

        self.w_o = np.nan*np.ones((self.n_flavours,
                                   self.n_blocks,
                                   self.n_heads*self.d_v,
                                   self.d_embed))
        self.b_o = np.nan*np.ones((self.n_flavours,
                                   self.n_blocks,
                                   1,
                                   self.d_embed))

        self.w_up = np.nan*np.ones((self.n_flavours,
                                    self.n_blocks,
                                    self.d_embed,
                                    self.d_mlp))
        self.b_up = np.nan*np.ones((self.n_flavours,
                                    self.n_blocks,
                                    1,
                                    self.d_mlp))

        self.w_down = np.nan*np.ones((self.n_flavours,
                                      self.n_blocks,
                                      self.d_mlp,
                                      self.d_embed))
        self.b_down = np.nan*np.ones((self.n_flavours,
                                      self.n_blocks,
                                      1,
                                      self.d_embed))

        self.gamma = np.nan*np.ones((self.n_flavours, 2*self.n_blocks+1, 1, self.d_embed))
        self.beta = np.nan*np.ones((self.n_flavours, 2*self.n_blocks+1, 1, self.d_embed))

        self.weights = [self.w_embedding,
                        self.w_unembedding,
                        self.w_q,
                        self.w_k,
                        self.w_v,
                        self.w_o,
                        self.w_up,
                        self.w_down]

        self.biases = [self.b_unembedding,
                       self.b_q,
                       self.b_k,
                       self.b_v,
                       self.b_o,
                       self.b_up,
                       self.b_down]

        self.gammas = [self.gamma]

        self.betas = [self.beta]

        self.params = self.weights+self.biases+self.gammas+self.betas

        for w in self.weights:
            w[0] = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=w[0].shape)

        for b in self.biases:
            b[0] = 0
        
        check(self.b_q[0])

        for gamma in self.gammas:
            gamma[0] = 1

        for beta in self.betas:
            beta[0] = 0

        for p in self.params:
            p[2] = p[3] = 0

    @property
    def d_embed(self) -> int:
        return self.n_heads*self.d_k

    @property
    def d_stack(self) -> int:
        return self.n_heads*self.d_v
    
    def positional_encoding(self, n: int) -> npt.NDArray[np.floating]:
        c = self.max_length
        d = self.d_embed
        
        p = np.nan*np.ones((n, d))
        
        for i in range(n):
            p[i, :] = i
            
        # not the same j as in the textbook
        for j in range(0, d, 2):
            p[:, j] = np.sin(p[:, j]/(c**(j/d)))
        for j in range(1, d, 2):
            p[:, j] = np.cos(p[:, j]/(c**((j-1)/d)))

        check(p)
        
        return p
    
    def dropout(self, x: np.ndarray, p: float = 0.9) -> np.ndarray:
        return x*np.random.default_rng().choice([0, 1], size=x.shape, p=[1-p, p])

    def layernorm(self, x: np.ndarray, i: int) -> None:
        self.seminormed[i] = x-np.mean(x, -1, keepdims=True)
        adjusted_std = np.sqrt(np.var(x, -1, keepdims=True)+self.epsilon)
        self.normed[i] = self.seminormed[i]/adjusted_std
        self.renormed[i] = self.gamma[0, i]*self.normed[i] + self.beta[0, i]

    def d_layernorm(self, i: int) -> npt.NDArray[np.floating]:
        from_gamma = np.diag(self.gamma[0, i].reshape((self.d_embed,)))
        
        k = -(1/self.d_embed)*(self.seminormed[i, :, :, np.newaxis]@self.seminormed[i, :, np.newaxis])
        c = (1/self.d_embed)*np.sum(np.square(self.seminormed[i]), axis=1)[:, np.newaxis, np.newaxis]+self.epsilon
        from_std = (c*np.eye(self.d_embed)+k)/c**1.5

        from_mean = np.eye(self.d_embed)-np.ones((self.d_embed, self.d_embed))/self.d_embed
        
        return from_gamma@from_std@from_mean

    def mask(self, ps: np.ndarray) -> npt.NDArray[np.floating]:
        # works on one or many grids

        return ps - np.triu(np.inf*np.ones_like(ps), k=1)

    def multiheaded_selfattention(self, x: np.ndarray, i: int) -> npt.NDArray[np.floating]:
        check(x)
        
        self.pre_attention[i] = np.repeat([x], self.n_heads, 0)
        check(self.pre_attention[i])

        check(self.w_q[0, i])
        check(self.b_q[0, i])
        self.queries[i] = self.pre_attention[i]@self.w_q[0, i] + self.b_q[0, i]
        check(self.queries[i])
        self.keys[i] = self.pre_attention[i]@self.w_k[0, i] + self.b_k[0, i]
        check(self.keys[i])
        self.values[i] = self.pre_attention[i]@self.w_v[0, i] + self.b_v[0, i]
        check(self.values[i])

        self.patterns[i] = self.queries[i]@it(self.keys[i])
        check(self.patterns[i])
        self.masked_patterns[i] = self.mask(self.patterns[i])/np.sqrt(self.d_k)
        check(self.masked_patterns[i])
        self.adjusted_patterns[i] = self.dropout(softmax(self.masked_patterns[i], axis=2))
        grids = self.adjusted_patterns[i]@self.values[i]

        self.stack[i] = grids.transpose(1, 0, 2).reshape((self.n, self.d_stack))

        return self.stack[i]@self.w_o[0, i] + self.b_o[0, i]

    def attention_subblock(self, x: np.ndarray, i: int) -> npt.NDArray[np.floating]:
        raw = x
        self.layernorm(x, 2*i)
        results = self.multiheaded_selfattention(self.renormed[2*i], i)
        return self.dropout(results) + raw

    def mlp_subblock(self, x: np.ndarray, i: int) -> npt.NDArray[np.floating]:
        self.layernorm(x, 2*i+1)
        self.pre_mlp[i] = self.renormed[2*i+1]
        self.pre_gelu[i] = self.pre_mlp[i]@self.w_up[0, i] + self.b_up[0, i]
        self.post_gelu[i] = gelu(self.pre_gelu[i])
        a = self.post_gelu[i]@self.w_down[0, i] + self.b_down[0, i]
        return self.dropout(a) + self.pre_mlp[i]

    def block(self, x: np.ndarray, i: int) -> npt.NDArray[np.floating]:
        self.post_attention[i] = self.attention_subblock(x, i)
        return self.mlp_subblock(self.post_attention[i], i)

    def apply(self, x: np.ndarray) -> npt.NDArray[np.floating]:
        self.x = x
        n = self.n = len(x)
        
        embeddings = x@self.w_embedding[0] + self.positional_encoding(self.n)

        self.z = np.nan*np.ones((self.n_blocks+1, n, self.d_embed))

        self.seminormed = np.nan*np.ones((2*self.n_blocks+1, n, self.d_embed))
        self.normed = np.nan*np.ones((2*self.n_blocks+1, n, self.d_embed))
        self.renormed = np.nan*np.ones((2*self.n_blocks+1, n, self.d_embed))

        self.pre_mlp = np.nan*np.ones((self.n_blocks, n, self.d_embed))
        self.pre_gelu = np.nan*np.ones((self.n_blocks, n, self.d_mlp))
        self.post_gelu = np.nan*np.ones((self.n_blocks, n, self.d_mlp))
        self.pre_attention = np.nan*np.ones((self.n_blocks, self.n_heads, n, self.d_embed))
        self.queries = np.nan*np.ones((self.n_blocks, self.n_heads, n, self.d_k))
        self.keys = np.nan*np.ones((self.n_blocks, self.n_heads, n, self.d_k))
        self.values = np.nan*np.ones((self.n_blocks, self.n_heads, n, self.d_v))
        self.masked_patterns = np.nan*np.ones((self.n_blocks, self.n_heads, n, n))
        self.patterns = np.nan*np.ones((self.n_blocks, self.n_heads, n, n))
        self.adjusted_patterns = np.nan*np.ones((self.n_blocks, self.n_heads, n, n))
        self.stack = np.nan*np.ones((self.n_blocks, n, self.d_stack))
        self.post_attention = np.nan*np.ones((self.n_blocks, n, self.d_embed))
        
        self.z[0] = self.dropout(embeddings)

        for i in range(self.n_blocks):
            self.z[i+1] = self.block(self.z[i], i)

        self.layernorm(self.z[-1], -1)

        self.logits = self.renormed[-1]@self.w_unembedding[0] + self.b_unembedding[0]

        self.probs = softmax(self.logits, axis=-1)

        return self.probs

    def backprop(self, data: np.ndarray) -> None:
        x = data[:-1]
        y = data[1:]
        n = len(x)

        self.apply(x)

        diffs = (self.probs - y)/n
        
        self.b_unembedding[1] = np.sum(diffs, axis=0, keepdims=True)
        self.w_unembedding[1] = self.renormed[-1].T@diffs

        um1 = diffs@self.w_unembedding[0].T

        self.beta[1, -1] = np.sum(um1, axis=0, keepdims=True)
        self.gamma[1, -1] = np.sum(um1*self.normed[-1], axis=0, keepdims=True)

        self.b_down[1] = np.zeros((self.n_blocks, 1, self.d_embed))
        self.w_down[1] = np.zeros((self.n_blocks, self.d_mlp, self.d_embed))

        self.b_up[1] = np.zeros((self.n_blocks, 1, self.d_mlp))
        self.w_up[1] = np.zeros((self.n_blocks, self.d_embed, self.d_mlp))

        self.beta[1, :-1] = 0
        self.gamma[1, :-1] = 0

        self.w_o[1] = 0
        self.b_o[1] = 0
        self.w_v[1] = np.zeros((self.n_blocks, self.n_heads, self.d_embed, self.d_v))
        self.b_v[1] = 0
        self.w_k[1] = np.zeros((self.n_blocks, self.n_heads, self.d_embed, self.d_k))
        self.b_k[1] = 0
        self.w_q[1] = np.zeros((self.n_blocks, self.n_heads, self.d_embed, self.d_k))
        self.b_q[1] = 0

        self.through_attention = np.nan*np.ones((n, n, self.n_heads, 1, self.d_embed))

        self.w_embedding[1] = np.zeros((self.n_vocab, self.d_embed))

        s = np.nan*np.ones((self.n_blocks, self.n_heads, n, n, n))

        u0 = um1.reshape((n, 1, self.d_embed))@self.d_layernorm(-1)
        u1 = np.nan*np.ones((n, 1, self.d_mlp))
        u2 = np.nan*np.ones((n, 1, self.d_embed))
        u4 = np.nan*np.ones((n, 1, self.d_embed))
        u3 = np.nan*np.ones((n, 1, self.d_embed))
        u7 = np.nan*np.ones((n, 1, self.d_stack))
        u8 = np.nan*np.ones((n, self.n_heads, 1, self.d_v))

        for l in range(self.n_blocks-1, -1, -1):
            for i in range(n):
                m = np.eye(n)
                m[(i+1):, :] = 0
                
                s = np.nan*np.ones((self.n_blocks, self.n_heads, n, n, n))
                s[l, :, i] = -self.adjusted_patterns[l, :, i, :, np.newaxis]@self.adjusted_patterns[l, :, np.newaxis, i]
                s[l, :, i] += self.adjusted_patterns[l, :, i, :, np.newaxis]*np.eye(n)
                
                self.b_down[1, l] += u0[i]
                self.w_down[1, l] += self.post_gelu[l, i, np.newaxis].T@u0[i]

                u1[i] = u0[i]@self.w_down[0, l].T@np.diag(gelu_prime(self.pre_gelu[l, i]))

                self.b_up[1, l] += u1[i]
                self.w_up[1, l] += self.pre_mlp[l, i, np.newaxis].T@u1[i]

                u2[i] = u1[i]@self.w_up[0, l].T + u0[i]

                self.beta[1, 2*l+1] += u2[i]
                self.gamma[1, 2*l+1] += u2[i]*self.normed[2*l+1, i, np.newaxis]

                # print(u3[i].shape, u2[i].shape, self.d_layernorm(2*l+1).shape)
                u4[i] = u2[i]@self.d_layernorm(2*l+1)[i]

                u3[i] = u4[i]

                self.w_o[1, l] += self.stack[l, i, :, np.newaxis]@u3[i]
                self.b_o[1, l] += u3[i]

                u7[i] = u3[i]@self.w_o[0, l].T
                u8[i] = u7[i].reshape((self.n_heads, 1, self.d_v))

                self.w_v[1, l] += it(self.adjusted_patterns[l, :, np.newaxis, i]@self.pre_attention[l])@u8[i]
                self.b_v[1, l] += np.sum(it(self.adjusted_patterns[l, :, np.newaxis, i])@u8[i], axis=1, keepdims=True)

                self.w_k[1, l] += (1/np.sqrt(self.d_k))*it(self.pre_attention[l])@m@s[l, :, i]@self.values[l]@it(u8[i])@self.queries[l, :, i, np.newaxis]
                self.b_k[1, l] += np.sum((1/np.sqrt(self.d_k))*m@s[l, :, i]@self.values[l]@it(u8[i])@self.queries[l, :, i, np.newaxis], axis=1, keepdims=True)
                
                self.w_q[1, l] += (1/np.sqrt(self.d_k))*it(self.pre_attention[l, :, i, np.newaxis])@u8[i]@it(self.values[l])@s[l, :, i]@m.T@self.keys[l]
                self.b_q[1, l] += (1/np.sqrt(self.d_k))*u8[i]@it(self.values[l])@s[l, :, i]@m.T@self.keys[l]
                
                for j in range(n):
                    inner = np.zeros((self.n_heads, n, self.d_embed))
                    if i == j:
                        inner[:, :, :] = self.keys[l]@it(self.w_q[0, l])
                    inner[:, j, np.newaxis] += self.queries[l, :, i, np.newaxis]@it(self.w_k[0, l])

                    inner = s[l, :, i]@m@inner
                    inner /= np.sqrt(self.d_k)

                    r = self.adjusted_patterns[l, :, i, np.newaxis, j, np.newaxis]*np.eye(self.d_embed)
                    self.through_attention[i, j] = u8[i]@(it(self.w_v[0, l])@(it(self.pre_attention[l])@inner + r) + it(self.b_v[0, l].repeat(self.n, axis=1))@inner)
                        
                    # self.through_attention[i, j] = u3[i]@it(self.w_v[0, l])@(self.adjusted_patterns[l, :, i, j, np.newaxis, np.newaxis]*np.eye(self.d_embed))
                    # # print(self.through_attention[i, j])
                    # if i == j:
                    #     w = self.pre_attention[l]
                    #     a = self.w_k[0, l]@it(self.w_q[0, l])
                    #     r = w[:, i, np.newaxis]@it(a)
                    #     x = w@a
                    #     x[:, i, np.newaxis] += r
                    #     x += self.b_k[0, l].repeat(n, axis=1)@it(self.w_q[0, l])
                    # else:
                    #     r = self.pre_attention[l, :, i, np.newaxis]@self.w_q[0, l]@it(self.w_k[0, l])
                    #     x = np.zeros((self.n_heads, n, self.d_embed))
                    #     x[:, j, np.newaxis] = r
                    #     # print(x.shape)
                    # x[:, j, np.newaxis] += self.b_q[0, l]@it(self.w_k[0, l])
                    # x /= np.sqrt(self.d_k)
                    # y = s[l, :, i, np.newaxis, j]@m.T
                    # z = y@x
                    # self.through_attention[i, j] += u3[i]@it(self.w_v[0, l])@z.repeat(self.d_embed, axis=1)
                    # self.through_attention[i, j] += u3[i]@it(self.b_v[0, l]).repeat(n, axis=2)@s[l, :, i]@m.T@x

            u5 = self.through_attention.sum(axis=(0, 2))

            for i in range(n):
                self.beta[1, 2*l] += u5[i]
                self.gamma[1, 2*l] += u5[i]*self.normed[2*l, i, np.newaxis]

            u6 = u5@self.d_layernorm(2*l)

            u0 = u6+u4
            check(u0, u1, u2, u3, u5, u6)
            check(self.through_attention)

        for i in range(n):
            self.w_embedding[1] += self.x[i, :, np.newaxis]@u0[i]

    def train(self, data: np.ndarray, runs: numbers.Integral) -> None:
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 0.000001
        rho = 0.001

        def nll():
            return -np.mean(np.log(np.sum(self.probs*data[0, 1:], axis=1)))

        def c(p, *args, **kwargs):
            check(p[0], p[1])
            epsilon = 0.01 + 0.001/(np.linalg.norm(p[1]))
            # print(epsilon)
            self.apply(data[0, :-1])
            x = nll()
            p[0] -= epsilon*p[1]
            self.apply(data[0, :-1])
            y = nll()
            # print(x, y)
            print(x-y, *args, **kwargs)  # should be positive
            p[0] += epsilon*p[1]
        """
        self.apply(data[0, :-1])
        print(self.probs)

        self.backprop(data[0])

        c(self.b_unembedding)
        c(self.w_unembedding)
        
        c(self.beta[:, -1])
        c(self.gamma[:, -1])
        
        c(self.b_down[:, -1])
        c(self.w_down[:, -1])
        c(self.b_up[:, -1])
        c(self.w_up[:, -1])

        c(self.beta[:, -2])
        c(self.gamma[:, -2])

        c(self.b_v[:, -1])
        c(self.w_v[:, -1])

        print()
        c(self.b_k[:, -1])
        c(self.w_k[:, -1])
        c(self.b_q[:, -1])
        c(self.w_q[:, -1])
        """
        # exit()

        for t in range(1, runs+1):
                
            self.backprop(data[0])

            if (t % (runs//10)) == 0:
                ...
                # print()
                # c(self.w_unembedding)
                # c(self.b_unembedding)
                # c(self.final_gamma)
                # c(self.final_beta)
                # for p in self.params:
                #     c(p)
                #     for i in range(p.shape[1]):
                #         c(p[:, i])
                #     print()
                # print()
                # for l in range(self.n_blocks):
                #     c(self.w_down[:, l])
                #     ...
            check(self.gamma[1, -2])
            for p in self.params:
                check(p[2])
                check(p[1])
                p[2] = beta_1*p[2] + (1-beta_1)*p[1]
                check(p[2])
                check(p[3])
                p[3] = beta_2*p[3] + (1-beta_2)*p[1]*p[1]
                check(p[3])
                p[0] -= rho*np.reciprocal(np.sqrt(p[3]/(1-beta_2**t))+epsilon)*p[2]/(1-beta_1**t)
                check(p[0])
                # p[0] -= rho*np.reciprocal(np.sqrt(p[3])+epsilon)*p[2]
                check(p[0])
                # p[0] -= rho*p[1]

            if (t % (runs//10)) == 0:
                # print()
                print(t)
                print(nll())

        self.apply(data[0, :-1])
        print(self.probs)
        print()
