#!/usr/bin/env python3

import numbers
import typing

import numpy as np

import numpy.typing as npt

from scipy.special import softmax
from scipy.stats import norm, truncnorm


def gelu(x: npt.ArrayLike) -> np.ndarray:
    return x*norm.cdf(x)


def gelu_prime(x: npt.ArrayLike) -> np.ndarray:
    return x*norm.pdf(x) + norm.cdf(x)


def it(x: np.ndarray) -> np.ndarray:
    # inner transpose
    assert len(x.shape) >= 2
    return x.swapaxes(-1, -2)


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
        self.training = False
        
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
        if self.training:
            return x*np.random.default_rng().choice([0, 1], size=x.shape, p=[1-p, p])
        else:
            return (1-p)*x

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

        from_mean = np.eye(self.d_embed) - 1/self.d_embed
        
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
        u7 = np.nan*np.ones((n, 1, self.d_stack))
        u8 = np.nan*np.ones((n, self.n_heads, 1, self.d_v))

        m = np.repeat([np.eye(n)], n, axis=0)

        for i in range(n):
            m[i, (i+1):, :] = 0

        s = -self.adjusted_patterns.reshape((self.n_blocks, self.n_heads, n, n, 1))@self.adjusted_patterns.reshape((self.n_blocks, self.n_heads, n, 1, n)) + self.adjusted_patterns.reshape((self.n_blocks, self.n_heads, n, 1, n))*np.eye(n)

        fancy_mask = np.zeros((n, self.n_heads, n, self.d_embed))
        for j in range(n):
            fancy_mask[j, :, j, :] = 1

        for l in range(self.n_blocks-1, -1, -1):
            self.b_down[1, l] = u0.sum(axis=0)
            self.w_down[1, l] = np.sum(self.post_gelu[l].reshape((n, self.d_mlp, 1))@u0, axis=0)

            u1 = u0@self.w_down[0, l].T*gelu_prime(self.pre_gelu[l]).reshape((n, 1, self.d_mlp))

            self.b_up[1, l] = u1.sum(axis=0)
            self.w_up[1, l] = np.sum(self.pre_mlp[l].reshape((n, self.d_embed, 1))@u1, axis=0)
                
            u2 = u1@self.w_up[0, l].T + u0

            self.beta[1, 2*l+1] = u2.sum(axis=0)
            self.gamma[1, 2*l+1] = np.sum(u2*self.normed[2*l+1, :, np.newaxis], axis=0)

            u4 = u2@self.d_layernorm(2*l+1)

            self.b_o[1, l] = u4.sum(axis=0)
            self.w_o[1, l] = np.sum(self.stack[l].reshape((n, self.d_stack, 1))@u4, axis=0)

            u7 = u4@self.w_o[0, l].T

            u8 = u7.reshape((n, self.n_heads, 1, self.d_v))
            
            self.w_v[1, l] = np.sum(it(self.pre_attention[l])@self.adjusted_patterns[l].reshape((self.n_heads, n, n, 1)).transpose(1, 0, 2, 3)@u8, axis=0)
            self.b_v[1, l] = np.sum(self.adjusted_patterns[l].reshape((self.n_heads, n, n, 1)).transpose(1, 0, 2, 3)@u8, axis=(0, 2), keepdims=False).reshape(self.b_v[1, l].shape)

            self.w_k[1, l] = np.sum((1/np.sqrt(self.d_k))*it(self.pre_attention[l])@np.broadcast_to(m, (self.n_heads, n, n, n)).swapaxes(0, 1)@s[l].swapaxes(0, 1)@self.values[l]@it(u8)@np.reshape(self.queries[l].swapaxes(0, 1), (n, self.n_heads, 1, self.d_k)), axis=0)
            self.b_k[1, l] = np.sum(1/np.sqrt(self.d_k)*np.broadcast_to(m, (self.n_heads, n, n, n)).swapaxes(0, 1)@(s[l]).swapaxes(0, 1)@self.values[l]@it(u8)@np.reshape(self.queries[l].swapaxes(0, 1), (n, self.n_heads, 1, self.d_k)), axis=(0, 2)).reshape((self.n_heads, 1, self.d_k))

            self.w_q[1, l] = np.sum(1/np.sqrt(self.d_k)*self.pre_attention[l].swapaxes(0, 1).reshape((n, self.n_heads, self.d_embed, 1))@u8@it(self.values[l])@s[l].swapaxes(0, 1)@np.broadcast_to(m, (self.n_heads, n, n, n)).swapaxes(0, 1)@self.keys[l], axis=0)
            self.b_q[1, l] = np.sum(1/np.sqrt(self.d_k)*u8@it(self.values[l])@s[l].swapaxes(0, 1)@np.broadcast_to(m, (self.n_heads, n, n, n)).swapaxes(0, 1)@self.keys[l], axis=0)

            inner_0 = np.zeros((n, n, self.n_heads, n, self.d_embed))

            inner_0[np.diag_indices(n)] += self.keys[l]@it(self.w_q[0, l])

            inner_0 += fancy_mask*np.broadcast_to(self.queries[l]@it(self.w_k[0, l]), (n, n, self.n_heads, n, self.d_embed)).swapaxes(0, 3)

            inner_1 = np.broadcast_to(m, (n, self.n_heads, n, n, n)).transpose(2, 0, 1, 3, 4)@inner_0
            inner = np.broadcast_to(s[l], (n, self.n_heads, n, n, n)).transpose(2, 0, 1, 3, 4)@inner_1
            inner /= np.sqrt(self.d_k)

            gr = self.adjusted_patterns[l].transpose(1, 2, 0).reshape((n, n, self.n_heads, 1, 1))*np.eye(self.d_embed)
            hr = np.broadcast_to(it(self.b_v[0, l]), (self.n_heads, self.d_v, n))
            self.through_attention = np.broadcast_to(u8, (n, n, self.n_heads, 1, self.d_v)).swapaxes(0, 1)@(it(self.w_v[0, l])@(it(self.pre_attention[l])@inner + gr) + hr@inner)

            check(self.through_attention)
            u5 = self.through_attention.sum(axis=(0, 2))
            check(u5)

            self.beta[1, 2*l] = u5.sum(axis=0)
            self.gamma[1, 2*l] = np.sum(u5*self.normed[2*l].reshape((n, 1, self.d_embed)), axis=0)

            u6 = u5@self.d_layernorm(2*l)
            check(u6)

            u0 = u6+u4
            check(u0)
            check(u1)
            check(u4)

        self.w_embedding[1] = np.sum(x[:, :, np.newaxis]@u0, axis=0)

    def train(self, data: np.ndarray, runs: numbers.Integral) -> None:
        self.training = True
        
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 0.000001
        rho = 0.001

        def nll():
            return -np.mean(np.log(np.sum(self.probs*data[0, 1:], axis=1)))            

        def c(p, *args, **kwargs):
            t = self.training
            self.training = False
            
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

            self.training = t
        # exit()

        for t in range(1, runs+1):
                
            self.backprop(data[0])

            if runs < 10 or t % (runs//10) == 0 or t % 10 == 0:
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

            if runs < 10 or (t % (runs//10)) == 0:
                # print()
                print(t)
                print(nll())

        self.training = False
                
        self.apply(data[0, :-1])
        print(self.probs)
        print()
