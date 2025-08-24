#!/usr/bin/env python3

import numpy as np

from transformer import *

if __name__ == "__main__":
    # t = Transformer(n_vocab=1, d_k=0, d_mlp=0, n_blocks=0, n_heads=0, max_length=10_000, epsilon=0.01)
    # assert t.apply(np.random.random((1, 1))) == np.array([[1]])

    # t = Transformer(n_vocab=2, d_k=0, d_mlp=0, n_blocks=0, n_heads=0, max_length=10_000, epsilon=0.01)
    # assert np.all(t.apply(np.random.random((2, 2))) == 0.5*np.ones((2, 2)))

    # t = Transformer(n_vocab=2, d_k=0, d_mlp=0, n_blocks=1, n_heads=1, max_length=10_000, epsilon=0.01)
    # assert np.all(t.apply(np.random.random((2, 2))) == 0.5*np.ones((2, 2)))

    t = Transformer(n_vocab=2, d_k=3, d_mlp=4, n_blocks=5, n_heads=6, max_length=10_000, epsilon=0.01)
    assert t.apply(np.random.random((7, 2))).shape == (7, 2)

    t = Transformer(n_vocab=2, d_k=3, d_mlp=4, n_blocks=5, n_heads=6, max_length=10_000, epsilon=0.01)
    # t.backprop(np.random.random((8, 2)))

    t = Transformer(n_vocab=2, d_k=3, d_mlp=4, n_blocks=5, n_heads=6, max_length=10_000, epsilon=0.01)
    # t.train(np.array([[[0, 1], [0, 1]]]), 100)

    t = Transformer(n_vocab=2, d_k=10, d_mlp=11, n_blocks=12, n_heads=13, max_length=10_000, epsilon=0.01)
    # t.train(np.array([[[0, 1], [0, 1]]]), 100)

    t = Transformer(n_vocab=2, d_k=10, d_mlp=11, n_blocks=12, n_heads=13, max_length=10_000, epsilon=0.01)
    t.train(np.array([[[0, 1],
                       [1, 0],
                       [0, 1],
                       [1, 0],
                       [0, 1],
                       [1, 0],
                       [0, 1]]]), 100)

    # print(t.apply(np.array([[1, 0],
    #                         [0, 1],
    #                         [1, 0],
    #                         [0, 1],
    #                         [1, 0],
    #                         [0, 1]])))
