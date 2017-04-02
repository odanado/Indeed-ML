#!/usr/bin/env python

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


def main():
    y = np.array([[0, 1], [1, 0]], 'f')
    t = np.array([1, 0], 'i')
    precision, recall, fbeta, support = F.classification_summary(y, t)
    print(precision.data)
    print(recall.data)
    print(fbeta.data)
    print(support.data)
    # n_layers = 2
    # batch_size = 2
    # n_units = 100
    # n_vocab = 10
    # xs = [[0, 1], [2, 3, 4], [5], [6, 7, 8, 9]]
    # xs = [np.array(x) for x in  xs]
    # print(F.transpose_sequence(xs))
    # zero = np.zeros((n_layers, batch_size, n_units), 'f')
    # l = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
    # embed=L.EmbedID(n_vocab, n_units)


if __name__ == '__main__':
    main()
