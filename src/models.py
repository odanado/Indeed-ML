import numpy
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda


def argsort_list_descent(lst):
    return numpy.argsort([-len(x) for x in lst]).astype('i')


def permutate_list(lst, indices, inv):
    ret = [None] * len(lst)
    if inv:
        for i, ind in enumerate(indices):
            ret[ind] = lst[i]
    else:
        for i, ind in enumerate(indices):
            ret[i] = lst[ind]
    return ret


def score(t, y):
    xp = cuda.get_array_module(t)

    tp = xp.sum(t[t == y])
    support = xp.sum(t)
    relevant = xp.sum(y)

    precision = tp / relevant
    recall = tp / support

    if xp.isnan(precision):
        precision = 0
    if xp.isnan(recall):
        recall = 0
    f1 = 2 * precision * recall / (recall + precision)
    if xp.isnan(f1):
        f1 = 0

    return precision, recall, f1


class Encoder(chainer.Chain):

    def __init__(self, n_vocab, n_units, n_out, dropout_ratio=0.2):
        super(Encoder, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            forward=L.LSTM(n_units, n_units),
            linear=L.Linear(n_units, n_out)
        )
        self.dropout_ratio = dropout_ratio
        for param in self.params():
            param.data[...] = self.xp.random.uniform(
                -0.08, 0.08, param.data.shape)

    def reset_state(self):
        self.forward.reset_state()

    def __call__(self, *inputs):
        assert len(inputs) == 2
        x_list, t = inputs
        t = self.xp.asarray(t, 'i')

        self.y = self.encode(x_list)
        self.loss = F.sigmoid_cross_entropy(self.y, t)

        self.labels = self.xp.zeros_like(self.y)
        self.labels[self.y.data > 0.5] = 1
        _, _, self.f1 = score(t, self.labels)

        chainer.report({'f1': self.f1}, self)
        chainer.report({'loss': self.loss}, self)

        return self.loss

    def encode(self, x_list):
        indices = argsort_list_descent(x_list)
        x_list = permutate_list(x_list, indices, inv=False)

        self.reset_state()
        dropout_ratio = self.dropout_ratio
        x_list = F.transpose_sequence(x_list)
        h_list = [self.embed(x) for x in x_list]

        for h in h_list:
            self.forward(F.dropout(h, dropout_ratio))

        h = F.permutate(self.forward.h, indices, inv=True)
        y = self.linear(F.dropout(h, dropout_ratio))
        # y = permutate_list(y, indices, inv=True)
        return y
