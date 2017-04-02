import numpy
import chainer
import chainer.functions as F
import chainer.links as L


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


class Encoder(chainer.Chain):

    def __init__(self, n_vocab, n_units, n_out, dropout_ratio=0.2):
        super(Encoder, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            forward=L.LSTM(n_units, n_units),
            linear=L.Linear(n_units, n_out)
        )
        self.dropout_ratio = dropout_ratio
        for param in self.params():
            param.data[...] = self.xp.random.uniform(-0.08, 0.08, param.data.shape)

    def reset_state(self):
        self.forward.reset_state()

    def __call__(self, x_list):
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
