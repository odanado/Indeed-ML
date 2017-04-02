#!/usr/bin/env python

import argparse
import copy
import os

import chainer
import chainer.links as L
import chainer.functions as F
import cupy as xp

from chainer import config
from chainer import datasets
from chainer import training
from chainer.training import extensions
from chainer import reporter as reporter_module

import pandas as pd

from vocab import Vocabulary
from models import Encoder


class SequenceUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, device):
        super(SequenceUpdater, self).__init__(
            train_iter, optimizer, None, device=device)

    def update_core(self):
        optimizer = self.get_optimizer('main')
        train_iter = self.get_iterator('main')

        model = optimizer.target

        X, y = zip(*next(train_iter))

        loss = model(X, xp.array(y))

        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        optimizer.update()  # Update the parameters


class SequenceEvaluator(extensions.Evaluator):

    def __init__(self, iterator, target,
                 device=None, eval_hook=None, eval_func=None):
        super(SequenceEvaluator, self).__init__(
            iterator, target, None, device, eval_hook, eval_func)
        self.epoch = 1

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func or target

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = reporter_module.DictSummary()
        results = []

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                X, y = zip(*batch)
                eval_func(X, xp.array(y))
                pred = F.argmax(target.y, axis=1)
                for i in range(len(X)):
                    results.append((y[i], X[i].tolist(), pred[i].data))

            summary.add(observation)

        fname = os.path.join(
            config.model, 'result_{:03}.tsv'.format(self.epoch))
        with open(fname, 'w') as f:
            for ret in results:
                line = []
                line.append(str(ret[0]))
                line.append(' '.join(map(str, ret[1])))
                line.append(str(ret[2]))
                f.write('{}\n'.format('\t'.join(line)))
        self.epoch += 1
        return summary.compute_mean()


def create_vocab(df):
    sentences = '\n'.join(df.loc[:, 'description'].values)
    return Vocabulary.create(sentences, config.vocab_size)


def save_config(fname):
    with open(fname, 'w') as f:
        config.show(f)


def load_config(fname):
    with open(fname) as f:
        for line in f:
            k, v = line.split()
            if v.isdigit():
                v = int(v)
            setattr(config, k, v)


def encode_tag(tags, labels):
    if isinstance(tags, float):
        return 0

    encoded = 0
    for tag in tags.split(' '):
        encoded += 2 ** labels.index(tag)

    return encoded


def decode_tag(encoded, labels):
    ret = []
    for i in range(len(labels)):
        if (encoded // 2 ** i) % 2 == 1:
            ret.append(labels[i])

    return ' '.join(ret[::-1])


def load_labels():
    with open('labels.txt') as f:
        labels = [line.rstrip() for line in f]
    return labels


def calc_valid_size(length, batch_size):
    ret = length // 10
    return ret + batch_size - ret % batch_size


def create_dataset(df, vocab, labels):
    data = []
    for i, row in df.iterrows():
        description = row.description
        tags = row.tags
        ids = vocab.sentence_to_ids(description)
        encoded = encode_tag(tags, labels)

        data.append((xp.array(ids, dtype=xp.int32), xp.int32(encoded)))

    test, train = datasets.split_dataset_random(
        data, calc_valid_size(len(data), config.batch_size), 0)

    return train, test


def main(args):
    if os.path.exists(args.model):
        raise ValueError('{} directory already exists'.format(args.model))

    os.makedirs(args.model)
    load_config(args.config)
    print('show config')
    config.show()
    save_config('{}/config.txt'.format(args.model))
    for k, v in args._get_kwargs():
        setattr(config, k, v)

    train_df = pd.read_csv('./dataset/train.tsv', delimiter='\t')

    vocab_file = os.path.join(args.model, 'vocab.txt')
    if os.path.exists(vocab_file):
        vocab = Vocabulary.load(vocab_file)
    else:
        vocab = create_vocab(train_df)
        vocab.save(vocab_file)

    labels = load_labels()

    train, test = create_dataset(train_df, vocab, labels)
    train = train
    model = L.Classifier(
        Encoder(len(vocab), config.unit_size, 2 ** len(labels)))

    if config.gpu >= 0:
        chainer.cuda.get_device(config.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))

    train_iter = chainer.iterators.SerialIterator(train, config.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, config.batch_size,
                                                 repeat=False, shuffle=False)
    updater = SequenceUpdater(train_iter, optimizer, device=config.gpu)
    trainer = training.Trainer(
        updater, (config.epoch, 'epoch'), out=args.model)

    trainer.extend(SequenceEvaluator(test_iter, model, device=config.gpu))
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('model')
    args = parser.parse_args()

    main(args)
