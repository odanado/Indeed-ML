#!/usr/bin/env python

import argparse
import os

import chainer

from chainer import config
from chainer import training
from chainer.training import extensions

import pandas as pd

from vocab import Vocabulary
from models import Encoder
from utils import load_config, save_config, create_vocab, load_labels, create_dataset
from sequence_updater import SequenceUpdater
from sequence_evaluator import SequenceEvaluator


def main(args):
    config_file = args.config
    if args.resume:
        com = os.path.commonpath([args.resume, args.model])
        if len(com) == 0:
            raise ValueError('{} and {} don\'t have commonpath'.format(
                args.resume, args.model))
        config_file = os.path.join(args.model, 'config.txt')
    else:
        if os.path.exists(args.model):
            raise ValueError('{} directory already exists'.format(args.model))
        os.makedirs(args.model)

    load_config(config_file)
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
    model = Encoder(len(vocab), config.unit_size,
                    len(labels), config.dropout_ratio)

    if config.gpu >= 0:
        chainer.cuda.get_device(config.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(config.grad_clip))

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
         'main/f1', 'validation/main/f1', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar(update_interval=1))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    trainer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--config', default='')
    parser.add_argument('--resume', default='')
    args = parser.parse_args()

    main(args)
