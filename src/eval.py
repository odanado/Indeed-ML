#!/usr/bin/env python

import argparse
import os

import pandas as pd

import chainer
import chainer.functions as F
import cupy as xp

from chainer import config
from chainer import training
from tqdm import tqdm

from models import Encoder
from vocab import Vocabulary
from utils import load_config, load_labels, decode_tag

from sequence_updater import SequenceUpdater


def create_dataset(df, vocab):
    data = []
    for i, row in df.iterrows():
        description = row.description
        ids = vocab.sentence_to_ids(description)

        data.append(xp.array(ids, dtype=xp.int32))

    return data


def main(args):
    model_dir, _ = os.path.split(args.resume)

    load_config(os.path.join(model_dir, 'config.txt'))
    vocab = Vocabulary.load(os.path.join(model_dir, 'vocab.txt'))
    test_df = pd.read_csv('./dataset/test.tsv', delimiter='\t')
    labels = load_labels()

    test = create_dataset(test_df, vocab)

    model = Encoder(len(vocab), config.unit_size,
                    len(labels), config.dropout_ratio)

    if config.gpu >= 0:
        chainer.cuda.get_device(config.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    test_iter = chainer.iterators.SerialIterator(test, config.batch_size,
                                                 repeat=False, shuffle=False)
    updater = SequenceUpdater(test_iter, optimizer, device=config.gpu)
    trainer = training.Trainer(
        updater, (config.epoch, 'epoch'), out=model_dir)
    chainer.serializers.load_npz(args.resume, trainer)

    tags = []
    with chainer.using_config('train', False):
        for it in tqdm(test):
            y = model.encode([it])
            y = list(map(int, (F.sigmoid(y).data > 0.5)[0]))
            tags.append(decode_tag(y, labels))

    with open('tags.tsv', 'w') as f:
        f.write('tags\n')
        for tag in tags:
            f.write('{}\n'.format(tag))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('resume')
    args = parser.parse_args()
    main(args)
