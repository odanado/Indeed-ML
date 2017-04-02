import cupy as xp
from chainer import config
from chainer import datasets

from vocab import Vocabulary


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
            else:
                try:
                    v = float(v)
                except:
                    pass

            setattr(config, k, v)


def encode_tag(tags, labels):
    ret = [0] * len(labels)
    if isinstance(tags, float):
        return ret

    for tag in tags.split(' '):
        ret[labels.index(tag)] = 1

    return ret


def decode_tag(encoded, labels):
    ret = []
    for i in range(len(labels)):
        if encoded[i] == 1:
            ret.append(labels[i])

    return ' '.join(ret[::-1])


def load_labels():
    with open('labels.txt') as f:
        labels = [line.rstrip() for line in f]
    return labels


def calc_valid_size(length, batch_size):
    ret = length // 10
    return ret + batch_size - ret % batch_size


def create_dataset(df, vocab, labels, seed=0, split=True):
    data = []
    for i, row in df.iterrows():
        description = row.description
        tags = row.tags
        ids = vocab.sentence_to_ids(description)
        encoded = encode_tag(tags, labels)

        data.append((xp.array(ids, dtype=xp.int32),
                     xp.array(encoded, xp.int32)))

    if split:
        test, train = datasets.split_dataset_random(
            data, calc_valid_size(len(data), config.batch_size), seed)

        return train, test

    return data
