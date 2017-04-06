import os
import pickle

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer

from vocab import Vocabulary
from utils import encode_tag


def split_df(df, split_ratio=0.1, seed=42):
    # shuffle
    df = df.sample(frac=True, random_state=seed)
    size = int(len(df) * split_ratio)
    valid_df = df[:size]
    train_df = df[size:]

    return train_df, valid_df


class Dataset(object):

    def __init__(self, train_df, valid_df, labels, vocab_size=10000, use_cache=True):
        self.train_df = train_df
        self.valid_df = valid_df
        self.labels = labels
        self.vocab_size = vocab_size
        self.use_cache = use_cache

        self.train_data = self.train_df.to_dict('list')
        self.valid_data = self.valid_df.to_dict('list')

        self.vocab = None

        if use_cache:
            file_dir = os.path.dirname(os.path.realpath(__file__))
            self.cache_dir = os.path.join(
                file_dir, '..', '.dataset', 'indeed', 'vocab-{}'.format(vocab_size))
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)

            vocab_file = os.path.join(self.cache_dir, 'vocab.txt')
            if os.path.exists(vocab_file):
                self.vocab = Vocabulary.load(vocab_file)

        if self.vocab is None:
            sentences = '\n'.join(self.train_data['description'])
            self.vocab = Vocabulary.create(sentences, vocab_size)

        if use_cache:
            if not os.path.exists(vocab_file):
                self.vocab.save(vocab_file)

    def _load_cache(self, cache_file):
        if self.use_cache:
            cache_path = os.path.join(self.cache_dir, cache_file)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)

        return None

    def _save_cache(self, data, cache_file):
        if self.use_cache:
            cache_path = os.path.join(self.cache_dir, cache_file)
            if not os.path.exists(cache_path):
                with open(cache_path, 'wb') as f:
                    return pickle.dump(data, f)

    def _get_sequence(self, data):
        X = [self.vocab.sentence_to_ids(x) for x in data['description']]
        y = [encode_tag(tag, self.labels) for tag in data['tags']]

        return X, np.array(y)

    def _get_dataset(self, get_func, cache_file):
        ret = self._load_cache(cache_file)
        if ret:
            return ret

        X_train, y_train = get_func(self.train_data)
        X_valid, y_valid = get_func(self.valid_data)

        ret = (X_train, y_train, X_valid, y_valid)
        self._save_cache(ret, cache_file)
        return ret

    def _get_BoW(self, data, binary=True):
        X, y = self._get_sequence(data)
        row = []
        val = []
        col = []

        for i, x in enumerate(X):
            cnt = {}
            for v in x:
                if v in cnt:
                    if not binary:
                        cnt[v] += 1
                else:
                    cnt[v] = 1

            for k, v in cnt.items():
                col.append(k)
                val.append(v)
            row.extend([i] * len(cnt))

        shape = (len(X), len(self.vocab))
        X = csr_matrix((val, (row, col)), shape=shape)

        return X, y

    def _get_tfidf(self, data, transformer):
        X, y = self._get_BoW(data, binary=False)
        X = transformer.fit_transform(X)
        return X, y

    def get_sequence(self):
        return self._get_dataset(self._get_sequence, 'sequence.pkl')

    def get_BoW(self, binary=True):
        cache_file = 'BoW-{}.pkl'.format('binary' if binary else 'freq')
        return self._get_dataset(lambda data: self._get_BoW(data, binary), cache_file)

    def get_tfidf(self, **params):
        transformer = TfidfTransformer(**params)
        return self._get_dataset(lambda data: self._get_tfidf(data, transformer), 'tf-idf.pkl')


if __name__ == '__main__':
    import pandas as pd
    from time import time
    s = time()
    with open('./labels.txt') as f:
        labels = [line.rstrip() for line in f]
    train_df = pd.read_csv('./dataset/train.tsv', delimiter='\t').fillna('')
    train_df, valid_df = split_df(train_df)
    dataset = Dataset(train_df, valid_df, labels, vocab_size=10000)

    print(time() - s)
    X_train, y_train, X_valid, y_valid = dataset.get_tfidf()
    # print(X_train[0])
    print(time() - s)
