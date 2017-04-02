import copy
import os
from chainer import config
from chainer.training import extensions
from chainer import reporter as reporter_module


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
                eval_func(X, y)
                pred = target.labels
                for i in range(len(X)):
                    results.append((y[i], X[i].tolist(), pred[i]))

            summary.add(observation)

        fname = os.path.join(
            config.model, 'result_{:03}.tsv'.format(self.epoch))
        with open(fname, 'w') as f:
            for ret in results:
                line = []
                line.append(' '.join(map(str, ret[0])))
                line.append(' '.join(map(str, ret[1])))
                line.append(' '.join(map(str, ret[2])))
                f.write('{}\n'.format('\t'.join(line)))
        self.epoch += 1
        return summary.compute_mean()
