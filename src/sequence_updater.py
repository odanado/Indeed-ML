from chainer import training

class SequenceUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, device):
        super(SequenceUpdater, self).__init__(
            train_iter, optimizer, None, device=device)

    def update_core(self):
        optimizer = self.get_optimizer('main')
        train_iter = self.get_iterator('main')

        model = optimizer.target

        X, y = zip(*next(train_iter))

        loss = model(X, y)

        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        optimizer.update()  # Update the parameters
