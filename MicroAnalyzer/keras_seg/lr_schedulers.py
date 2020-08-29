import matplotlib.pyplot as plt
import numpy as np


class LearningRateScheduler:

    def plot(self, num_epochs, title="Learning Rate Schedule"):
        # calculate all results up to epoch
        lrs = [self(i) for i in range(num_epochs)]

        # new figure
        fig = plt.figure()

        # set labels
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")

        # plot learning rates
        plt.plot(np.arange(1, num_epochs + 1), lrs)

        # show figure
        plt.show()

        return fig  # return figure for reuse

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class StepScheduler(LearningRateScheduler):
    def __init__(self, initial_lr: float, factor: float, step: int):
        self.initial_lr = initial_lr
        self.factor = factor
        self.step = step

    def __call__(self, epoch):
        # count the number of steps completed
        num_steps_completed = np.floor(epoch / self.step)

        # calculate next lr
        factor_at_epoch = self.factor ** num_steps_completed
        lr = self.initial_lr * factor_at_epoch

        return float(lr)
