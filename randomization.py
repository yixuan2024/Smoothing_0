import torch
from scipy.stats import norm, binom_test
import numpy as np
import math
from statsmodels.stats.proportion import proportion_confint


def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
    """
    :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
    :param num_classes:
    :param sigma: the noise level hyperparameter
    """
    self.base_classifier = base_classifier
    self.num_classes = num_classes
    self.sigma = sigma


def _randomization_noise(self, x: torch.tensor, num: int, batch_size):
    """ Sample the base classifier's prediction under noisy corruptions of the input x.

    :param x: the input [channel x width x height]
    :param num: number of samples to collect
    :param batch_size:
    :应该return什么呢？
    """
    with torch.no_grad():
        num_iterations = math.ceil(num / batch_size)
        for _ in range(num_iterations):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size

            batch = x.repeat((this_batch_size, 1, 1, 1))
            noise = torch.randn_like(batch, device='cuda') * self.sigma
            randomization = batch + noise