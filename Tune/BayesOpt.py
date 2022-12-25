import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from UtilFuncTrain import *
from MultiVarGaussian import *

class UtilityFunction(object):
    def __init__(self, kind, kappa, xi, kappa_decay=1, kappa_decay_delay=0, device="cuda"):
        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay
        self.xi = xi
        self._iters_counter = 0

        if kind not in ['ucb', 'ei', 'poi', "game"]:
            err = "The utility function {} has not been implemented, please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

        if kind == "game":
            self.game_util_func = torch.load("./data_and_models/UtilFunc.pt")

    def update_params(self):
        self._iters_counter += 1
        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)
        if self.kind == "game":
            return self._game(x, gp)

    @staticmethod
    def _ucb(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
            mean, std = mean.flatten(), std.diagonal()
        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
            mean, std = mean.flatten(), std.diagonal()
        a = (mean - y_max - xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
            mean, std = mean.flatten(), std.diagonal()
        z = (mean - y_max - xi)/std
        return norm.cdf(z)

    @staticmethod
    def _game(self, x, gp):
        mean, std = gp.predict(x, return_std=True)
        temp_max = abs(mean).max()
        mean, std = mean/temp_max, np.diag(std)/temp_max
        scores = [self.game_util_func.forward(torch.Tensor([mean[i], std[i]]).to(self.device).float().view(1,1,-1)).item() for i in range(len(mean))]
        return scores

class BayesianOptimizer:
    def __init__(self, _func, _dim, _lower_bounds, _upper_bounds, _y_max, _utilfunc, _viewlen, _nsamples):
        self.func = _func
        self.dim = _dim
        self.lower_bounds, self.upper_bounds = _lower_bounds, _upper_bounds
        self.y_max = _y_max
        self.utilfunc = _utilfunc
        self.gp = GPR()
        self.records = [] # records the collected points
        self.viewlen = _viewlen # number of points used for fitting. if 0, includes all.
        self.nsamples = _nsamples

    def probe(self, x):
        if self.func == None:
            print("The objective function is not defined in this case.")
        return self.func(x)

    def guess(self, x):
        return self.gp.predict(x)

    def tell(self, x, y):
        self.gp.fit(x, y)
        self.records.append([x, y])
        if (self.viewlen != 0) and (len(self.gp.train_X) > self.viewlen):
            train_X, train_y = self.gp.train_X[-self.viewlen:, :], self.gp.train_y[-self.viewlen:, :]
            self.gp = GPR()
            self.gp.fit(train_X, train_y)
            
        if y.item() > self.y_max: self.y_max = y.item()

    def suggest(self):
        sample_points = np.random.uniform(np.array(self.lower_bounds), np.array(self.upper_bounds), size=(self.nsamples, self.dim))
        util_values =  self.utilfunc.utility(sample_points, self.gp, self.y_max) # in shape (-1, )
        return sample_points[np.argsort(util_values)[0]].reshape(1, -1)

    def reinit(self):
        self.gp = GPR()
        self.records = []

class BOCollection:
    def __init__(self, _dim, _lower_bounds, _upper_bounds, _utilfunc, _viewlen, _nsamples, _bias_rate):
        self.lower_bounds, self.upper_bounds = _lower_bounds, _upper_bounds
        self.utilfunc = _utilfunc
        self.records = [] # records the collected points
        self.viewlen = _viewlen # number of points used for fitting. if 0, includes all.
        self.nsamples = _nsamples
        self.bias_rate = _bias_rate

        self.dim = _dim
        self.numBOs = self.dim // 32
        self.dimBOs = [32] * self.numBOs
        if self.dim % 32 != 0:
            self.numBOs += 1
            self.dimBOs.append(self.dim % 32)

        self.BOs = []
        index = 0
        for i in range(self.numBOs):
            self.BOs.append(BayesianOptimizer(_func=None, _dim=self.dimBOs[i],
                                              _lower_bounds=self.lower_bounds[index:index + self.dimBOs[i]],
                                              _upper_bounds=self.upper_bounds[index:index + self.dimBOs[i]],
                                              _y_max=1,
                                              _utilfunc=self.utilfunc, _viewlen=self.viewlen, _nsamples=self.nsamples))
            index = index + self.dimBOs[i]

    def tell(self, x, y):
        index = 0
        for i in range(self.numBOs):
            self.BOs[i].tell(x[index:index + self.dimBOs[i]].reshape(1,-1), y)
            index = index + self.dimBOs[i]
        self.records.append([x, y])

    def suggest(self):
        temp = []
        for i in range(self.numBOs):
            temp.append(self.BOs[i].suggest())
        return np.hstack(temp)

    def reinit(self):
        for i in range(self.numBOs):
            self.BOs[i].reinit()