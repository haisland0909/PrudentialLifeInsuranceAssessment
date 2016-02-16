from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa
import numpy as np


class OptimCutPoint(object):

    def __init__(self):
        self._cut_points = [x / 10.0 for x in range(15, 85, 10)]

    @staticmethod
    def _cut_qwk_score(cut_points, y_pred, y_true):
        """

        :param list cut_points:
        :param numpy.array y_true:
        :param numpy.array y_pred:
        :rtype: float
        """
        try:
            d_pred = np.digitize(y_pred[:, 0], cut_points) + 1
        except ValueError:

            return 1
        kappa = quadratic_weighted_kappa(y_true, d_pred)

        return -kappa

    def fit(self, y_pred, y_true):
        train_func = lambda x: self._cut_qwk_score(x, y_pred, y_true)
        self._cut_points = fmin_powell(train_func, self._cut_points)
        print(self._cut_points)

        return self

    def transform(self, y_pred):

        return np.digitize(y_pred[:, 0], self._cut_points) + 1
