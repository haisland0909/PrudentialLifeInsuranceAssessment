from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa
import numpy as np


class OptimOffset(object):

    def __init__(self, all_fit=False):
        self._offset = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1, 0])
        self._all_fit = all_fit

    def _offset_qwk_score(self, offset):
        """

        :param numpy.array offset:
        :param numpy.array y_true:
        :param numpy.array y_pred:
        :rtype: float
        """
        offset_pred = self._apply_offset(self._data, offset)
        kappa = quadratic_weighted_kappa(self._data[:, 2], offset_pred)

        return -kappa

    @staticmethod
    def _apply_offset(data, offset):
        for j in range(9):
            flg = data[:, 0].astype(int) == j
            data[flg, 1] = data[flg, 0] + offset[j]
        offset_pred = np.clip(np.round(data[:, 1]), 1, 8)\
            .astype(int)

        return offset_pred

    def _score_offset(self, bin_offset, sv):
        flg = self._data[:, 0].astype(int) == sv
        self._data[flg, 1] = self._data[flg, 0] + bin_offset
        offset_pred = np.clip(np.round(self._data[:, 1]), 1, 8)\
            .astype(int)
        kappa = quadratic_weighted_kappa(self._data[:, 2], offset_pred)

        return -kappa

    def fit(self, y_pred, y_true):
        self._data = np.c_[y_pred, y_pred, y_true[None].T]
        if self._all_fit:
            self._offset = fmin_powell(self._offset_qwk_score, self._offset)
        else:
            for j in range(9):
                flg = self._data[:, 0].astype(int) == j
                self._data[flg, 1] = self._data[flg, 0] + self._offset[j]
            for j in range(9):
                train_func = lambda x: self._score_offset(x, j)
                self._offset[j] = fmin_powell(train_func, self._offset[j])
        print(self._offset)

        return self

    def transform(self, y_pred):
        data = np.c_[y_pred, y_pred]

        return self._apply_offset(data, self._offset)
