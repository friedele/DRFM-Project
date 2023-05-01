from numpy import linalg as LA
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy import special, sparse
from scipy import special
import numpy as np


def dummy_var(x_0):
    n, m = x_0.shape
    x_1 = np.ones((n, 1))
    x = np.hstack((x_0, x_1))
    return x


def read_file(path):
    X, y = load_svmlight_file(path)
    m, n = np.size(X, 0), np.size(X, 1)
    ones = np.ones((X.shape[0], 1))
    X = sparse.hstack([X, ones])
    y = np.asarray(y)
    y = y.astype(int)
    labels = np.unique(y)
    y[y == labels[0]] = 0
    y[y == labels[1]] = 1
    y = np.reshape(y, (m, ))
    X = normalize(X, axis=0, norm='max')
    return X, y


class LogisticRegression:

    def __init__(self, x, y, seed):
        np.random.seed(seed)
        self.data = x
        self.labels = y
        self.oracle_calls = 0
        self.n = x.shape[0]
        self.m = x.shape[1]
        self.w_0 = np.random.normal(size=self.m)
        self.iter_num = 0
        self.total_time = []
        self.r_k = []

    def sigmoid(self, w_k, ind=None):
        if ind is None:
            ind = np.arange(self.n)
        f = self.data[ind] @ w_k
        return special.expit(f)

    def calc_func(self, w_k, ind=None):
        if ind is None:
            ind = np.arange(self.n)
        p = self.sigmoid(w_k, ind)
        return -1 / len(ind) * (
                self.labels[ind] @ np.nan_to_num(np.log(p)) + (1 - self.labels[ind]) @ np.nan_to_num(np.log(1 - p)))

    def calc_grad(self, w_k, ind=None):
        if ind is None:
            ind = np.arange(self.n)
        p = self.sigmoid(w_k, ind)
        return -1 / len(ind) * self.data[ind].T @ (self.labels[ind] - p)

    def calc_hes(self, w_k):
        p = self.sigmoid(w_k)
        return 1 / self.n * self.data.T @ np.diag(p * (1 - p)) @ self.data

    def null_oracle(self, w_k, ind=None):
        if ind is None:
            ind = np.arange(self.n)
        self.oracle_calls += 1
        return self.calc_func(w_k, ind)

    def first_oracle(self, w_k, ind=None):
        if ind is None:
            ind = np.arange(self.n)
        self.oracle_calls += 1
        return self.calc_func(w_k, ind), self.calc_grad(w_k, ind)

    def second_oracle(self, w_k):
        self.oracle_calls += 1
        return self.calc_func(w_k), self.calc_grad(w_k), self.calc_hes(w_k)

    def one_dimension_null(self, w_k, d_k):
        def inner(s):
            w_new = w_k + s * d_k
            return self.null_oracle(w_new)

        return inner

    def one_dimension_first(self, w_k, d_k):
        def inner(s):
            w_new = w_k + s * d_k
            return self.first_oracle(w_new)

        return inner

    def one_dimension_grad(self, w_k, d_k):
        def inner(s):
            w_new = w_k + s * d_k
            return self.calc_grad(w_new)

        return inner






