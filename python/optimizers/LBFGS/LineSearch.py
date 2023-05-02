from scipy import optimize
from numpy import linalg as LA
import numpy as np


def golden_section_search(func, a, b, e):
    x_a, x_b = a, b
    t = (np.sqrt(5) - 1) / 2
    L_0 = x_b - x_a
    L_i = L_0 * t
    x_l, x_r = x_b - L_i, x_a + L_i
    f_l, f_r = func(x_l), func(x_r)
    while x_b - x_a > e:
        L_j = L_i * t

        if f_l >= f_r:
            x_a = x_l
            x_l = x_r
            x_r = x_a + L_j
            f_l = f_r
            f_r = func(x_r)

        else:
            x_b = x_r
            x_r = x_l
            x_l = x_b - L_j
            f_r = f_l
            f_l = func(x_l)

        L_i = L_j

    return x_l if f_l < f_r else x_r


def gss_step(M, w, d, e):
    a, b = -1, 1
    return golden_section_search(M.one_dimension_null(w, d), a, b, e)


def brent_step(M, w, d, e):
    a, b = -1, 1
    func_to_min = M.one_dimension_null(w, d)
    return optimize.fminbound(func_to_min, a, b, xtol=e)


def armijo_step(M, w, d, *args):
    c = 0.01
    a, mu = 1, 2
    f_val_0, grad_val_0 = M.first_oracle(w)
    func_to_min = M.one_dimension_first(w, d)
    while func_to_min(a)[0] > f_val_0 + a * (-grad_val_0.T @ grad_val_0) * c:
        a /= mu
    return a


def wolfe_step(M, w, d, *args):
    func_to_min = M.one_dimension_null(w, d)
    c1, c2 = 1e-04, 0.9

    def func_to_min_grad(a):
        gradient = M.one_dimension_grad(w, d)
        return gradient(a).T @ (-gradient(0))

    step = optimize.linesearch.scalar_search_wolfe2(func_to_min, func_to_min_grad, c1=c1, c2=c2)[0]
    return step


def lipsitz_step(M, x, *args):
    L = 4
    beta = 2
    grad = M.first_oracle(x)[1]
    y = x - 1 / L * grad

    while M.null_oracle(y) > M.null_oracle(x) + M.first_oracle(x)[1].T @ (y - x) + L / 2 * LA.norm((y - x) ** 2):
        temp = y
        y = x - 1 / L * M.first_oracle(x)[1]
        x = temp
        L /= beta

    return 1 / L

