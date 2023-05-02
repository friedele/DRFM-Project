from collections import deque
from numpy import linalg as LA
from LineSearch import wolfe_step
import numpy as np
import copy
import time


def lbfgs_vect_matr_mult(v, H, alpha_0):
    if len(H):
        s, y = H.pop()
        v_i = v - (s.T @ v) / (y.T @ s) * y
        z = lbfgs_vect_matr_mult(v_i, H, alpha_0)
        return z + ((s.T @ v - y.T @ z) / (y.T @ s)) * s
    return alpha_0 * v


def lbfgs_find_direction(s_y_pairs, grad_k):
    s_prev, y_prev = s_y_pairs[-1]
    alpha_0 = y_prev.T @ s_prev / (y_prev.T @ y_prev)
    return lbfgs_vect_matr_mult(-grad_k, copy.deepcopy(s_y_pairs), alpha_0)


def lbfgs(M, e, max_iter, history_size):
    M.total_time = time.time()
    s_y_pairs = deque(history_size)
    w_cur = M.w_0
    H_cur = np.eye(w_cur.shape[0])
    func_cur, grad_cur = M.first_oracle(w_cur)
    func_0, grad_0 = func_cur, copy.deepcopy(grad_cur)

    d = -H_cur @ grad_cur
    step = wolfe_step(M, w_cur, d)
    if step is None:
        step = 1

    w_next = w_cur + step * d
    func_next, grad_next = M.first_oracle(w_next)
    s_y_pairs.append((w_next - w_cur, grad_next - grad_cur))

    M.r_k.append(LA.norm(grad_0) ** 2)

    while LA.norm(grad_cur) ** 2 / LA.norm(grad_0) ** 2 > e and M.iter_num < max_iter:
        w_cur, func_cur, grad_cur = w_next, func_next, grad_next
        d = lbfgs_find_direction(s_y_pairs, grad_cur)
        step = wolfe_step(M, w_cur, d)
        if step is None:
            step = 1
        w_next = w_cur + step * d

        func_next, grad_next = M.first_oracle(w_next)
        s_y_pairs.append((w_next - w_cur, grad_next - grad_cur))

        M.iter_num += 1
        M.r_k.append(LA.norm(grad_cur) ** 2)
    M.total_time = time.time() - M.total_time
    return w_cur

