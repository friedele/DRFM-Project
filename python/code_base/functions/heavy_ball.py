from math import sqrt
import numpy as np

from .algorithm import Algorithm


class HeavyBall(Algorithm):

    def __init__(self, parametrization="Constant"):
        super().__init__()

        self.name = "HB"

        if parametrization == "Constant":
            self.name += " with \nconstant tuning"
            self.set_parameters = lambda **params: self.set_constant_parameters(**params)
        elif parametrization == "Adaptive":
            self.name += " with \nPolyak step-size \nbased tuning"
            self.set_parameters = lambda **params: self.set_adaptive_parameters(**params)
        else:
            raise ValueError("\'parametrization\' must be either \'Constant\' or \'Adaptive\'. Got {}".format(
                parametrization
            ))

    @staticmethod
    def set_constant_parameters(function, **kwargs):

        step_size = 2 / (function.mu + function.L)
        momentum_size = ((sqrt(function.L) - sqrt(function.mu)) / (sqrt(function.L) + sqrt(function.mu))) ** 2

        return step_size, momentum_size

    @staticmethod
    def set_adaptive_parameters(function, x, x_pre, **kwargs):

        f_k = function.forward(x)
        f_km1 = function.forward(x_pre)
        f_star = function.min_value

        g_k = function.backward(x)
        g_km1 = function.backward(x_pre)

        step_size = 2 * (f_k - f_star) / np.sum(g_k ** 2)
        momentum_size = - ((f_k - f_star) * np.dot(g_k, g_km1)) / ((f_km1 - f_star) * np.dot(g_k, g_k) + (f_k - f_star) * np.dot(g_k, g_km1))

        return step_size, momentum_size

    def step(self, function, x, x_pre, first_step=False):

        g_x = function.backward(x)
        step_size, momentum_size = self.set_parameters(function=function, x=x, x_pre=x_pre)
        if first_step:
            momentum_size = 0

        x_new = x - (1 + momentum_size) * step_size * g_x + momentum_size * (x - x_pre)

        return x_new

    def run(self, function, x0, nb_steps):

        x_list = [x0]
        f_list = [function.forward(x0)]
        function.explored_min = f_list[-1]

        x_new = self.step(function=function, x=x0, x_pre=x0, first_step=True)
        x_list.append(x_new)
        f_list.append(function.forward(x_new))

        for _ in range(1, nb_steps):

            x_new = self.step(function=function, x=x_list[-1], x_pre=x_list[-2])
            x_list.append(x_new)
            f_list.append(function.forward(x_new))

        return x_list, f_list
