import numpy as np
import tensorflow as tf
# import torch
# import torch.nn as nn

p = -1
arraySize = 100
learning_rate = 1.0
num_iterations = 10
theta = 1.0
a = 1.0  # Unity for now
c = 1.0
alpha = 1.0
gamma = 0.101
A = 2.0
k = 1.1

class MeanSquaredError(tf.keras.losses.MeanSquaredError):
    """Provides mean squared error metrics: loss / residuals.

    Use mean squared error for regression problems with one or more outputs.
    """

    def residuals(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return y_true - y_pred

for k in range(10):
    ak = a/(k+1.0+A)**alpha    # Scaled evaluation step size
    ck = c/(k+1.0)**gamma      # Step size along the negative gradient
    delta = 2*np.round(np.random.choice([p, 1], size=arraySize))
    theta_plus = MeanSquaredError(theta+ck*delta)
    theta_minus = MeanSquaredError(theta-ck*delta)
# Simultaneous perturbation approximation to the unknown gradient
    est_gradient = (theta_plus-theta_minus)/(2*ck*delta)
    theta = theta-ak*est_gradient  # Update of the gradient estimation
