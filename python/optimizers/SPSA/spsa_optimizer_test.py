# Copyright 2020 The TensorFlow Quantum Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test module for tfq.python.optimizers.spsa_minimizer optimizer."""
# Remove PYTHONPATH collisions for protobuf.
# pylint: disable=wrong-import-position
import sys
NEW_PATH = [x for x in sys.path if 'com_google_protobuf' not in x]
sys.path = NEW_PATH
# pylint: enable=wrong-import-position

from operator import mul
from functools import reduce
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
import spsa_minimizer


def loss_function_with_model_parameters(model, loss, train_x, train_y):
    """Create a new function that assign the model parameter to the model
    and evaluate its value.
    Args:
        model : an instance of `tf.keras.Model` or its subclasses.
        loss : a function with signature loss_value = loss(pred_y, true_y).
        train_x : the input part of training data.
        train_y : the output part of training data.
    Returns:
        A function that has a signature of:
            loss_value = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    count = 0
    sizes = []

    # Record the shape of each parameter
    for shape in shapes:
        n = reduce(mul, shape)
        sizes.append(n)
        count += n

    # Function accept the parameter and evaluate model
    @tf.function
    def func(params):
        """A function that can be used by tfq.optimizer.spsa_minimize.
        Args:
           params [in]: a 1D tf.Tensor.
        Returns:
            Loss function value
        """

        # update the parameters of the model
        start = 0
        for i, size in enumerate(sizes):
            model.trainable_variables[i].assign(
                tf.reshape(params[start:start + size], shape))
            start += size

        # evaluate the loss
        loss_value = loss(model(train_x, training=True), train_y)
        if loss_value.shape != ():
            loss_value = tf.cast(tf.math.reduce_mean(loss_value), tf.float32)
        return loss_value

    return func


class SPSAMinimizerTest(tf.test.TestCase, parameterized.TestCase):
    """Tests for the SPSA optimization algorithm."""

    def test_nonlinear_function_optimization(self):
        """Test to optimize a non-linear function.
        """
        func = lambda x: x[0]**2 + x[1]**2

        result = spsa_minimizer.minimize(func, tf.random.uniform(shape=[2]))
        self.assertAlmostEqual(func(result.position).numpy(), 0, delta=1e-4)
        self.assertTrue(result.converged)

    def test_quadratic_function_optimization(self):
        """Test to optimize a sum of quadratic function.
        """
        n = 2
        coefficient = tf.random.uniform(minval=0, maxval=1, shape=[n])
        func = lambda x: tf.math.reduce_sum(np.power(x, 2) * coefficient)

        result = spsa_minimizer.minimize(func, tf.random.uniform(shape=[n]))
        self.assertAlmostEqual(func(result.position).numpy(), 0, delta=2e-4)
        self.assertTrue(result.converged)

    def test_noisy_sin_function_optimization(self):
        """Test noisy ssinusoidal function
        """
        n = 10
        func = lambda x: tf.math.reduce_sum(
            tf.math.sin(x) + tf.random.uniform(
                minval=-0.1, maxval=0.1, shape=[n]))

        result = spsa_minimizer.minimize(func, tf.random.uniform(shape=[n]))
        self.assertLessEqual(func(result.position).numpy(), -n + 0.1 * n)

    def test_failure_optimization(self):
        """Test a function that is completely random and cannot be minimized
        """
        n = 100
        func = lambda x: np.random.uniform(-10, 10, 1)[0]
        it = 50

        result = spsa_minimizer.minimize(func,
                                         tf.random.uniform(shape=[n]),
                                         max_iterations=it)
        self.assertFalse(result.converged)
        self.assertEqual(result.num_iterations, it)

    def test_blocking(self):
        """Test the blocking functionality.
        """
        n = 10
        it = 50

        init = 1
        self.incr = 0

        def block_func1(params):
            self.incr += init
            return self.incr

        result = spsa_minimizer.minimize(block_func1,
                                         tf.random.uniform(shape=[n]),
                                         blocking=True,
                                         allowed_increase=0.5,
                                         max_iterations=it)
        self.assertFalse(result.converged)
        self.assertEqual(result.num_iterations, it)
        self.assertEqual(result.objective_value,
                         init * 4)  # function executd 3 (in step) +
        # 1 (initial evaluation) times

        init = 1 / 6 * 0.49
        self.incr = 0

        def block_func2(params):
            self.incr += init
            return self.incr

        result = spsa_minimizer.minimize(block_func2,
                                         tf.random.uniform(shape=[n]),
                                         blocking=True,
                                         allowed_increase=0.5,
                                         max_iterations=it)
        self.assertFalse(result.converged)
        self.assertEqual(result.num_iterations, it)
        self.assertEqual(result.objective_value, init * 3 * it + init)


if __name__ == "__main__":
    tf.test.main()


