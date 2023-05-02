# Copyright 2018 The TensorFlow Probability Authors.
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
# ============================================================================
"""Tests for the unconstrained L-BFGS optimizer."""

import functools

from absl.testing import parameterized
import numpy as np
from scipy.stats import special_ortho_group

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.optimizer import bfgs_utils
from tensorflow_probability.python.optimizer import lbfgs

# use float64 by default
tf.keras.backend.set_floatx("float64")
input_size = 20000
batch_size = 1000
#conda install spyder=5.5.3
x_train = np.linspace(-1, 1, input_size, dtype=np.float64)
y_train = np.sinc(10 * x_train)

# x_train = tf.expand_dims(tf.cast(x_train, tf.float32), axis=-1)
# y_train = tf.expand_dims(tf.cast(y_train, tf.float32), axis=-1)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(input_size)
train_dataset = train_dataset.batch(batch_size).cache()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(40, activation='tanh', input_shape=(1,)),  # Better results
    tf.keras.layers.Dense(1, activation='linear')])

def _make_val_and_grad_fn(value_fn):
  @functools.wraps(value_fn)
  def val_and_grad(x):
    return gradient.value_and_gradient(value_fn, x)
  return val_and_grad


def _norm(x):
  return np.linalg.norm(x, np.inf)


@test_util.test_all_tf_execution_regimes
class LBfgsTest(test_util.TestCase):
  """Tests for LBFGS optimization algorithm."""

  def test_quadratic_bowl_2d(self):
    """Can minimize a two dimensional quadratic function."""
    minimum = np.array([1.0, 1.0])
    scales = np.array([2.0, 3.0])

    @_make_val_and_grad_fn
    def quadratic(x):
      return tf.reduce_sum(scales * tf.math.squared_difference(x, minimum))

    start = tf.constant([0.6, 0.8])
    results = self.evaluate(
        lbfgs.minimize(model, initial_position=start, tolerance=1e-8))
    self.assertTrue(results.converged)

  
if __name__ == '__main__':
  test_util.main()