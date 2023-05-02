# Copyright 2023 (c) EG Friedel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

import collections
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
import numpy as np

# Results from SPSA
SPSAOptimizerResults = collections.namedtuple(
    'SPSAOptimizerResults',
    [
        'converged',
        # Scalar boolean tensor indicating whether the minimum
        # was found within tolerance.
        'num_iterations',
        # The number of iterations of the SPSA update.
        'num_objective_evaluations',
        # The total number of objective
        # evaluations performed.
        'position',
        # A tensor containing the last argument value found
        # during the search. If the search converged, then
        # this value is the argmin of the objective function.
        # A tensor containing the value of the objective from
        # previous iteration
        'objective_value_previous_iteration',
        # Save the evaluated value of the objective function
        # from the previous iteration
        'objective_value',
        # A tensor containing the value of the objective
        # function at the `position`. If the search
        # converged, then this is the (local) minimum of
        # the objective function.
        'tolerance',
        # Define the stop criteria. Iteration will stop when the
        # objective value difference between two iterations is
        # smaller than tolerance
        'lr',
        # Specifies the learning rate
        'alpha',
        # Specifies scaling of the learning rate
        'perturb',
        # Specifies the size of the perturbations
        'gamma',
        # Specifies scaling of the size of the perturbations
        'blocking',
        # If true, then the optimizer will only accept updates that improve
        # the objective function.
        'allowed_increase'
        # Specifies maximum allowable increase in objective function
        # (only applies if blocking is true).
    ])
# ==============================================================================


class MeanSquaredError(tf.keras.losses.MeanSquaredError):
    """Provides mean squared error metrics: loss / residuals.

    Use mean squared error for regression problems with one or more outputs.
    """

    def residuals(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return y_true - y_pred


class ReducedOutputsMeanSquaredError(tf.keras.losses.Loss):
    """Provides mean squared error metrics: loss / residuals.

    Consider using this reduced outputs mean squared error loss for regression
    problems with a large number of outputs or at least more then one output.
    This loss function reduces the number of outputs from N to 1, reducing both
    the size of the jacobian matrix and backpropagation complexity.
    Tensorflow, in fact, uses backward differentiation which computational
    complexity is  proportional to the number of outputs.
    """

    def __init__(self,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='reduced_outputs_mean_squared_error'):
        super(ReducedOutputsMeanSquaredError, self).__init__(
            reduction=reduction,
            name=name)

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        sq_diff = tf.math.squared_difference(y_true, y_pred)
        return tf.math.reduce_mean(sq_diff, axis=1)

    def residuals(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        sq_diff = tf.math.squared_difference(y_true, y_pred)
        eps = tf.keras.backend.epsilon()
        return tf.math.sqrt(eps + tf.math.reduce_mean(sq_diff, axis=1))


"""
    The gauss-newthon algorithm is obtained from the linear approximation of the
    squared residuals and it is used solve least square problems.
    A way to use cross-entropy instead of mean squared error is to compute
    residuals as the square root of the cross-entropy.
"""


class CategoricalCrossentropy(tf.keras.losses.CategoricalCrossentropy):
    """Provides cross-entropy metrics: loss / residuals.

    Use this cross-entropy loss for classification problems with two or more
    label classes. The labels are expected to be provided in a `one_hot`
    representation.
    """

    def residuals(self, y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        return tf.math.sqrt(eps + self.fn(y_true, y_pred, **self._fn_kwargs))


class SparseCategoricalCrossentropy(
        tf.keras.losses.SparseCategoricalCrossentropy):
    """Provides cross-entropy metrics: loss / residuals.

    Use this cross-entropy loss for classification problems with two or more
    label classes. The labels are expected to be provided as integers.
    """

    def residuals(self, y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        return tf.math.sqrt(eps + self.fn(y_true, y_pred, **self._fn_kwargs))


class BinaryCrossentropy(tf.keras.losses.BinaryCrossentropy):
    """Provides cross-entropy metrics: loss / residuals.

    Use this cross-entropy loss for classification problems with only two label
    classes (assumed to be 0 and 1). For each example, there should be a single
    floating-point value per prediction.
    """

    def residuals(self, y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        return tf.math.sqrt(eps + self.fn(y_true, y_pred, **self._fn_kwargs))


"""
    Other experimental losses for classification problems.
"""


class SquaredCategoricalCrossentropy(tf.keras.losses.Loss):
    """Provides squared cross-entropy metrics: loss / residuals.

    Use this cross-entropy loss for classification problems with two or more
    label classes. The labels are expected to be provided in a `one_hot`
    representation.
    """

    def __init__(self,
                 from_logits=False,
                 label_smoothing=0,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='squared_categorical_crossentropy'):
        super(SquaredCategoricalCrossentropy, self).__init__(
            reduction=reduction,
            name=name)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        return tf.math.square(tf.keras.losses.categorical_crossentropy(
            y_true,
            y_pred,
            self.from_logits,
            self.label_smoothing))

    def residuals(self, y_true, y_pred):
        return tf.keras.losses.categorical_crossentropy(
            y_true,
            y_pred,
            self.from_logits,
            self.label_smoothing)

    def get_config(self):
        config = {'from_logits': self.from_logits,
                  'label_smoothing': self.label_smoothing}
        base_config = super(SquaredCategoricalCrossentropy, self).get_config()
        return dict(base_config + config)


class CategoricalMeanSquaredError(tf.keras.losses.Loss):
    """Provides mean squared error metrics: loss / residuals.

    Use this categorical mean squared error loss for classification problems
    with two or more label classes. The labels are expected to be provided in a
    `one_hot` representation and the output activation to be softmax.
    """

    def __init__(self,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='categorical_mean_squared_error'):
        super(CategoricalMeanSquaredError, self).__init__(
            reduction=reduction,
            name=name)

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        # Selects the y_pred which corresponds to y_true equal to 1.
        prediction = tf.reduce_sum(tf.math.multiply(y_true, y_pred), axis=1)
        return tf.math.squared_difference(1.0, prediction)

    def residuals(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        # Selects the y_pred which corresponds to y_true equal to 1.
        prediction = tf.reduce_sum(tf.math.multiply(y_true, y_pred), axis=1)
        return 1.0 - prediction

# ==============================================================================


class DampingAlgorithm:
    """Default Levenberg–Marquardt damping algorithm.

    This is used inside the Trainer as a generic class. Many damping algorithms
    can be implemented using the same interface.
    """

    def __init__(self,
                 starting_value=1e-3,
                 dec_factor=0.1,
                 inc_factor=10.0,
                 min_value=1e-10,
                 max_value=1e+10,
                 adaptive_scaling=False,
                 fletcher=False):
        """Initializes `DampingAlgorithm` instance.

        Args:
          starting_value: (Optional) Used to initialize the Trainer internal
            damping_factor.
          dec_factor: (Optional) Used in the train_step decrease the
            damping_factor when new_loss < loss.
          inc_factor: (Optional) Used in the train_step increase the
            damping_factor when new_loss >= loss.
          min_value: (Optional) Used as a lower bound for the damping_factor.
            Higher values improve numerical stability in the resolution of the
            linear system, at the cost of slower convergence.
          max_value: (Optional) Used as an upper bound for the damping_factor,
            and as condition to stop the Training process.
          adaptive_scaling: Bool (Optional) Scales the damping_factor adaptively
            multiplying it with max(diagonal(JJ)).
          fletcher: Bool (Optional) Replace the identity matrix with
            diagonal of the gauss-newton hessian approximation, so that there is
            larger movement along the directions where the gradient is smaller.
            This avoids slow convergence in the direction of small gradient.
        """
        self.starting_value = starting_value
        self.dec_factor = dec_factor
        self.inc_factor = inc_factor
        self.min_value = min_value
        self.max_value = max_value
        self.adaptive_scaling = adaptive_scaling
        self.fletcher = fletcher

    def init_step(self, damping_factor, loss):
        return damping_factor

    def decrease(self, damping_factor, loss):
        return tf.math.maximum(
            damping_factor * self.dec_factor,
            self.min_value)

    def increase(self, damping_factor, loss):
        return tf.math.minimum(
            damping_factor * self.inc_factor,
            self.max_value)

    def stop_training(self, damping_factor, loss):
        return damping_factor >= self.max_value

    def apply(self, damping_factor, JJ):
        if self.fletcher:
            damping = tf.linalg.tensor_diag(tf.linalg.diag_part(JJ))
        else:
            damping = tf.eye(tf.shape(JJ)[0], dtype=JJ.dtype)

        scaler = 1.0
        if self.adaptive_scaling:
            scaler = tf.math.reduce_max(tf.linalg.diag_part(JJ))

        damping = tf.scalar_mul(scaler * damping_factor, damping)
        return tf.add(JJ, damping)

# ==============================================================================


class SPSA:
    """SPSA training algorithm.
    """
    # Init Method or Constructor
    def __init__(self,
                 model,
                 optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
                 loss = MeanSquaredError(),
                 converged = False,
                 num_iterations = 200,
                 num_objective_evaluations = 0,
                 position = 0,
                 objective_value = 0,
                 objective_value_previous_iteration = 0,
                 tolerance = 1e-5,
                 lr = 1,
                 alpha = 0,
                 perturb = 1.0,
                 gamma = 0.101,
                 blocking = False,
                 allowed_increase = 0.5,
                 seed = None,
                 attempts_per_step = 10,
                 experimental_use_pfor = True):                   
        """Initializes `Trainer` instance.

        Args:
          model: It is the Model to be trained, it is expected to inherit
            from tf.keras.Model and to be already built.
          optimizer: (Optional) Performs the update of the model trainable
            variables. When tf.keras.optimizers.SGD is used it is equivalent
            to the operation `w = w - learning_rate * updates`, where updates is
            the step computed using the Levenberg-Marquardt algorithm.
          loss: (Optional) An object which inherits from tf.keras.losses.Loss
          attempts_per_step: Integer (Optional) During the train step when new
            model variables are computed, the new loss is evaluated and compared
            with the old loss value. If new_loss < loss, then the new variables
            are accepted, otherwise the old variables are restored and
            new ones are computed using a different damping-factor.
            This argument represents the maximum number of attempts, after which
            the step is taken.
          experimental_use_pfor: (Optional) If true, vectorizes the jacobian
            computation. Else falls back to a sequential while_loop.
            Vectorization can sometimes fail or lead to excessive memory usage.
            This option can be used to disable vectorization in such cases.
        """
        if not model.built:
            raise ValueError('Trainer model has not yet been built. '
                             'Build the model first by calling `build()` or '
                             'calling `fit()` with some data, or specify an '
                             '`input_shape` argument in the first layer(s) for '
                             'automatic build.')

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.initial_position
        self.tolerance=tolerance
        self.num_iterations = num_iterations
        self.num_objective_evaluations = num_objective_evaluations
        self.position = position
        self.objective_value = objective_value
        self.objective_value_previous_iteration = objective_value_previous_iteration
        self.alpha = alpha
        self.lr = lr
        self.perturb = perturb
        self.gamma = gamma
        self.allowed_increase = allowed_increase

        # Used to backup and restore model variables.
        self._backup_variables = []

        # Since training updates are computed with shape (num_variables, 1),
        # self._splits and self._shapes are needed to split and reshape the
        # updates so that they can be applied to the model trainable_variables.
        self._splits = []
        self._shapes = []
        
        for variable in self.model.trainable_variables:
            variable_shape = tf.shape(variable)
            variable_size = tf.reduce_prod(variable_shape)
            backup_variable = tf.Variable(
                tf.zeros_like(variable),
                trainable=False)

            self._backup_variables.append(backup_variable)
            self._splits.append(variable_size)
            self._shapes.append(variable_shape)

        self._num_variables = tf.reduce_sum(self._splits).numpy().item()
        self._num_outputs = None

    @tf.function
    def _minimize_spsa(self, lr, num_iterations, alpha, gamma, loss):
        # The set of initialization variables are as follows:
            # theta
            # A
            # a
            # c
            # p
        theta = 1.0    
        A = lr*num_iterations
        a = 1.0  # Unity for now
        c = 1.0
        p = -1
        arraySize = 1000
        
        # The optimization will follow a negative gradient of the objective function
        # to find the function minimum. 
        for k in range(num_iterations):
            ak = a/(k+1.0+A)**alpha    # Scaled evaluation step size
            ck = c/(k+1.0)**gamma      # Step size along the negative gradient
            delta = 2*round(np.random.choice([p, 1], size=arraySize))  # Random pertubation vectorr 0.5 probablity draw between +-2
            theta_plus = loss(theta+ck*delta)
            theta_minus = loss(theta-ck*delta)
            est_gradient = (theta_plus-theta_minus)/(2*ck*delta) # Simultaneous perturbation approximation to the unknown gradient
            theta = theta-ak*est_gradient  # Update of the gradient estimation

        # if paired:
        #     # upper bound needs to be set to signed 32-bit integer
        #     # see https://github.com/numpy/numpy/issues/4085#issuecomment-29570567
        #     fkwargs['seed'] = np.random.randint(0, np.iinfo(np.int32).max)
        # if bounds is None:
        #     grad = (funcf(x + ck*delta, **fkwargs) - funcf(x - ck*delta, **fkwargs)) / (2*ck*delta)
        #     x -= ak*grad
        # else:
        #     # ensure evaluation points are feasible
        #     xplus = project(x + ck*delta)
        #     xminus = project(x - ck*delta)
        #     grad = (funcf(xplus, **fkwargs) - funcf(xminus, **fkwargs)) / (xplus-xminus)
        #     x = project(x - ak*grad)
        # print status updates every 100th iteration if disp=True
        # if disp and (k % max([1, num_iterations//100])) == 0:
        #     print(x)
        # if callback is not None:
        #     callback(x)



    def _train_step(self, inputs, targets,
                    init_gauss_newton, compute_gauss_newton):
        # J: jacobian matrix not used in the overdetermined case.
        # JJ: gauss-newton hessian approximation
        # rhs: gradient when overdetermined, residuals when underdetermined.
        # outputs: prediction of the model for the current inputs.
        J, JJ, rhs, outputs = init_gauss_newton(inputs, targets)

        # Perform normalization for numerical stability.
        batch_size = tf.shape(inputs)[0]
        normalization_factor = 1.0 / tf.dtypes.cast(
            batch_size,
            dtype=self.model.dtype)

        JJ *= normalization_factor
        rhs *= normalization_factor

        # Compute the current loss value.
        loss = self.loss(targets, outputs)

        stop_training = False
        attempt = 0

        attempts = tf.constant(self.attempts_per_step, dtype=tf.int32)

        while tf.constant(True, dtype=tf.bool):
            update_computed = False
            try:
                # Apply the damping to the gauss-newton hessian approximation.
                JJ_damped = self.damping_algorithm.apply(damping_factor, JJ)

                # Compute the updates:
                # overdetermined: updates = (J'*J + damping)^-1*J'*residuals
                # underdetermined: updates = J'*(J*J' + damping)^-1*residuals
                updates = compute_gauss_newton(J, JJ_damped, rhs)
            except Exception as e:
                del e
            else:
                if tf.reduce_all(tf.math.is_finite(updates)):
                    update_computed = True
                    # Split and Reshape the updates
                    updates = tf.split(tf.squeeze(
                        updates, axis=-1), self._splits)
                    updates = [tf.reshape(update, shape)
                               for update, shape in zip(updates, self._shapes)]

                    # Apply the updates to the model trainable_variables.
                    self.optimizer.apply_gradients(
                        zip(updates, self.model.trainable_variables))

            if attempt < attempts:
                attempt += 1

                if update_computed:
                    # Compute the new loss value.
                    outputs = self.model(inputs, training=False)
                    new_loss = self.loss(targets, outputs)

                    if new_loss < loss:
                        # Accept the new model variables and backup them.
                        loss = new_loss
                        damping_factor = self.damping_algorithm.decrease(
                            damping_factor, loss)
                        self.backup_variables()
                        break

                    # Restore the old variables and try a new damping_factor.
                    self.restore_variables()

                damping_factor = self.damping_algorithm.increase(
                    damping_factor, loss)

                stop_training = self.damping_algorithm.stop_training(
                    damping_factor, loss)
                if stop_training:
                    break
            else:
                break

        # Update the damping_factor which will be used in the next train_step.
        self.damping_factor.assign(damping_factor)
        return loss, outputs, attempt, stop_training

    def _compute_num_outputs(self, inputs, targets):
        input_shape = inputs.shape[1::]
        target_shape = targets.shape[1::]
        _inputs = tf.keras.Input(shape=input_shape,
                                 dtype=inputs.dtype)
        _targets = tf.keras.Input(shape=target_shape,
                                  dtype=targets.dtype)
        outputs = self.model(_inputs)
        residuals = self.loss.residuals(_targets, outputs)
        return tf.reduce_prod(residuals.shape[1::])

    def reset_damping_factor(self):
        self.damping_factor.assign(self.damping_algorithm.starting_value)

    def backup_variables(self):
        zip_args = (self.model.trainable_variables, self._backup_variables)
        for variable, backup in zip(*zip_args):
            backup.assign(variable)

    def restore_variables(self):
        zip_args = (self.model.trainable_variables, self._backup_variables)
        for variable, backup in zip(*zip_args):
            variable.assign(backup)

    def train_step(self, inputs, targets):
        if self._num_outputs is None:
            self._num_outputs = self._compute_num_outputs(inputs, targets)

        batch_size = tf.shape(inputs)[0]
        num_residuals = batch_size * self._num_outputs
        overdetermined = num_residuals >= self._num_variables

        if overdetermined:
            loss, outputs, attempts, stop_training = self._train_step(
                inputs,
                targets,
                self._init_gauss_newton_overdetermined,
                self._compute_gauss_newton_overdetermined)
        else:
            loss, outputs, attempts, stop_training = self._train_step(
                inputs,
                targets,
                self._init_gauss_newton_underdetermined,
                self._compute_gauss_newton_underdetermined)

        return loss, outputs, attempts, stop_training

    def fit(self, dataset, epochs=1, metrics=None):
        """Trains self.model on the dataset for a fixed number of epochs.

        Arguments:
            dataset: A `tf.data` dataset, must return a tuple (inputs, targets).
            epochs: Integer. Number of epochs to train the model.
            metrics: List of metrics to be evaluated during training.
        """
        self.backup_variables()
        steps = dataset.cardinality().numpy().item()
        stop_training = False

        if metrics is None:
            metrics = []

        pl = tf.keras.callbacks.ProgbarLogger(
            count_mode='steps',
            stateful_metrics=["damping_factor", "attempts"])

        pl.set_params(
            {"verbose": 1, "epochs": epochs, "steps": steps})

        pl.on_train_begin()

        for epoch in range(epochs):
            if stop_training:
                break

            # Reset metrics.
            for m in metrics:
                m.reset_states()

            pl.on_epoch_begin(epoch)

            iterator = iter(dataset)

            for step in range(steps):
                if stop_training:
                    break

                pl.on_train_batch_begin(step)

                data = next(iterator)

                data = data_adapter.expand_1d(data)
                inputs, targets, sample_weight = \
                    data_adapter.unpack_x_y_sample_weight(data)

                loss, outputs, attempts, stop_training = \
                    self.train_step(inputs, targets)

                # Update metrics.
                for m in metrics:
                    m.update_state(targets, outputs)

                logs = {"damping_factor": self.damping_factor,
                        "attempts": attempts,
                        "loss": loss}
                logs.update({m.name: m.result() for m in metrics})

                pl.on_train_batch_end(step, logs)

            pl.on_epoch_end(epoch)

        pl.on_train_end()

# ==============================================================================


class ModelWrapper(tf.keras.Sequential):
    """Wraps a keras model.

    When fit is called, the wrapped model is trained using SPSA.
    """

    def __init__(self, model):
        if not model.built:
            raise ValueError('This model has not yet been built. '
                             'Build the model first by calling `build()` or '
                             'calling `fit()` with some data, or specify an '
                             '`input_shape` argument in the first layer(s) for '
                             'automatic build.')

        super(ModelWrapper, self).__init__([model])
        self.model = model
        self.trainer = None

    def compile(self,
                optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
                loss=MeanSquaredError(),
                damping_algorithm=DampingAlgorithm(),
                attempts_per_step=10,
                solve_method='qr',
                jacobian_max_num_rows=100,
                experimental_use_pfor=True,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                **kwargs):

        super(ModelWrapper, self).compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=True)

        self.trainer = SPSA(
            model=self.model,
            optimizer=optimizer,
            loss=loss,
            damping_algorithm=damping_algorithm,
            attempts_per_step=attempts_per_step,
            solve_method=solve_method,
            jacobian_max_num_rows=jacobian_max_num_rows,
            experimental_use_pfor=experimental_use_pfor)

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        inputs, targets, sample_weight = \
            data_adapter.unpack_x_y_sample_weight(data)

        loss, outputs, attempts, stop_training = \
            self.trainer.train_step(inputs, targets)

        self.compiled_metrics.update_state(targets, outputs)

        logs = {"damping_factor": self.trainer.damping_factor,
                "attempts": attempts,
                "loss": loss}
        logs.update({m.name: m.result() for m in self.metrics})

        # BUG: In tensorflow v2.2.0 and v2.3.0 setting model.stop_training=True
        # does not stop training immediately, but only at the end of the epoch.
        # https://github.com/tensorflow/tensorflow/issues/41174
        self.stop_training = stop_training

        return logs

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            **kwargs):
        if verbose > 0:
            if callbacks is None:
                callbacks = []

            callbacks.append(tf.keras.callbacks.ProgbarLogger(
                count_mode='steps',
                stateful_metrics=["damping_factor", "attempts"]))

        return super(ModelWrapper, self).fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            **kwargs)
