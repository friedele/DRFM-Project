from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import backend as kb
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops


class WAME(Optimizer):
    """ Optimizer that implements the WAME optimization algorithm.

    References:
        Original paper:
            https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2017-50.pdf

        Pull request of optimizer code by paper author:
            https://github.com/keras-team/keras/pull/10899/commits/301c42ac50e53a789683c3aca90637e279198936

        Tensorflow base class:
            https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/optimizer_v2/optimizer_v2.py

        Tensorflow rmsprop implementation:
            https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/optimizer_v2/rmsprop.py

    """
    def __init__(
            self,
            learning_rate=0.001,
            alpha=0.9,
            eta_plus=1.2,
            eta_minus=0.1,
            zeta_min=0.01,
            zeta_max=100,
            name='WAME',
            **kwargs
    ):
        """ Initializes WAME optimizer object.

        Inherits from tensorflow keras OptimizerV2 base class.

        Args:
            learning_rate(float): Learning rate.
            alpha(float): Decay rate.
            eta_plus(float): Eta plus value.
            eta_minus(float): Eta minus value.
            zeta_min(float): Minimum per-weight acceleration factor.
            zeta_max(float): Maximum per-weight acceleration factor.
            name(str): Optional name prefix for operations created when applying gradients.
            **kwargs: Arbitrary keyword arguments.

        """
        super(WAME, self).__init__(name, **kwargs)

        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('alpha', alpha)
        self._set_hyper('eta_plus', eta_plus)
        self._set_hyper('eta_minus', eta_minus)
        self._set_hyper('zeta_min', zeta_min)
        self._set_hyper('zeta_max', zeta_max)
        self.epsilon = kb.epsilon()

    def _prepare_local(self, var_device, var_dtype, apply_state):
        """
        """
        super(WAME, self)._prepare_local(var_device, var_dtype, apply_state)

        apply_state[(var_device, var_dtype)].update(
            {
                'alpha': array_ops.identity(self._get_hyper('alpha', var_dtype)),
                'eta_plus': array_ops.identity(self._get_hyper('eta_plus', var_dtype)),
                'eta_minus': array_ops.identity(self._get_hyper('eta_minus', var_dtype)),
                'zeta_min': array_ops.identity(self._get_hyper('zeta_min', var_dtype)),
                'zeta_max': array_ops.identity(self._get_hyper('zeta_max', var_dtype)),
                'epsilon': ops.convert_to_tensor_v2(self.epsilon, var_dtype)
            }
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        """ Add ops to apply dense gradients to the variable `var`.

        Args:
            grad: A `Tensor` representing the gradient.
            var: A `Tensor` of data type `resource` which points to the variable to be updated.
            apply_state: A dict which is used across multiple apply calls.

        Returns:
            An `Operation` which updates the value of the variable.

        """
        var_device, var_dtype = var.device, var.dtype.base_dtype

        coefficients = (
                (apply_state or {}).get((var_device, var_dtype))
                or self._fallback_apply_state(var_device, var_dtype)
        )

        # Static Coefficients
        learning_rate = coefficients['lr_t']
        alpha = coefficients['alpha']
        eta_plus = coefficients['eta_plus']
        eta_minus = coefficients['eta_minus']
        zeta_min = coefficients['zeta_min']
        zeta_max = coefficients['zeta_max']
        epsilon = coefficients['epsilon']

        # Variables
        previous_gradient = self.get_slot(var, 'previous_gradient')
        z = self.get_slot(var, 'z')
        theta = self.get_slot(var, 'theta')
        zeta = self.get_slot(var, 'zeta')

        # Calculate weight delta
        zeta_t = kb.switch(
            math_ops.less(grad * previous_gradient, 0),
            theta * eta_plus,
            kb.switch(
                math_ops.greater(grad * previous_gradient, 0),
                theta * eta_minus,
                zeta
            )
        )
        zeta_t = kb.clip(zeta_t, zeta_min, zeta_max)
        z_t = (alpha * z) + ((1 - alpha) * zeta_t)
        theta_t = (alpha * theta) + ((1 - alpha) * math_ops.square(grad))
        w_delta = - (learning_rate * (z_t * grad * (1 / (theta_t + epsilon))))
        var_t = var + w_delta

        """ Update variables. A `Tensor` that will hold the new value of these 
            variables after the assignment has completed.
        """
        state_ops.assign(previous_gradient, grad, use_locking=self._use_locking)
        state_ops.assign(zeta, zeta_t, use_locking=self._use_locking)
        state_ops.assign(z, z_t, use_locking=self._use_locking)
        state_ops.assign(theta, theta_t, use_locking=self._use_locking)

        return state_ops.assign(var, var_t, use_locking=self._use_locking).op

    def _create_slots(self, var_list):
        """ Defines trainable TensorFlow variables.

        Args:
            var_list: list of `Variable` objects that will be minimized using this optimizer.

        """
        for var in var_list:
            self.add_slot(var, 'z')
            self.add_slot(var, 'theta')
            self.add_slot(var, 'previous_gradient')
            self.add_slot(var, 'zeta')

    def get_config(self):
        """ Returns the config of the optimizer.

        An optimizer config is a Python dictionary (serializable)
        containing the configuration of an optimizer.
        The same optimizer can be reinstantiated later
        (without any saved state) from this configuration.

        It is good practice to define the get_config and from_config
        methods when writing a custom model or layer class.
        This allows you to easily update the computation later if needed.

        Returns:
            dict: Model configuration dictionary.

        """
        config = super(WAME, self).get_config()

        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'alpha': self._serialize_hyperparameter('alpha'),
            'eta_plus': self._serialize_hyperparameter('eta_plus'),
            'eta_minus': self._serialize_hyperparameter("eta_minus"),
            'zeta_min': self._serialize_hyperparameter("zeta_min"),
            'zeta_max': self._serialize_hyperparameter("zeta_max"),
        })

        return config

