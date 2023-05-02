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

import numpy as np
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.util.tf_export import keras_export

"""
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
"""
# ==============================================================================
@keras_export('keras.optimizers.SPSA') 
# This is a subclass of the version 2 optimizer 
class SPSA(optimizer_v2.OptimizerV2):
    """SPSA training algorithm.
    """
    # Init Method or Constructor
    def __init__(self,
                 learning_rate=1.0,
                 num_iterations = 10,
                 theta = 1.0,    
                 a = 1.0,  # Unity for now
                 c = 1.0,
                 alpha = 0,
                 gamma = 0.101,
                 name = 'SPSA',
                 **kwargs
    ):                   
      
        """ Initializes SPSA optimizer object. Also inherits from the from 
        tensorflow keras OptimizerV2 base class. The super function sets the
        hyperparameters for the optimizer with "self" which represents the 
        instance of the class. By using “self” we can access the attributes and
        methods of the class in python. It also binds the attributes with the 
        given arguments.
        
        # Gradient Descent parameters
          learning_rate = 0.01
        
        # SPSA optimizer parameters
        self.a = tf.constant(a, dtype=tf.float32)
        self.c = tf.constant(c, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.gamma = tf.constant(gamma, dtype=tf.float32)
        """
        super(SPSA, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('theta', theta)
        self._set_hyper('a', a)
        self._set_hyper('c', c)
        self._set_hyper('alpha', alpha)
        self._set_hyper('gamma', gamma)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        """
        """
        super(SPSA, self)._prepare_local(var_device, var_dtype, apply_state)

        apply_state[(var_device, var_dtype)].update(
            {
                'alpha': array_ops.identity(self._get_hyper('alpha', var_dtype)),
                'gamma': array_ops.identity(self._get_hyper('gamma', var_dtype)),
                'a': array_ops.identity(self._get_hyper('a', var_dtype)),
                'c': array_ops.identity(self._get_hyper('c', var_dtype)),
                'theta': array_ops.identity(self._get_hyper('theta', var_dtype)),
                'learning_rate': array_ops.identity(self._get_hyper('learning_rate', var_dtype)),
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
        gamma = coefficients['gamma']
        a = coefficients['a']
        c = coefficients['c']
        num_iterations = 10
        A = learning_rate*num_iterations
        p = -1
        arraySize = 1000
        theta = 1

        # The optimization will follow a negative gradient of the objective function
        # to find the function minimum. 
        for k in range(num_iterations):
            ak = a/(k+1.0+A)**alpha    # Scaled evaluation step size
            ck = c/(k+1.0)**gamma      # Step size along the negative gradient
            delta = 2*np.round(np.random.choice([p, 1], size=arraySize))  # Random pertubation vector 0.5 probablity              draw between +-2
            # Expectation value functions.  How far do we perturb from the state position defined by (theta) 
            theta_plus = theta+ck*delta 
            theta_minus = theta-ck*delta
            est_gradient = (theta_plus-theta_minus)/(2*ck*delta) # Simultaneous perturbation approximation to the unknown gradient
            theta = theta-ak*est_gradient  # Update of the gradient estimation
  
        """ Update variables. A `Tensor` that will hold the new value of these 
            variables after the assignment has completed.
        """
        state_ops.assign(ak, ck, use_locking=self._use_locking)
        state_ops.assign(delta,  use_locking=self._use_locking)
        state_ops.assign(theta_plus, theta_minus, use_locking=self._use_locking)

        return  state_ops.assign(est_gradient, theta, use_locking=self._use_locking).op
        zip_args = (self.model.trainable_variables, self._backup_variables)
        for variable, backup in zip(*zip_args):
            variable.assign(backup)

    def _create_slots(self, var_list):
        """ Defines trainable TensorFlow variables.

        Args:
            var_list: list of `Variable` objects that will be minimized using this optimizer.

        """
        for var in var_list:
            self.add_slot(var, 'ak')
            self.add_slot(var, 'ck')
            self.add_slot(var, 'delta')
            self.add_slot(var, 'est_gradient')

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
        config = super(SPSA, self).get_config()

        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'alpha': self._serialize_hyperparameter('alpha'),
            'gamma': self._serialize_hyperparameter('gamma'),
            'a': self._serialize_hyperparameter("a"),
            'c': self._serialize_hyperparameter("c"),
        })

        return config
# ==============================================================================  