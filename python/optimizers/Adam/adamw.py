from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.ops import array_ops, control_flow_ops, math_ops, state_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras import backend as K
# from .utils import _init_weight_decays, _apply_weight_decays, _check_args
# from .utils import _update_t_cur_eta_t_v2, _apply_lr_multiplier, _set_autorestart
# from .utils import K_eval as KE


# def K_eval(x):
#     return KE(x, K)


@keras_export('keras.optimizers.AdamW')
class AdamW(OptimizerV2):
    """AdamW optimizer.
    Default parameters follow those provided in the original paper.
    For extended documentation, see optimizer_v2.Adam.__doc__.
    # Arguments
        model: keras.Model/tf.keras.Model. Pass as first positional argument
            to constructor (AdamW(model, ...)). If passed, automatically extracts
            weight penalties from layers and overrides `weight_decays`.
        zero_penalties: bool. If True and `model` is passed, will zero weight
            penalties (loss-based). (RECOMMENDED; see README "Use guidelines").
        learning_rate: A Tensor or a floating point value.  The learning rate.
        beta_1: A float value or a constant float tensor. The exponential decay
            rate for the 1st moment estimates.
        beta_2: A float value or a constant float tensor. The exponential decay
            rate for the 2nd moment estimates.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper.
        amsgrad: boolean. Whether to apply AMSGrad variant of this algorithm from
            the paper "On the Convergence of Adam and beyond".
        name: Optional name for the operations created when applying gradients.
            Defaults to "Adam".  @compatibility(eager) When eager execution is
            enabled, `learning_rate`, `beta_1`, `beta_2`, and `epsilon` can each
            be a callable that takes no arguments and returns the actual value
            to use. This can be useful for changing these values across different
            invocations of optimizer functions. @end_compatibility
        **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
            `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
            gradients by value, `decay` is included for backward compatibility to
            allow time inverse decay of learning rate. `lr` is included for
            backward compatibility, recommended to use `learning_rate` instead.
        model: keras.Model/tf.keras.Model/None. If not None, automatically
            extracts weight penalties from layers, and overrides `weight_decays`.
        zero_penalties: bool. If True and `model` is passed, will zero weight
            penalties (loss-based). (RECOMMENDED; see README "Use guidelines").
        total_iterations: int >= 0. Total expected iterations / weight updates
                          throughout training, used for normalization; <1>
        lr_multipliers:   dict / None. Name-value pairs specifying per-layer lr
                          multipliers, as {<layer name>:<multiplier value>}; <2>
        weight_decays:    dict / None. Name-value pairs specifying weight decays,
                          as {<weight matrix name>:<weight decay value>}; <2>
        use_cosine_annealing: bool. If True, multiplies lr each train iteration
                              as a function of eta_min, eta_max, total_iterations,
                              and t_cur (current); [2]-Appendix, 2
        autorestart: bool / None. If True, will automatically do Warm Restarts
                     by resetting `t_cur=0` after `total_iterations`. If None,
                     will default to same as `use_cosine_annealing`. If True
                     but `use_cosine_annealing` is False, will raise ValueError.
                     Note: once optimizer is built (happens on first model fit),
                     changing `autorestart` has no effect; optimizer needs to be
                     re-built.
        eta_min, eta_max: int, int. Min & max values of cosine annealing
                          lr multiplier; [2]-Appendix, 2
        t_cur: int. Value to initialize t_cur to - used for 'warm restarts'.
               To be used together with use_cosine_annealing==True
        total_iterations_wd: int / None. If not None, weight_decays will be
                     applied according to total_iterations_wd instead of
                     total_iterations, contrary to authors' scheme. Set to
                     sum(total_iterations) over all restarts to normalize over
                     all epochs. May yield improvement over `None`.
        init_verbose: bool. If True, print weight-name--weight-decay, and
                      lr-multiplier--layer-name value pairs set during
                      optimizer initialization (recommended)
    # <1> - if using 'warm restarts', then refers to total expected iterations
            for a given restart; can be an estimate, and training won't stop
            at iterations == total_iterations. [2]-Appendix, pg 1
    # <2> - [AdamW Keras Implementation - Github repository]
            (https://github.com/OverLordGoldDragon/keras_adamw)
    # References
        - [1][Adam - A Method for Stochastic Optimization]
             (http://arxiv.org/abs/1412.6980v8)
        - [2][Fixing Weight Decay Regularization in Adam]
             (https://arxiv.org/abs/1711.05101)
    """
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False,
                 model=None, zero_penalties=True,
                 total_iterations=0, total_iterations_wd=None,
                 use_cosine_annealing=False, lr_multipliers=None,
                 weight_decays=None, autorestart=None,
                 init_verbose=True, eta_min=0, eta_max=1, t_cur=0,
                 name="AdamW", **kwargs):
        # if total_iterations > 1:
        #     weight_decays = _init_weight_decays(model, zero_penalties,
        #                                         weight_decays)
        eta_t = kwargs.pop('eta_t', 1.)

        super(AdamW, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)

        self.eta_min = K.constant(eta_min, name='eta_min')
        self.eta_max = K.constant(eta_max, name='eta_max')
        self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
        self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')
        self.total_iterations = total_iterations
        self.total_iterations_wd = total_iterations_wd or total_iterations
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays or {}
        self.init_verbose = init_verbose
        self.use_cosine_annealing = use_cosine_annealing
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad

        # _set_autorestart(self, autorestart, use_cosine_annealing)
        # _check_args(self, total_iterations, use_cosine_annealing, weight_decays)
        self._init_lr = kwargs.get('lr', learning_rate)  # to print lr_mult setup
        self._updates_processed = 0  # to track num calls to '_resource_apply_...'
        self._init_notified = False

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'vhat')
        self._updates_per_iter = len(var_list)

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)

        lr_t = lr_t * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        # Learning rate multipliers
        # if self.lr_multipliers is not None:
        #     lr_t = _apply_lr_multiplier(self, lr_t, var)

        m_t = state_ops.assign(m,
                               beta_1_t * m + (1.0 - beta_1_t) * grad,
                               use_locking=self._use_locking)
        v_t = state_ops.assign(v,
                               beta_2_t * v + (1.0 - beta_2_t
                                               ) * math_ops.square(grad),
                               use_locking=self._use_locking)
        if self.amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = state_ops.assign(vhat,
                                      math_ops.maximum(vhat, v_t),
                                      use_locking=self._use_locking)
            var_delta = m_t / (math_ops.sqrt(vhat_t) + epsilon_t)
        else:
            var_delta = m_t / (math_ops.sqrt(v_t) + epsilon_t)
        var_t = math_ops.sub(var, self.eta_t * lr_t * var_delta)

        # Weight decays
        # if var.name in self.weight_decays.keys():
        #     var_t = _apply_weight_decays(self, var, var_t)

        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)

        # Cosine annealing
        # (iteration_done, t_cur_update, eta_t_update
        #  ) = _update_t_cur_eta_t_v2(self, lr_t, var)
        # if iteration_done and not self._init_notified:
        #     self._init_notified = True

        updates = [var_update, m_t, v_t]
        # if iteration_done:
        #     updates += [t_cur_update]
        # if self.use_cosine_annealing and iteration_done:
        #     updates += [eta_t_update]
        # if self.amsgrad:
        #     updates.append(vhat_t)
        return control_flow_ops.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)

        lr_t = lr_t * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        # Learning rate multipliers
        # if self.lr_multipliers is not None:
        #     lr_t = _apply_lr_multiplier(self, lr_t, var)

        m_scaled_g_values = grad * (1 - beta_1_t)
        m_t = state_ops.assign(m, m * beta_1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
        v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if self.amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = state_ops.assign(vhat,
                                      math_ops.maximum(vhat, v_t),
                                      use_locking=self._use_locking)
            var_delta = m_t / (math_ops.sqrt(vhat_t) + epsilon_t)
        else:
            var_delta = m_t / (math_ops.sqrt(v_t) + epsilon_t)
        var_t = math_ops.sub(var, self.eta_t * lr_t * var_delta)

        # Weight decays
        # if var.name in self.weight_decays.keys():
        #     var_t = _apply_weight_decays(self, var, var_t)

        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)

        # Cosine annealing
        # (iteration_done, t_cur_update, eta_t_update
        #  ) = _update_t_cur_eta_t_v2(self, lr_t, var)
        # if iteration_done and not self._init_notified:
        #     self._init_notified = True

        # updates = [var_update, m_t, v_t]
        # if iteration_done:
        #     updates += [t_cur_update]
        # if self.use_cosine_annealing and iteration_done:
        #     updates += [eta_t_update]
        # if self.amsgrad:
        #     updates.append(vhat_t)
        return control_flow_ops.group()

    def set_weights(self, weights):
        params = self.weights
        # If the weights are generated by Keras V1 optimizer, it includes vhats
        # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
        # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(AdamW, self).set_weights(weights)

    def get_config(self):
        config = super(AdamW, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'total_iterations': int(self.total_iterations),
            'weight_decays': self.weight_decays,
            'use_cosine_annealing': self.use_cosine_annealing,
            'autorestart': self.autorestart,
            # 't_cur': int(K_eval(self.t_cur)),
            # 'eta_t': float(K_eval(self.eta_t)),
            # 'eta_min': float(K_eval(self.eta_min)),
            # 'eta_max': float(K_eval(self.eta_max)),
            'init_verbose': self.init_verbose
        })
        return config


