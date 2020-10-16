import tensorflow as tf
import numpy as np

import functools
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.training import training_ops
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.eager import context
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import gradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


def _filter_grads(grads_vars_and_constraints):
    """Filter out iterable with grad equal to None."""
    grads_vars_and_constraints = tuple(grads_vars_and_constraints)
    if not grads_vars_and_constraints:
        return grads_vars_and_constraints
    filtered = []
    vars_with_empty_grads = []
    for gvc in grads_vars_and_constraints:
        grad = gvc[0]
        if grad is None:
            vars_with_empty_grads.append(var)
        else:
            filtered.append(gvc)
    filtered = tuple(filtered)
    if not filtered:
        raise ValueError("No gradients provided for any variable: %s." %
                                         ([v.name for _, v in grads_vars_and_constraints],))
    if vars_with_empty_grads:
        logging.warning(
                ("Gradients do not exist for variables %s when minimizing the loss."),
                ([v.name for v in vars_with_empty_grads]))
    return filtered

class ConstrainedOptimizer(tf.keras.optimizers.Optimizer):

    def __init__(self, name='ConstrainedOptimizer', **kwargs):
        super().__init__(name, **kwargs)

    def _aggregate_gradients(self, grads_vars_and_constraints):
            """Returns all-reduced gradients.
            Args:
                grads_vars_and_constraints: List of (gradient, variable, constraint) pairs.
            Returns:
                A list of all-reduced gradients.
            """
            grads_and_vars = [(g,v) for g,v,_ in grads_vars_and_constraints]
            filtered_grads_and_vars = _filter_grads(grads_and_vars)
            def all_reduce_fn(distribution, grads_and_vars):
                return distribution.extended.batch_reduce_to(
                        ds_reduce_util.ReduceOp.SUM, grads_and_vars)

            if filtered_grads_and_vars:
                reduced = distribute_ctx.get_replica_context().merge_call(
                        all_reduce_fn, args=(filtered_grads_and_vars,))
            else:
                reduced = []
            reduced_with_nones = []
            reduced_pos = 0
            for g, _ in grads_and_vars:
                if g is None:
                    reduced_with_nones.append(None)
                else:
                    reduced_with_nones.append(reduced[reduced_pos])
                    reduced_pos += 1
            assert reduced_pos == len(reduced), "Failed to add all gradients"
            return reduced_with_nones

    def _distributed_apply(self, distribution, grads_vars_and_constraints, name, apply_state):
        """`apply_gradients` using a `DistributionStrategy`."""

        def apply_grad_to_update_var(var, grad, constraint):
            """Apply gradient to variable."""
            if isinstance(var, ops.Tensor):
                raise NotImplementedError("Trying to update a Tensor ", var)

            apply_kwargs = {}
            if isinstance(grad, ops.IndexedSlices):
                if var.constraint is not None:
                    raise RuntimeError(
                            "Cannot use a constraint function on a sparse variable.")
                if "apply_state" in self._sparse_apply_args:
                    apply_kwargs["apply_state"] = apply_state
                return self._resource_apply_sparse_duplicate_indices(
                        grad.values, var, grad.indices, **apply_kwargs)

            if "apply_state" in self._dense_apply_args:
                apply_kwargs["apply_state"] = apply_state
            return self._resource_apply_dense(grad, var, constraint, **apply_kwargs)

        eagerly_outside_functions = ops.executing_eagerly_outside_functions()
        update_ops = []
        with ops.name_scope(name or self._name, skip_on_eager=True):
            for grad, var, constraint in grads_vars_and_constraints:
                def _assume_mirrored(grad):
                    if isinstance(grad, ds_values.PerReplica):
                        return ds_values.Mirrored(grad.values)
                    return grad

                grad = nest.map_structure(_assume_mirrored, grad)
                with distribution.extended.colocate_vars_with(var):
                    with ops.name_scope("update" if eagerly_outside_functions else
                                                            "update_" + var.op.name, skip_on_eager=True):
                        update_ops.extend(distribution.extended.update(
                                var, apply_grad_to_update_var, args=(grad, constraint), group=False))

            any_symbolic = any(isinstance(i, ops.Operation) or
                                                 tf_utils.is_symbolic_tensor(i) for i in update_ops)
            if not context.executing_eagerly() or any_symbolic:
                with ops._get_graph_from_inputs(update_ops).as_default():
                    with ops.control_dependencies(update_ops):
                        return self._iterations.assign_add(1, read_value=False)

            return self._iterations.assign_add(1)

    def apply_gradients(self, grads_vars_and_constraints, name=None, experimental_aggregate_gradients=True):
        grads_vars_and_constraints = _filter_grads(grads_vars_and_constraints)
        var_list = [v for (_, v, _) in grads_vars_and_constraints]
        constraint_list = [c for (_, _, c) in grads_vars_and_constraints]

        with backend.name_scope(self._name):
            with ops.init_scope():
                self._create_all_weights(var_list)

            if not grads_vars_and_constraints:
                return control_flow_ops.no_op()

            if distribute_ctx.in_cross_replica_context():
                raise RuntimeError(
                        "`apply_gradients() cannot be called in cross-replica context. "
                        "Use `tf.distribute.Strategy.run` to enter replica "
                        "context.")

            strategy = distribute_ctx.get_strategy()
            if (not experimental_aggregate_gradients and strategy and isinstance(
                    strategy.extended,
                    parameter_server_strategy.ParameterServerStrategyExtended)):
                raise NotImplementedError(
                        "`experimental_aggregate_gradients=False is not supported for "
                        "ParameterServerStrategy and CentralStorageStrategy")

            apply_state = self._prepare(var_list)
            if experimental_aggregate_gradients:
                reduced_grads = self._aggregate_gradients(grads_vars_and_constraints)
                var_list = [v for _, v, _ in grads_vars_and_constraints]
                grads_vars_and_constraints = list(zip(reduced_grads, var_list, constraint_list))
            return distribute_ctx.get_replica_context().merge_call(
                    functools.partial(self._distributed_apply, apply_state=apply_state),
                    args=(grads_vars_and_constraints,),
                    kwargs={
                            "name": name,
                    })


class SFW(ConstrainedOptimizer):

    def __init__(self, learning_rate=0.1, momentum=0.9, rescale='diameter', name='SFW', **kwargs):
        super().__init__(name, **kwargs)

        self.rescale = rescale

        self._momentum = False
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("'momentum' must be between [0, 1].")

        self._set_hyper('momentum', kwargs.get('m', momentum))
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))

    def _resource_apply_dense(self, grad, var, constraint, apply_state):
        update_ops = []

        grad = ops.convert_to_tensor(grad, var.dtype.base_dtype)
        lr = math_ops.cast(self._get_hyper('learning_rate'), var.dtype.base_dtype)

        if self._momentum:
            m = math_ops.cast(self._get_hyper('momentum'), var.dtype.base_dtype)
            momentum_var = self.get_slot(var, 'momentum')
            modified_grad = momentum_var.assign(math_ops.add(m * momentum_var, (1 - m) * grad))
        else:
            modified_grad = grad

        v = ops.convert_to_tensor(constraint.lmo(modified_grad), var.dtype.base_dtype)
        vminvar = math_ops.subtract(v, var)

        if self.rescale is None:
            factormath_ops.cast(1. , var.dtype.base_dtype)
        elif self.rescale == 'diameter':
            factor = math_ops.cast(1. / constraint.get_diameter(), var.dtype.base_dtype)
        elif self.rescale == 'gradient':
            factor = math_ops.cast(tf.norm(modified_grad, ord=2) / tf.norm(vminvar, ord=2) , var.dtype.base_dtype)
        clipped_lr = math_ops.ClipByValue(t=lr*factor, clip_value_min=0, clip_value_max=1)

        update_ops.append( state_ops.assign_add(var, clipped_lr * vminvar) )

        return control_flow_ops.group(*update_ops)

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, 'momentum', initializer="zeros")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]['momentum'] = array_ops.identity(self._get_hyper('momentum', var_dtype))

    def get_config(self):
        config = super().get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'momentum': self._serialize_hyperparameter('momentum'),
        })
        return config


class AdaSFW(ConstrainedOptimizer):
    def __init__(self, learning_rate=0.01, inner_steps=2, delta=1e-8, name='AdaSFW', **kwargs):
        super().__init__(name, **kwargs)

        self.K = kwargs.get('K', inner_steps)

        self._momentum = False
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")

        self._set_hyper('momentum', kwargs.get('m', momentum))
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('delta', kwargs.get('delta', delta))

    def set_learning_rate(self, learning_rate):
        self._set_hyper('learning_rate', learning_rate)

    def _resource_apply_dense(self, grad, var, constraint, apply_state):

        if self._momentum:
            m = math_ops.cast(self._get_hyper('momentum'), var.dtype.base_dtype)
            momentum_var = self.get_slot(var, 'momentum')
            grad = momentum_var.assign(math_ops.add(m * momentum_var, (1 - m) * grad))

        else:
            grad = ops.convert_to_tensor(grad, var.dtype.base_dtype)

        learning_rate = math_ops.cast(self._get_hyper('learning_rate'), var.dtype.base_dtype)
        delta = math_ops.cast(self._get_hyper('delta'), var.dtype.base_dtype)
        accumulator = state_ops.assign_add(self.get_slot(var, "accumulator"), math_ops.square(grad))
        H = math_ops.add( delta, math_ops.sqrt( accumulator ) )
        y = state_ops.assign(self.get_slot(var, "y"), var)

        for idx in range(self.K):
            delta_q = math_ops.add(grad, math_ops.multiply(H, math_ops.divide(math_ops.subtract(y, var), learning_rate)))
            v = ops.convert_to_tensor(constraint.lmo(delta_q), var.dtype.base_dtype)
            vy_diff = math_ops.subtract(v, y)
            gamma_unclipped = math_ops.divide(math_ops.reduce_sum( - learning_rate * math_ops.multiply(delta_q, vy_diff)),  math_ops.reduce_sum( math_ops.multiply(H, math_ops.square(vy_diff))))
            gamma = math_ops.ClipByValue(t=gamma_unclipped, clip_value_min=0, clip_value_max=1)
            y = state_ops.assign_add(y,  gamma * vy_diff)

        return state_ops.assign(var, y)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'accumulator', init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype))#, initializer="zeros")
            self.add_slot(var, 'y', init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype))#, initializer="zeros")
        if self._momentum:
            for var in var_list:
                self.add_slot(var, 'momentum', initializer="zeros")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]['momentum'] = array_ops.identity(self._get_hyper('momentum', var_dtype))

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            learning_rate      = self._serialize_hyperparameter('learning_rate'),
            delta    = self._serialize_hyperparameter('delta'),
            momentum = self._serialize_hyperparameter('momentum'),
        ))
        return config


class AdamSFW(ConstrainedOptimizer):
    def __init__(self, learning_rate=0.01, inner_steps=2, delta=1e-8, beta1=0.9, beta2=0.999, name='AdamSFW', **kwargs):
        super().__init__(name, **kwargs)

        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('delta', kwargs.get('delta', delta))
        self._set_hyper('beta1', kwargs.get('b1', beta1))
        self._set_hyper('beta2', kwargs.get('b2', beta2))
        self.K = kwargs.get('K', inner_steps)

    def set_learning_rate(self, learning_rate):
        self._set_hyper('learning_rate', learning_rate)

    def _resource_apply_dense(self, grad, var, constraint, apply_state):
        grad = ops.convert_to_tensor(grad, var.dtype.base_dtype)

        b1 = math_ops.cast(self._get_hyper('beta1'), var.dtype.base_dtype)
        m_accumulator = self.get_slot(var, "m_accumulator")
        m_accumulator.assign(b1*m_accumulator + (1-b1)*grad)

        b2 = math_ops.cast(self._get_hyper('beta2'), var.dtype.base_dtype)
        v_accumulator = self.get_slot(var, "v_accumulator")
        v_accumulator.assign(b2*v_accumulator + (1-b2)*math_ops.square(grad))

        vhat_accumulator = self.get_slot(var, "vhat_accumulator")
        vhat_accumulator.assign(tf.math.maximum(vhat_accumulator, v_accumulator))

        delta = math_ops.cast(self._get_hyper('delta'), var.dtype.base_dtype)
        H = math_ops.add( delta, math_ops.sqrt( vhat_accumulator ) )

        learning_rate = math_ops.cast(self._get_hyper('learning_rate'), var.dtype.base_dtype)

        y = state_ops.assign(self.get_slot(var, "y"), var)

        for idx in range(self.K):
            delta_q = math_ops.add(m_accumulator, math_ops.multiply(H, math_ops.divide(math_ops.subtract(y, var), learning_rate)))
            v = ops.convert_to_tensor(constraint.lmo(delta_q), var.dtype.base_dtype)
            vy_diff = math_ops.subtract(v, y)
            gamma_unclipped = math_ops.divide(math_ops.reduce_sum( - learning_rate * math_ops.multiply(delta_q, vy_diff)),  math_ops.reduce_sum( math_ops.multiply(H, math_ops.square(vy_diff))))
            gamma = math_ops.ClipByValue(t=gamma_unclipped, clip_value_min=0, clip_value_max=1)
            y = state_ops.assign_add(y,  gamma * vy_diff)

        return state_ops.assign(var, y)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm_accumulator', init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype))#, initializer="zeros")
            self.add_slot(var, 'v_accumulator', init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype))#, initializer="zeros")
            self.add_slot(var, 'vhat_accumulator', init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype))#, initializer="zeros")
            self.add_slot(var, 'y', init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype))#, initializer="zeros")

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            learning_rate = self._serialize_hyperparameter('learning_rate'),
            delta         = self._serialize_hyperparameter('delta'),
            beta1         = self._serialize_hyperparameter('beta1'),
            beta2         = self._serialize_hyperparameter('beta2'),
        ))
        return config


class SGD(ConstrainedOptimizer):

    def __init__(self, learning_rate=0.01, momentum=0.0, momentum_style='pytorch_convex', weight_decay=0.0, name='SGD', **kwargs):
        super().__init__(name, **kwargs)

        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper("weight_decay", kwargs.get("wd", weight_decay))
        self._set_hyper('momentum', kwargs.get('m', momentum))

        self._weight_decay = False
        if isinstance(weight_decay, ops.Tensor) or callable(weight_decay) or weight_decay > 0:
            self._weight_decay = True

        self._momentum = False
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
            self._momentum_style = momentum_style or 'convex_pytorch'
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")

    def _resource_apply_dense(self, grad, var, constraint, apply_state):

        grad = ops.convert_to_tensor(grad, var.dtype.base_dtype)
        lr = math_ops.cast(self._get_hyper('learning_rate'), var.dtype.base_dtype)

        if self._weight_decay:
            wd = math_ops.cast(self._get_hyper('weight_decay'), var.dtype.base_dtype)
            if self._logging:
              self.get_slot(var, "log.weight_decay").assign_add(wd * lr)
              self.get_slot(var, "log.decoupled_weight_decay").assign_add(wd)
            modified_grad = tf.math.add(grad, wd*var)
        else:
            modified_grad = grad

        if self._momentum:
            m = math_ops.cast(self._get_hyper('momentum'), var.dtype.base_dtype)
            momentum_var = self.get_slot(var, 'momentum')

            if 'original' in self._momentum_style:
                if 'convex' in self._momentum_style:
                    momentum_var.assign(math_ops.add(m * momentum_var, (1-m) * lr * modified_grad))
                else:
                    momentum_var.assign(math_ops.add(m * momentum_var, lr * modified_grad))
                var_update = state_ops.assign_sub(var, momentum_var)
            elif 'pytorch' in self._momentum_style:
                if 'convex' in self._momentum_style:
                    momentum_var.assign(math_ops.add(m * momentum_var, (1-m) * modified_grad))
                else:
                    momentum_var.assign(math_ops.add(m * momentum_var, modified_grad))
                var_update = state_ops.assign_sub(var, lr * momentum_var)
            else:
                raise NotImplementedError(f"Unknown momentum style {self._momentum_style}")
        else:
            var_update = state_ops.assign_sub(var, lr * modified_grad)

        project_op, was_projected = constraint.euclidean_project(var)
        return tf.cond(was_projected, true_fn=lambda: state_ops.assign(var, project_op), false_fn=lambda: var_update)

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, 'momentum', initializer="zeros")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]['momentum'] = array_ops.identity(self._get_hyper('momentum', var_dtype))

    def get_config(self):
        config = super().get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            "weight_decay": self._serialize_hyperparameter("weight_decay"),
            'momentum': self._serialize_hyperparameter('momentum'),
        })
        return config


class Adam(ConstrainedOptimizer):

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               amsgrad=False,
               name='Adam',
               **kwargs):
    super(Adam, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self.epsilon = epsilon or backend_config.epsilon()
    self.amsgrad = amsgrad

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

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(Adam, self)._prepare_local(var_device, var_dtype, apply_state)

    local_step = math_ops.cast(self.iterations + 1, var_dtype)
    beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
    beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
    beta_1_power = math_ops.pow(beta_1_t, local_step)
    beta_2_power = math_ops.pow(beta_2_t, local_step)
    lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
          (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
    apply_state[(var_device, var_dtype)].update(
        dict(
            lr=lr,
            epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
            beta_1_t=beta_1_t,
            beta_1_power=beta_1_power,
            one_minus_beta_1_t=1 - beta_1_t,
            beta_2_t=beta_2_t,
            beta_2_power=beta_2_power,
            one_minus_beta_2_t=1 - beta_2_t))

  def set_weights(self, weights):
    params = self.weights
    # If the weights are generated by Keras V1 optimizer, it includes vhats
    # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
    # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
    num_vars = int((len(params) - 1) / 2)
    if len(weights) == 3 * num_vars + 1:
      weights = weights[:len(params)]
    super(Adam, self).set_weights(weights)

  def _resource_apply_dense(self, grad, var, constraint, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    if not self.amsgrad:
      var_update = training_ops.resource_apply_adam(
          var.handle,
          m.handle,
          v.handle,
          coefficients['beta_1_power'],
          coefficients['beta_2_power'],
          coefficients['lr_t'],
          coefficients['beta_1_t'],
          coefficients['beta_2_t'],
          coefficients['epsilon'],
          grad,
          use_locking=self._use_locking)
    else:
      vhat = self.get_slot(var, 'vhat')
      var_update = training_ops.resource_apply_adam_with_amsgrad(
          var.handle,
          m.handle,
          v.handle,
          vhat.handle,
          coefficients['beta_1_power'],
          coefficients['beta_2_power'],
          coefficients['lr_t'],
          coefficients['beta_1_t'],
          coefficients['beta_2_t'],
          coefficients['epsilon'],
          grad,
          use_locking=self._use_locking)

    project_op, was_projected = constraint.euclidean_project(var)
    return tf.cond(was_projected, true_fn=lambda: state_ops.assign(var, project_op), false_fn=lambda: var_update)

  def get_config(self):
    config = super(Adam, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._serialize_hyperparameter('decay'),
        'beta_1': self._serialize_hyperparameter('beta_1'),
        'beta_2': self._serialize_hyperparameter('beta_2'),
        'epsilon': self.epsilon,
        'amsgrad': self.amsgrad,
    })
    return config
