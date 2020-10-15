import tensorflow as tf
import numpy as np

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import ops


class SFW(tf.keras.optimizers.Optimizer):

    def __init__(self, feasible_region, learning_rate=0.1, momentum=0.9, rescale='diameter', name='SFW', **kwargs):
        super().__init__(name, **kwargs)

        self.feasible_region = feasible_region

        self.rescale = rescale

        self._momentum = False
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("'momentum' must be between [0, 1].")

        self._set_hyper('momentum', kwargs.get('m', momentum))
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))

    def _resource_apply_dense(self, grad, var, apply_state):
        update_ops = []

        grad = ops.convert_to_tensor(grad, var.dtype.base_dtype)
        lr = math_ops.cast(self._get_hyper('learning_rate'), var.dtype.base_dtype)

        if self._momentum:
            m = math_ops.cast(self._get_hyper('momentum'), var.dtype.base_dtype)
            momentum_var = self.get_slot(var, 'momentum')
            modified_grad = momentum_var.assign(math_ops.add(m * momentum_var, (1 - m) * grad))
        else:
            modified_grad = grad

        v = ops.convert_to_tensor(self.feasible_region.lmo(modified_grad), var.dtype.base_dtype)
        vminvar = math_ops.subtract(v, var)

        if self.rescale is None:
            factormath_ops.cast(1. , var.dtype.base_dtype)
        elif self.rescale == 'diameter'
            factor = math_ops.cast(1. / self.feasible_region.get_diameter(var.shape), var.dtype.base_dtype)
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


class AdaSFW(tf.keras.optimizers.Optimizer):
    def __init__(self, feasible_region, learning_rate=0.01, inner_steps=2, delta=1e-8, name='AdaSFW', **kwargs):
        super().__init__(name, **kwargs)

        self.feasible_region = feasible_region

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

    def _resource_apply_dense(self, grad, var, apply_state):

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

        # executing all steps after the first in this loop
        for idx in range(self.K):

            #delta_q = grad + H * (y - var) / learning_rate
            delta_q = math_ops.add(grad, math_ops.multiply(H, math_ops.divide(math_ops.subtract(y, var), learning_rate)))

            #v = tf.convert_to_tensor(self.feasible_region.lmo(delta_q), var.dtype.base_dtype)
            v = ops.convert_to_tensor(self.feasible_region.lmo(delta_q), var.dtype.base_dtype)

            #vy_diff = v - y
            vy_diff = math_ops.subtract(v, y)

            #gamma_unclipped = -learning_rate * tf.tensordot(delta_q, vy_diff, len(var.shape)) / tf.tensordot(vy_diff, H * vy_diff, len(var.shape))
            gamma_unclipped = math_ops.divide(math_ops.reduce_sum( - learning_rate * math_ops.multiply(delta_q, vy_diff)),  math_ops.reduce_sum( math_ops.multiply(H, math_ops.square(vy_diff))))

            #gamma = tf.clip_by_value(t=gamma_unclipped, clip_value_min=0, clip_value_max=1)
            gamma = math_ops.ClipByValue(t=gamma_unclipped, clip_value_min=0, clip_value_max=1)

            y = state_ops.assign_add(y,  gamma * vy_diff)

        return state_ops.assign(var, y)

    def _resource_apply_sparse(self, grad, var, indices, apply_state):
        raise NotImplementedError("Sparse gradient updates are not supported.")

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


class AdamSFW(tf.keras.optimizers.Optimizer):
    def __init__(self, feasible_region, learning_rate=0.01, inner_steps=2, delta=1e-8, beta1=0.9, beta2=0.999, name='AdamSFW', **kwargs):
        super().__init__(name, **kwargs)

        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('delta', kwargs.get('delta', delta))
        self._set_hyper('beta1', kwargs.get('b1', beta1))
        self._set_hyper('beta2', kwargs.get('b2', beta2))
        self.K = kwargs.get('K', inner_steps)

        self.feasible_region = feasible_region

    def set_learning_rate(self, learning_rate):
        self._set_hyper('learning_rate', learning_rate)

    def _resource_apply_dense(self, grad, var, apply_state):
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

        # speeding up first step in the loop using the fact that y = var
        # v = ops.convert_to_tensor(self.feasible_region.lmo(m_accumulator), var.dtype.base_dtype)
        # vy_diff = math_ops.subtract(v, y)
        # gamma_unclipped = math_ops.divide(math_ops.reduce_sum( - learning_rate * math_ops.multiply(m_accumulator, vy_diff)),  math_ops.reduce_sum( math_ops.square(vy_diff)))
        # gamma = math_ops.ClipByValue(t=gamma_unclipped, clip_value_min=0, clip_value_max=1)
        # y = state_ops.assign_add(y,  gamma * vy_diff)

        # executing all steps after the first in this loop
        for idx in range(self.K):
            delta_q = math_ops.add(m_accumulator, math_ops.multiply(H, math_ops.divide(math_ops.subtract(y, var), learning_rate)))
            v = ops.convert_to_tensor(self.feasible_region.lmo(delta_q), var.dtype.base_dtype)
            vy_diff = math_ops.subtract(v, y)
            gamma_unclipped = math_ops.divide(math_ops.reduce_sum( - learning_rate * math_ops.multiply(delta_q, vy_diff)),  math_ops.reduce_sum( math_ops.multiply(H, math_ops.square(vy_diff))))
            gamma = math_ops.ClipByValue(t=gamma_unclipped, clip_value_min=0, clip_value_max=1)
            y = state_ops.assign_add(y,  gamma * vy_diff)

        return state_ops.assign(var, y)

    def _resource_apply_sparse(self, grad, var, indices, apply_state):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm_accumulator', init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype))#, initializer="zeros")
            self.add_slot(var, 'v_accumulator', init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype))#, initializer="zeros")
            self.add_slot(var, 'vhat_accumulator', init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype))#, initializer="zeros")
            self.add_slot(var, 'y', init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype))#, initializer="zeros")

    # def _prepare_local(self, var_device, var_dtype, apply_state):
    #     super()._prepare_local(var_device, var_dtype, apply_state)
    #     apply_state[(var_device, var_dtype)]['momentum'] = array_ops.identity(self._get_hyper('momentum', var_dtype))

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            learning_rate = self._serialize_hyperparameter('learning_rate'),
            delta         = self._serialize_hyperparameter('delta'),
            beta1         = self._serialize_hyperparameter('beta1'),
            beta2         = self._serialize_hyperparameter('beta2'),
        ))
        return config


class SGD(tf.keras.optimizers.Optimizer):

    def __init__(self, feasible_region=None, learning_rate=0.01, momentum=0.0, momentum_style='pytorch_convex', weight_decay=0.0, name='SGD', **kwargs):
        super().__init__(name, **kwargs)

        self.feasible_region = feasible_region

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

    def _resource_apply_dense(self, grad, var, apply_state):

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

        if self.feasible_region is not None:
            project_op, was_projected = self.feasible_region.project(var)
            var_update = tf.cond(was_projected, true_fn=lambda: state_ops.assign(var, project_op), false_fn=lambda: var)

        return var_update

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
