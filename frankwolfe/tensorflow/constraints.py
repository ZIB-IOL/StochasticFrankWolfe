import tensorflow as tf
import numpy as np
import math

tolerance = 1e-10


def get_avg_init_norm(layer_shape, initializer='glorot_uniform', ord=2, repetitions=10):
    output = 0
    for _ in range(repetitions):
        output += tf.norm( getattr(tf.keras.initializers, initializer)()(layer_shape), ord=2).numpy() / repetitions
    return output


class LpBall:

    def __init__(self, ord=2, value=100, mode='diameter', initializer='glorot_uniform'):

        self.p = np.inf if ord=='inf' else ord
        self.q = self._get_complementary_order(self.p)

        self.value = value
        self.mode = mode
        self.initializer = initializer

        self._diameter_dict = dict()
        self._radius_dict = dict()

    def _convert_radius(self, r, N, in_ord=2, out_ord=np.inf):
        in_ord_rec = 0.5 if in_ord == 1 else 1/in_ord
        out_ord_rec = 0.5 if out_ord == 1 else 1/out_ord
        return r * N**(out_ord_rec - in_ord_rec)

    def _get_complementary_order(self, ord):
        if ord == np.inf: return 1
        elif ord == 1: return np.inf
        elif ord >= 2: return 1 / (1 - 1/ord)
        else: raise NotImplementedError(f"Order {ord} not supported.")

    def _set_diameter_and_radius(self, layer_shape):
        if self.mode == 'diameter':
            self._diameter_dict[layer_shape] = self.value
            self._radius_dict[layer_shape] = self._convert_radius(self.value / 2.0, np.prod(layer_shape), in_ord=2, out_ord=self.p)
        elif self.mode == 'radius':
            self._diameter_dict[layer_shape] = 2.0 * self._convert_radius(self.value, np.prod(layer_shape), in_ord=self.p, out_ord=2)
            self._radius_dict[layer_shape] = self.value
        elif self.mode == 'initialization':
            avg_norm = get_avg_init_norm(layer_shape, initializer=self.initializer, ord=2)
            self._diameter_dict[layer_shape] = 2.0 * self.value * avg_norm
            self._radius_dict[layer_shape] = self._convert_radius(self.value * avg_norm, np.prod(layer_shape), in_ord=2, out_ord=self.p)
        else:
            raise ValueError("Neither diameter nor radius nor width is specified")

    def _get_radius(self, layer_shape):
        layer_shape = tuple(layer_shape)
        if not layer_shape in self._radius_dict:
            self._set_diameter_and_radius(layer_shape)
        return self._radius_dict[layer_shape]

    def _get_diameter(self, layer_shape):
        layer_shape = tuple(layer_shape)
        if not layer_shape in self._diameter_dict:
            self._set_diameter_and_radius(layer_shape)
        return self._diameter_dict[layer_shape]

    def oracle(self, x):
        r = self._get_radius(x.shape)
        if self.p == 1:
            max_index = tf.math.argmax( tf.abs( tf.keras.backend.flatten(x) ) )
            indices = tf.range(0, tf.size(x), dtype=max_index.dtype)
            boolean_mask = tf.reshape( tf.equal(indices, max_index), x.shape )
            return tf.where(boolean_mask, tf.cast(-r * tf.sign(x), x.dtype), tf.zeros_like(x))
        elif self.p == 2:
            xnorm = tf.norm(x, ord=2)
            normalize_fn = lambda: -r * x / xnorm
            zero_fn = lambda: tf.zeros_like(x)
            return tf.cond(xnorm > tolerance, normalize_fn, zero_fn)
        elif self.p == np.inf:
            random_part = r * tf.cast( 2 * tf.random.uniform( x.shape, minval=0, maxval=2, dtype=tf.dtypes.int32 ) - 1, x.dtype)
            deterministic_part = -r * tf.cast(tf.sign(x), x.dtype)
            return tf.where(tf.equal(x, 0), random_part, deterministic_part)
        else:
            sgnx = tf.where(tf.equal(x,0), tf.ones_like(x), tf.sign(x))
            absxqp = tf.math.pow( tf.abs(x), self.q/self.p )
            xnorm = tf.math.pow( tf.norm(x, ord=self.q), self.q/self.p )
            normalize_fn = lambda: tf.cast( -r * sgnx * absxqp / xnorm, x.dtype)
            zero_fn = lambda: tf.zeros_like(x)
            return tf.cond(xnorm > tolerance, normalize_fn, zero_fn)

    def shift_inside(self, x):
        r = self._get_radius(x.shape)
        x_norm = np.linalg.norm(x.flatten(), ord=self.p)
        if x_norm > r: return r * x / x_norm
        return x

    def euclidean_project(self, x):
        r = self._get_radius(x.shape)
        if self.p == 1:
            def proj_x_fn():
                sorted = tf.sort( tf.math.abs( tf.reshape(x, [-1]) ), direction='DESCENDING')
                running_mean = (tf.math.cumsum(sorted, axis=0) - r) / tf.range(1, tf.size(sorted)+1, dtype=sorted.dtype)
                is_less_or_equal = tf.math.less_equal(sorted, running_mean)
                idx = tf.size(is_less_or_equal) - tf.math.reduce_sum(tf.cast(is_less_or_equal, tf.int32)) - 1
                return (tf.sign(x) * tf.math.maximum(tf.math.abs(x) - tf.gather(running_mean, idx), tf.zeros_like(x)), True)
            x_fn = lambda: (x, False)
            return tf.cond(tf.norm(x, ord=1) > r, proj_x_fn, x_fn)
        elif self.p == 2:
            x_norm = tf.norm(x, ord=2)
            proj_x_fn = lambda: (r * x / x_norm, True)
            x_fn = lambda: (x, False)
            return tf.cond(x_norm > r, proj_x_fn, x_fn)
        elif self.p == np.inf:
            x_norm = tf.norm(x, ord=np.inf)
            proj_x_fn = lambda: (tf.clip_by_value(x, -r, r), True)
            x_fn = lambda: (x, False)
            return tf.cond(x_norm > r, proj_x_fn, x_fn)
        else:
            raise NotImplementedError(f"Projection not implemented for order {self.p}")
