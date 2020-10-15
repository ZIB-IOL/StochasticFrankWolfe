import tensorflow as tf
import numpy as np
import math

tolerance = 1e-10


def get_avg_init_norm(shape, initializer='glorot_uniform', ord=2, repetitions=100):
    initializer = getattr(tf.keras.initializers, initializer)()
    return np.mean([tf.norm( initializer(shape), ord=2).numpy() for _ in range(repetitions)])

def convert_lp_radius(radius, n, in_ord=2, out_ord=np.inf):
    in_ord_rec = 0.5 if in_ord == 1 else 1.0/in_ord
    out_ord_rec = 0.5 if out_ord == 1 else 1.0/out_ord
    return radius * n**(out_ord_rec - in_ord_rec)

def get_lp_complementary_order(ord):
    if ord == np.inf: return 1
    elif ord == 1: return np.inf
    elif ord > 1: return 1.0 / (1 - 1.0/ord)
    else: raise NotImplementedError(f"Order {ord} not supported.")

def print_constraints(model, constraints):
    for var, constraint in zip(model.trainable_variables, constraints):
        print(f"variable {var._shared_name}")
        print(f"  shape is {var.shape}")
        print(f"  size is {constraint.n}")
        print(f"  constraint type is {type(constraint)}")
        try: print(f"  order is {constraint.p}")
        except: continue
        try: print(f"  radius is {constraint.get_radius()}")
        except: continue
        print(f"  diameter is {constraint.get_diameter()}")
        print("\n")

def create_unconstraints(model):
    constraints = []

    for var in model.trainable_variables:
        n = np.prod(var.shape)
        constraints.append(Unconstrained(n))

    return constraints

def create_lp_constraints(model, ord=2, value=300, mode='initialization', initializer='glorot_uniform'):
    constraints = []

    for var in model.trainable_variables:
        n = np.prod(var.shape)

        if mode=='radius':
            constraint = LpBall(n, ord=ord, diameter=None, radius=value)
        elif mode=='diameter':
            constraint = LpBall(n, ord=ord, diameter=value, radius=None)
        elif mode=='initialization':
            avg_norm = get_avg_init_norm(var.shape, initializer=initializer, ord=2)
            diameter = 2.0*value*avg_norm
            constraint = LpBall(n, ord=ord, diameter=diameter, radius=None)
        else:
            raise ValueError(f"Unknown mode {mode}")

        constraints.append(constraint)

    return constraints


class Constraint:

    def __init__(self, n):
        self.n = n
        pass

    def get_diameter(self):
        pass

    def lmo(self, x):
        assert np.prod(x.shape) == n, f"shape {x.shape} does not match dimension {n}"
        pass

    def shift_inside(self, x):
        assert np.prod(x.shape) == n, f"shape {x.shape} does not match dimension {n}"
        pass

    def euclidean_project(self, x):
        assert np.prod(x.shape) == n, f"shape {x.shape} does not match dimension {n}"
        pass


class Unconstrained(Constraint):

    def __init__(self, n):
        super().__init__(n)

    def get_diameter(self):
        return np.inf

    def lmo(self, x):
        super().__init__(x)
        raise ValueError("no lmo for unconstrained parameters")

    def shift_inside(self, x):
        super().__init__(x)
        return x

    def euclidean_project(self, x):
        super().__init__(x)
        return x, False


class LpBall(Constraint):

    def __init__(self, n, ord=2, diameter=None, radius=None):
        super().__init__(n)

        assert ord >= 1, f"Invalid order {ord}"

        self.p = np.inf if ord=='inf' else ord
        self.q = get_lp_complementary_order(self.p)

        if diameter is None and radius is None:
            raise ValueError("Neither diameter and radius given")
        elif diameter is None:
            self._radius = radius
            self._diameter = 2*convert_lp_radius(radius, self.n, in_ord=self.p, out_ord=2)
        elif radius is None:
            self._radius = convert_lp_radius(diameter/2.0, self.n, in_ord=2, out_ord=self.p)
            self._diameter = diameter
        else:
            raise ValueError("Both diameter and radius given")

    def get_diameter(self):
        return self._diameter

    def get_radius(self):
        return self._radius

    def lmo(self, x):
        super().__init__(x)
        if self.p == 1:
            max_index = tf.math.argmax( tf.abs( tf.keras.backend.flatten(x) ) )
            indices = tf.range(0, tf.size(x), dtype=max_index.dtype)
            boolean_mask = tf.reshape( tf.equal(indices, max_index), x.shape )
            return tf.where(boolean_mask, tf.cast(-self._radius * tf.sign(x), x.dtype), tf.zeros_like(x))
        elif self.p == 2:
            xnorm = tf.norm(x, ord=2)
            normalize_fn = lambda: -self._radius * x / xnorm
            zero_fn = lambda: tf.zeros_like(x)
            return tf.cond(xnorm > tolerance, normalize_fn, zero_fn)
        elif self.p == np.inf:
            random_part = self._radius * tf.cast( 2 * tf.random.uniform( x.shape, minval=0, maxval=2, dtype=tf.dtypes.int32 ) - 1, x.dtype)
            deterministic_part = -self._radius * tf.cast(tf.sign(x), x.dtype)
            return tf.where(tf.equal(x, 0), random_part, deterministic_part)
        else:
            sgnx = tf.where(tf.equal(x,0), tf.ones_like(x), tf.sign(x))
            absxqp = tf.math.pow( tf.abs(x), self.q/self.p )
            xnorm = tf.math.pow( tf.norm(x, ord=self.q), self.q/self.p )
            normalize_fn = lambda: tf.cast( -self._radius * sgnx * absxqp / xnorm, x.dtype)
            zero_fn = lambda: tf.zeros_like(x)
            return tf.cond(xnorm > tolerance, normalize_fn, zero_fn)

    def shift_inside(self, x):
        super().__init__(x)
        x_norm = np.linalg.norm(x.flatten(), ord=self.p)
        if x_norm > self._radius: return self._radius * x / x_norm
        return x

    def euclidean_project(self, x):
        super().__init__(x)
        if self.p == 1:
            def proj_x_fn():
                sorted = tf.sort( tf.math.abs( tf.reshape(x, [-1]) ), direction='DESCENDING')
                running_mean = (tf.math.cumsum(sorted, axis=0) - self._radius) / tf.range(1, tf.size(sorted)+1, dtype=sorted.dtype)
                is_less_or_equal = tf.math.less_equal(sorted, running_mean)
                idx = tf.size(is_less_or_equal) - tf.math.reduce_sum(tf.cast(is_less_or_equal, tf.int32)) - 1
                return (tf.sign(x) * tf.math.maximum(tf.math.abs(x) - tf.gather(running_mean, idx), tf.zeros_like(x)), True)
            x_fn = lambda: (x, False)
            return tf.cond(tf.norm(x, ord=1) > self._radius, proj_x_fn, x_fn)
        elif self.p == 2:
            x_norm = tf.norm(x, ord=2)
            proj_x_fn = lambda: (self._radius * x / x_norm, True)
            x_fn = lambda: (x, False)
            return tf.cond(x_norm > self._radius, proj_x_fn, x_fn)
        elif self.p == np.inf:
            x_norm = tf.norm(x, ord=np.inf)
            proj_x_fn = lambda: (tf.clip_by_value(x, -self._radius, self._radius), True)
            x_fn = lambda: (x, False)
            return tf.cond(x_norm > self._radius, proj_x_fn, x_fn)
        else:
            raise NotImplementedError(f"Projection not implemented for order {self.p}")
