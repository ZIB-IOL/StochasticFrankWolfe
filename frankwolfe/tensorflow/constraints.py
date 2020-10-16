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
        try: print(f"  radius is {constraint.get_radius()}")
        except: pass
        print(f"  diameter is {constraint.get_diameter()}")
        try: print(f"  order is {constraint.p}")
        except: pass
        try: print(f"  K is {constraint.K}")
        except: pass
        print("\n")

def make_feasible(model, constraints):
    trainable_vars_to_constraints = dict()

    for var, constraint in zip(model.trainable_variables, constraints):
        trainable_vars_to_constraints[var._shared_name] = constraint

    complete_constraints = []

    for var in model.variables:
        complete_constraints.append( trainable_vars_to_constraints.get(var._shared_name, Unconstrained(tf.size(var).numpy())) )

    counter = 0
    for layer in model.layers:
        new_weights = []
        for w in layer.get_weights():
            new_weights.append( complete_constraints[counter].shift_inside(w) )
            counter += 1
        layer.set_weights(new_weights)

def create_unconstraints(model):
    constraints = []

    for var in model.trainable_variables:
        n = tf.size(var).numpy()
        constraints.append(Unconstrained(n))

    return constraints

def create_lp_constraints(model, ord=2, value=300, mode='initialization', initializer='glorot_uniform'):
    constraints = []

    for var in model.trainable_variables:
        n = tf.size(var).numpy()

        if mode=='radius':
            constraint = LpBall(n, ord=ord, diameter=None, radius=value)
        elif mode=='diameter':
            constraint = LpBall(n, ord=ord, diameter=value, radius=None)
        elif mode=='initialization':
            avg_norm = get_avg_init_norm(var.shape, initializer=initializer, ord=2)
            diameter = 2.0 * value * avg_norm
            constraint = LpBall(n, ord=ord, diameter=diameter, radius=None)
        else:
            raise ValueError(f"Unknown mode {mode}")

        constraints.append(constraint)

    return constraints

def create_k_sparse_constraints(model, K=1, K_frac=None, value=300, mode='initialization', initializer='glorot_uniform'):
    constraints = []

    for var in model.trainable_variables:
        n = tf.size(var).numpy()

        if K_frac is None and K is None:
            raise ValueError("Both K and K_frac are None")
        elif K_frac is None:
            real_K = min( int(K), n)
        elif K is None:
            real_K = min( mint(K_frac*n), n)
        else:
            real_K = min( max(int(K), int(K_frac*n)), n)

        if mode=='radius':
            constraint = KSparsePolytope(n, K=real_K, diameter=None, radius=value)
        elif mode=='diameter':
            constraint = KSparsePolytope(n, K=real_K, diameter=value, radius=None)
        elif mode=='initialization':
            avg_norm = get_avg_init_norm(var.shape, initializer=initializer, ord=2)
            diameter = 2.0 * value * avg_norm
            constraint = KSparsePolytope(n, K=real_K, diameter=diameter, radius=None)
        else:
            raise ValueError(f"Unknown mode {mode}")

        constraints.append(constraint)

    return constraints


class Constraint:

    def __init__(self, n):
        self.n = n

    def get_diameter(self):
        return self._diameter

    def get_radius(self):
        try: return self._radius
        except: raise ValueError("Tried to get radius from a constraint without one")

    def lmo(self, x):
        assert np.prod(x.shape) == self.n, f"shape {x.shape} does not match dimension {n}"

    def shift_inside(self, x):
        assert np.prod(x.shape) == self.n, f"shape {x.shape} does not match dimension {n}"

    def euclidean_project(self, x):
        assert np.prod(x.shape) == self.n, f"shape {x.shape} does not match dimension {n}"


class Unconstrained(Constraint):

    def __init__(self, n):
        super().__init__(n)
        self._diameter = np.inf

    def lmo(self, x):
        super().__init__(x)
        raise NotImplementedError("no lmo for unconstrained parameters")

    def shift_inside(self, x):
        super().__init__(x)
        return x

    def euclidean_project(self, x):
        super().__init__(x)
        return x, tf.constant(False, dtype=tf.bool)


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

    def lmo(self, x):
        super().lmo(x)
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
        super().shift_inside(x)
        x_norm = np.linalg.norm(x.flatten(), ord=self.p)
        if x_norm > self._radius: return self._radius * x / x_norm
        return x

    def euclidean_project(self, x):
        super().euclidean_project(x)
        if self.p == 1:
            def proj_x_fn():
                sorted = tf.sort( tf.math.abs( tf.reshape(x, [-1]) ), direction='DESCENDING')
                running_mean = (tf.math.cumsum(sorted, axis=0) - self._radius) / tf.range(1, tf.size(sorted)+1, dtype=sorted.dtype)
                is_less_or_equal = tf.math.less_equal(sorted, running_mean)
                idx = tf.size(is_less_or_equal) - tf.math.reduce_sum(tf.cast(is_less_or_equal, tf.int32)) - 1
                return (tf.sign(x) * tf.math.maximum(tf.math.abs(x) - tf.gather(running_mean, idx), tf.zeros_like(x)), tf.constant(True, dtype=tf.bool))
            x_fn = lambda: (x, tf.constant(False, dtype=tf.bool))
            return tf.cond(tf.norm(x, ord=1) > self._radius, proj_x_fn, x_fn)
        elif self.p == 2:
            x_norm = tf.norm(x, ord=2)
            proj_x_fn = lambda: (self._radius * x / x_norm, tf.constant(True, dtype=tf.bool))
            x_fn = lambda: (x, tf.constant(False, dtype=tf.bool))
            return tf.cond(x_norm > self._radius, proj_x_fn, x_fn)
        elif self.p == np.inf:
            x_norm = tf.norm(x, ord=np.inf)
            proj_x_fn = lambda: (tf.clip_by_value(x, -self._radius, self._radius), tf.constant(True, dtype=tf.bool))
            x_fn = lambda: (x, tf.constant(False, dtype=tf.bool))
            return tf.cond(x_norm > self._radius, proj_x_fn, x_fn)
        else:
            raise NotImplementedError(f"Projection not implemented for order {self.p}")


class KSparsePolytope(Constraint):

    def __init__(self, n, K=1, diameter=None, radius=None):
        super().__init__(n)

        self.K = min(K, n)

        if diameter is None and radius is None:
            raise ValueError("Neither diameter and radius given")
        elif diameter is None:
            self._radius = radius
            self._diameter = 2.0 * radius * math.sqrt(self.K)
        elif radius is None:
            self._radius = diameter / (2.0 * math.sqrt(self.K))
            self._diameter = diameter
        else:
            raise ValueError("Both diameter and radius given")

    def lmo(self, x):
        super().lmo(x)
        _, max_indices = tf.math.top_k(tf.abs(tf.keras.backend.flatten(x)), k=self.K)
        max_indices = tf.expand_dims(max_indices, -1)
        max_values = tf.gather_nd(tf.keras.backend.flatten(x), max_indices)
        return tf.reshape( tf.scatter_nd(max_indices, -self._radius * tf.sign(max_values), (self.n,)), x.shape)

    def shift_inside(self, x):
        super().shift_inside(x)
        l1norm =  np.linalg.norm(x.flatten(), ord=1)
        linfnorm = np.linalg.norm(x.flatten(), ord=np.inf)
        if l1norm > self._radius * self.K or linfnorm > self._radius:
            x_unit = x / max(l1norm, linfnorm)
            factor = min(math.floor(1.0 / np.linalg.norm(x_unit.flatten(), ord=np.inf)), self.K)
            assert 1 <= factor <= self.K
            return factor * self._radius * x_unit
        else:
            return x

    def euclidean_project(self, x):
        super().euclidean_project(x)
        raise NotImplementedError(f"Projection not implemented for K-sparse polytope")
