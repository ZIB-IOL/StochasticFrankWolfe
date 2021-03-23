# ===========================================================================
# Project:      StochasticFrankWolfe 2020 / IOL Lab @ ZIB
# File:         pytorch/constraints.py
# Description:  Contains LMO-oracle classes for Pytorch
# ===========================================================================
import torch
import torch.nn.functional as F
import math

tolerance = 1e-10


# Auxiliary methods
@torch.no_grad()
def get_avg_init_norm(layer, param_type=None, ord=2, repetitions=100):
    """Computes the average norm of default layer initialization"""
    output = 0
    for _ in range(repetitions):
        layer.reset_parameters()
        output += torch.norm(getattr(layer, param_type), p=ord).item()
    return float(output) / repetitions


def convert_lp_radius(r, N, in_ord=2, out_ord='inf'):
    """
    Convert between radius of Lp balls such that the ball of order out_order
    has the same L2 diameter as the ball with radius r of order in_order
    in N dimensions
    """
    # Convert 'inf' to float('inf') if necessary
    in_ord, out_ord = float(in_ord), float(out_ord)
    in_ord_rec = 0.5 if in_ord == 1 else 1.0 / in_ord
    out_ord_rec = 0.5 if out_ord == 1 else 1.0 / out_ord
    return r * N ** (out_ord_rec - in_ord_rec)


def get_lp_complementary_order(ord):
    """Get the complementary order"""
    ord = float(ord)
    if ord == float('inf'):
        return 1
    elif ord == 1:
        return float('inf')
    elif ord > 1:
        return 1.0 / (1.0 - 1.0 / ord)
    else:
        raise NotImplementedError(f"Order {ord} not supported.")


def print_constraints(model):
    for idx, (name, param) in enumerate(model.named_parameters()):
        if hasattr(param, 'constraint'):
            constraint = param.constraint
        else:
            print(f"No constraint found for variable {name}.")
            continue
        print(f"variable {name}")
        print(f"  shape is {param.shape}")
        print(f"  size is {constraint.n}")
        print(f"  constraint type is {type(constraint)}")
        try:
            print(f"  radius is {constraint.get_radius()}")
        except:
            pass
        print(f"  diameter is {constraint.get_diameter()}")
        try:
            print(f"  order is {constraint.p}")
        except:
            pass
        try:
            print(f"  K is {constraint.K}")
        except:
            pass
        print("\n")


# Method to ensure initial feasibility of the parameters of a model
@torch.no_grad()
def make_feasible(model, global_constraint=None):
    """Shift all model parameters inside the feasible region defined by constraints"""
    if not global_constraint:
        for idx, (name, param) in enumerate(model.named_parameters()):
            if hasattr(param, 'constraint'):
                param.copy_(param.constraint.shift_inside(param))
    else:
        param_list = [p for name, p in model.named_parameters()]
        shifted_vector = global_constraint.shift_inside(torch.cat([p.view(-1) for p in param_list]))
        for p in param_list:
            numberOfElements = p.numel()
            p.copy_(shifted_vector[:numberOfElements].view(p.shape))
            shifted_vector = shifted_vector[numberOfElements:]


# Methods for getting global constraints
@torch.no_grad()
def get_global_lp_constraint(model, ord=2, value=300, mode='initialization'):
    """Create 1 L_p constraint for entire model, where p == ord, and value depends on mode (is radius, diameter, or
    factor to multiply average initialization norm with)"""
    n = 0
    for layer in model.modules():
        for param_type in [entry for entry in ['weight', 'bias'] if
                           (hasattr(layer, entry) and type(getattr(layer, entry)) != type(None))]:
            param = getattr(layer, param_type)
            n += int(param.numel())

    if mode == 'radius':
        constraint = LpBall(n, ord=ord, diameter=None, radius=value)
    elif mode == 'diameter':
        constraint = LpBall(n, ord=ord, diameter=value, radius=None)
    elif mode == 'initialization':
        cum_avg_norm = 0.0
        for layer in model.modules():
            if hasattr(layer, 'reset_parameters'):
                for param_type in [entry for entry in ['weight', 'bias'] if
                                   (hasattr(layer, entry) and type(getattr(layer, entry)) != type(None))]:
                    avg_norm = get_avg_init_norm(layer, param_type=param_type, ord=2)
                    cum_avg_norm += avg_norm**2
        cum_avg_norm = math.sqrt(cum_avg_norm)
        diameter = 2.0 * value * cum_avg_norm
        constraint = LpBall(n, ord=ord, diameter=diameter, radius=None)
    else:
        raise ValueError(f"Unknown mode {mode}")

    return [constraint]


@torch.no_grad()
def get_global_k_sparse_constraint(model, K=1, K_frac=None, value=300, mode='initialization'):
    """Create KSparsePolytope constraints for entire model, and value depends on mode (is radius, diameter, or
        factor to multiply average initialization norm with). K can be given either as an absolute (K) or relative value (K_frac)."""
    n = 0
    for layer in model.modules():
        for param_type in [entry for entry in ['weight', 'bias'] if
                           (hasattr(layer, entry) and type(getattr(layer, entry)) != type(None))]:
            param = getattr(layer, param_type)
            n += int(param.numel())

    if K_frac is None and K is None:
        raise ValueError("Both K and K_frac are None")
    elif K_frac is not None and K is not None:
        raise ValueError("Both K and K_frac given.")
    elif K_frac is None:
        real_K = min(int(K), n)
    elif K is None:
        real_K = min(int(K_frac * n), n)
    else:
        real_K = min(max(int(K), int(K_frac * n)), n)

    if mode == 'radius':
        constraint = KSparsePolytope(n, K=real_K, diameter=None, radius=value)
    elif mode == 'diameter':
        constraint = KSparsePolytope(n, K=real_K, diameter=value, radius=None)
    elif mode == 'initialization':
        cum_avg_norm = 0.0
        for layer in model.modules():
            if hasattr(layer, 'reset_parameters'):
                for param_type in [entry for entry in ['weight', 'bias'] if
                                   (hasattr(layer, entry) and type(getattr(layer, entry)) != type(None))]:
                    avg_norm = get_avg_init_norm(layer, param_type=param_type, ord=2)
                    cum_avg_norm += avg_norm ** 2
        cum_avg_norm = math.sqrt(cum_avg_norm)
        diameter = 2.0 * value * cum_avg_norm
        constraint = KSparsePolytope(n, K=real_K, diameter=diameter, radius=None)
    else:
        raise ValueError(f"Unknown mode {mode}")

    return [constraint]

@torch.no_grad()
def get_global_k_support_norm_ball_constraint(model, K=1, K_frac=None, value=300, mode='initialization'):
    """Create KSupportNormBall constraints for entire model, and value depends on mode (is radius, diameter, or
        factor to multiply average initialization norm with). K can be given either as an absolute (K) or relative value (K_frac)."""
    n = 0
    for layer in model.modules():
        for param_type in [entry for entry in ['weight', 'bias'] if
                           (hasattr(layer, entry) and type(getattr(layer, entry)) != type(None))]:
            param = getattr(layer, param_type)
            n += int(param.numel())

    if K_frac is None and K is None:
        raise ValueError("Both K and K_frac are None")
    elif K_frac is not None and K is not None:
        raise ValueError("Both K and K_frac given.")
    elif K_frac is None:
        real_K = min(int(K), n)
    elif K is None:
        real_K = min(int(K_frac * n), n)
    else:
        real_K = min(max(int(K), int(K_frac * n)), n)

    if mode == 'radius':
        constraint = KSupportNormBall(n, K=real_K, diameter=None, radius=value)
    elif mode == 'diameter':
        constraint = KSupportNormBall(n, K=real_K, diameter=value, radius=None)
    elif mode == 'initialization':
        cum_avg_norm = 0.0
        for layer in model.modules():
            if hasattr(layer, 'reset_parameters'):
                for param_type in [entry for entry in ['weight', 'bias'] if
                                   (hasattr(layer, entry) and type(getattr(layer, entry)) != type(None))]:
                    avg_norm = get_avg_init_norm(layer, param_type=param_type, ord=2)
                    cum_avg_norm += avg_norm ** 2
        cum_avg_norm = math.sqrt(cum_avg_norm)
        diameter = 2.0 * value * cum_avg_norm
        constraint = KSupportNormBall(n, K=real_K, diameter=diameter, radius=None)
    else:
        raise ValueError(f"Unknown mode {mode}")

    return [constraint]

@torch.no_grad()
def get_global_knormball_constraint(model, K=1, K_frac=None, value=300, mode='initialization'):
    """Create KNormBall constraints for entire model, and value depends on mode (is radius, diameter, or
        factor to multiply average initialization norm with). K can be given either as an absolute (K) or relative value (K_frac)."""
    n = 0
    for layer in model.modules():
        for param_type in [entry for entry in ['weight', 'bias'] if
                           (hasattr(layer, entry) and type(getattr(layer, entry)) != type(None))]:
            param = getattr(layer, param_type)
            n += int(param.numel())

    if K_frac is None and K is None:
        raise ValueError("Both K and K_frac are None")
    elif K_frac is not None and K is not None:
        raise ValueError("Both K and K_frac given.")
    elif K_frac is None:
        real_K = min(int(K), n)
    elif K is None:
        real_K = min(int(K_frac * n), n)
    else:
        real_K = min(max(int(K), int(K_frac * n)), n)

    if mode == 'radius':
        constraint = KNormBall(n, K=real_K, diameter=None, radius=value)
    elif mode == 'diameter':
        constraint = KNormBall(n, K=real_K, diameter=value, radius=None)
    elif mode == 'initialization':
        cum_avg_norm = 0.0
        for layer in model.modules():
            if hasattr(layer, 'reset_parameters'):
                for param_type in [entry for entry in ['weight', 'bias'] if
                                   (hasattr(layer, entry) and type(getattr(layer, entry)) != type(None))]:
                    avg_norm = get_avg_init_norm(layer, param_type=param_type, ord=2)
                    cum_avg_norm += avg_norm ** 2
        cum_avg_norm = math.sqrt(cum_avg_norm)
        diameter = 2.0 * value * cum_avg_norm
        constraint = KNormBall(n, K=real_K, diameter=diameter, radius=None)
    else:
        raise ValueError(f"Unknown mode {mode}")

    return [constraint]

@torch.no_grad()
def get_global_k_L0_constraint(model, K=1, K_frac=None):
    """Create L0 constraints for entire model. K can be given either as an absolute (K) or relative value (K_frac)."""
    n = 0
    for layer in model.modules():
        for param_type in [entry for entry in ['weight', 'bias'] if
                           (hasattr(layer, entry) and type(getattr(layer, entry)) != type(None))]:
            param = getattr(layer, param_type)
            n += int(param.numel())

    if K_frac is None and K is None:
        raise ValueError("Both K and K_frac are None")
    elif K_frac is not None and K is not None:
        raise ValueError("Both K and K_frac given.")
    elif K_frac is None:
        real_K = min(int(K), n)
    elif K is None:
        real_K = min(int(K_frac * n), n)
    else:
        real_K = min(max(int(K), int(K_frac * n)), n)

    constraint = L0Ball(n=n, k=real_K)
    return [constraint]

# Methods for setting local constraints
@torch.no_grad()
def set_lp_constraints(model, ord=2, value=300, mode='initialization'):
    """Create L_p constraints for each layer, where p == ord, and value depends on mode (is radius, diameter, or
    factor to multiply average initialization norm with)"""
    # Compute average init norms if necessary
    init_norms = dict()
    if mode == 'initialization':
        for layer in model.modules():
            if hasattr(layer, 'reset_parameters'):
                for param_type in [entry for entry in ['weight', 'bias'] if
                                   (hasattr(layer, entry) and type(getattr(layer, entry)) != type(None))]:
                    param = getattr(layer, param_type)
                    shape = param.shape

                    avg_norm = get_avg_init_norm(layer, param_type=param_type, ord=2)
                    if avg_norm == 0.0:
                        # Catch unlikely case that weight/bias is 0-initialized (e.g. BatchNorm does this)
                        avg_norm = 1.0
                    init_norms[shape] = avg_norm

    constraints = []
    for name, param in model.named_parameters():
        n = param.numel()
        if mode == 'radius':
            constraint = LpBall(n, ord=ord, diameter=None, radius=value)
        elif mode == 'diameter':
            constraint = LpBall(n, ord=ord, diameter=value, radius=None)
        elif mode == 'initialization':
            diameter = 2.0 * value * init_norms[param.shape]
            constraint = LpBall(n, ord=ord, diameter=diameter, radius=None)
        else:
            raise ValueError(f"Unknown mode {mode}")
        param.constraint = constraint
        constraints.append(constraint)
    return constraints


def set_k_sparse_constraints(model, K=1, K_frac=None, value=300, mode='initialization'):
    """Create KSparsePolytope constraints for each layer, and value depends on mode (is radius, diameter, or
    factor to multiply average initialization norm with). K can be given either as an absolute (K) or relative value (K_frac)."""
    # Compute average init norms if necessary
    init_norms = dict()
    if mode == 'initialization':
        for layer in model.modules():
            if hasattr(layer, 'reset_parameters'):
                for param_type in [entry for entry in ['weight', 'bias'] if
                                   (hasattr(layer, entry) and type(getattr(layer, entry)) != type(None))]:
                    param = getattr(layer, param_type)
                    shape = param.shape

                    avg_norm = get_avg_init_norm(layer, param_type=param_type, ord=2)
                    if avg_norm == 0.0:
                        # Catch unlikely case that weight/bias is 0-initialized (e.g. BatchNorm does this)
                        avg_norm = 1.0
                    init_norms[shape] = avg_norm

    constraints = []
    for name, param in model.named_parameters():
        n = param.numel()

        if K_frac is None and K is None:
            raise ValueError("Both K and K_frac are None")
        elif K_frac is not None and K is not None:
            raise ValueError("Both K and K_frac given.")
        elif K_frac is None:
            real_K = min(int(K), n)
        elif K is None:
            real_K = min(int(K_frac * n), n)
        else:
            real_K = min(max(int(K), int(K_frac * n)), n)

        if mode == 'radius':
            constraint = KSparsePolytope(n, K=real_K, diameter=None, radius=value)
        elif mode == 'diameter':
            constraint = KSparsePolytope(n, K=real_K, diameter=value, radius=None)
        elif mode == 'initialization':
            diameter = 2.0 * value * init_norms[param.shape]
            constraint = KSparsePolytope(n, K=real_K, diameter=diameter, radius=None)
        else:
            raise ValueError(f"Unknown mode {mode}")
        param.constraint = constraint
        constraints.append(constraint)
    return constraints


# Constraint classes
class Constraint:
    """
    Parent/Base class for constraints.
    Important note: For pruning to work, Projections and LMOs must be such that 0 entries in the input receive 0 entries in the output.
    :param n: dimension of constraint parameter space
    """

    def __init__(self, n):
        self.n = n
        self._diameter, self._radius = None, None

    def get_diameter(self):
        return self._diameter

    def get_radius(self):
        try:
            return self._radius
        except:
            raise ValueError("Tried to get radius from a constraint without one")

    def lmo(self, x):
        assert x.numel() == self.n, f"shape {x.shape} does not match dimension {self.n}"

    def shift_inside(self, x):
        assert x.numel() == self.n, f"shape {x.shape} does not match dimension {self.n}"

    def euclidean_project(self, x):
        assert x.numel() == self.n, f"shape {x.shape} does not match dimension {self.n}"


class LpBall(Constraint):
    """
    Constraint class for the n-dim Lp-Ball (p=ord) with L2-diameter diameter or radius.
    """

    def __init__(self, n, ord=2, diameter=None, radius=None):
        super().__init__(n)
        self.p = float(ord)
        self.q = get_lp_complementary_order(self.p)

        assert float(ord) >= 1, f"Invalid order {ord}"
        if diameter is None and radius is None:
            raise ValueError("Neither diameter nor radius given.")
        elif diameter is None:
            self._radius = radius
            self._diameter = 2 * convert_lp_radius(radius, self.n, in_ord=self.p, out_ord=2)
        elif radius is None:
            self._radius = convert_lp_radius(diameter / 2.0, self.n, in_ord=2, out_ord=self.p)
            self._diameter = diameter
        else:
            raise ValueError("Both diameter and radius given")

    @torch.no_grad()
    def lmo(self, x):
        """Returns v with norm(v, self.p) <= r minimizing v*x"""
        super().lmo(x)
        if self.p == 1:
            v = torch.zeros_like(x)
            maxIdx = torch.argmax(torch.abs(x))
            v.view(-1)[maxIdx] = -self._radius * torch.sign(x.view(-1)[maxIdx])
            return v
        elif self.p == 2:
            x_norm = float(torch.norm(x, p=2))
            if x_norm > tolerance:
                return -self._radius * x.div(x_norm)
            else:
                return torch.zeros_like(x)
        elif self.p == float('inf'):
            return torch.full_like(x, fill_value=-self._radius) * torch.sign(x)
        else:
            sgn_x = torch.sign(x).masked_fill_(x == 0, 1.0)
            absxqp = torch.pow(torch.abs(x), self.q / self.p)
            x_norm = float(torch.pow(torch.norm(x, p=self.q), self.q / self.p))
            if x_norm > tolerance:
                return -(self._radius / x_norm) * sgn_x * absxqp
            else:
                return torch.zeros_like(x)

    @torch.no_grad()
    def shift_inside(self, x):
        """Projects x to the LpBall with radius r.
        NOTE: This is a valid projection, although not the one mapping to minimum distance points.
        """
        super().shift_inside(x)
        x_norm = torch.norm(x, p=self.p)
        return self._radius * x.div(x_norm) if x_norm > self._radius else x

    @torch.no_grad()
    def euclidean_project(self, x):
        """Projects x to the closest (i.e. in L2-norm) point on the LpBall (p = 1, 2, inf) with radius r."""
        super().euclidean_project(x)
        if self.p == 1:
            x_norm = torch.norm(x, p=1)
            if x_norm > self._radius:
                sorted = torch.sort(torch.abs(x.flatten()), descending=True).values
                running_mean = (torch.cumsum(sorted, 0) - self._radius) / torch.arange(1, sorted.numel() + 1,
                                                                                       device=x.device)
                is_less_or_equal = sorted <= running_mean
                # This works b/c if one element is True, so are all later elements
                idx = is_less_or_equal.numel() - is_less_or_equal.sum() - 1
                return torch.sign(x) * torch.max(torch.abs(x) - running_mean[idx], torch.zeros_like(x))
            else:
                return x
        elif self.p == 2:
            x_norm = torch.norm(x, p=2)
            return self._radius * x.div(x_norm) if x_norm > self._radius else x
        elif self.p == float('inf'):
            return torch.clamp(x, min=-self._radius, max=self._radius)
        else:
            raise NotImplementedError(f"Projection not implemented for order {self.p}")

class KSupportNormBall(Constraint):
    """
    # Convex hull of all v s.t. ||v||_2 <= r, ||v||_0 <= k
    # This is a 'smooth' version of the KSparsePolytope, i.e. a mixture of KSparsePolytope and L2Ball allowing sparse activations of different magnitude
    # Note that the oracle will always return a vector v s.t. ||v||_0 == k, unless the input x satisfied ||x||_0 < k.
    # This Ball is due to Argyriou et al (2012)
    """

    def __init__(self, n, K=1, diameter=None, radius=None):
        super().__init__(n)

        self.k = min(K, n)
        if diameter is None and radius is None:
            raise ValueError("Neither diameter nor radius given")
        elif diameter is None:
            self._radius = radius
            self._diameter = 2.0 * radius
        elif radius is None:
            self._radius = diameter / 2.0
            self._diameter = diameter
        else:
            raise ValueError("Both diameter and radius given")

    def update_k(self, k: int) -> None:
        self.k = k

    @torch.no_grad()
    def lmo(self, x):
        """Returns v in KSupportNormBall w/ radius r minimizing v*x"""
        super().lmo(x)
        d = x.numel()
        if self.k <= d//2:
            # It's fast to get the maximal k values
            v = torch.zeros_like(x)
            maxIndices = torch.topk(torch.abs(x.flatten()), k=self.k).indices
            v.view(-1)[maxIndices] = x.view(-1)[maxIndices] # Projection to axis
        else:
            # Faster to get the n-d smallest values
            v = x.clone().detach()
            minIndices = torch.topk(torch.abs(x.flatten()), k=d-self.k, largest=False).indices
            v.view(-1)[minIndices] = 0  # Projection to axis
        v_norm = float(torch.norm(v, p=2))
        if v_norm > tolerance:
            return -self._radius * v.div(v_norm)    # Projection to Ball
        else:
            return torch.zeros_like(x)

    @torch.no_grad()
    def shift_inside(self, x):
        """Projects x to the KSupportNormBall w/ radius r.
        NOTE: This is a valid projection, although not the one mapping to minimum distance points.
        """
        super().shift_inside(x)
        x_norm = self.k_support_norm(x)
        return self._radius * x.div(x_norm) if x_norm > self._radius else x

    @torch.no_grad()
    def euclidean_project(self, x):
        super().euclidean_project(x)
        raise NotImplementedError(f"Projection not implemented for KSupportNormBall.")

    @torch.no_grad()
    def k_support_norm(self, x, tol=1e-10):
        """Computes the k-support-norm of x"""
        sorted_increasing = torch.sort(torch.abs(x.flatten()), descending=False).values
        running_mean = torch.cumsum(sorted_increasing, 0)  # Compute the entire running_mean since this is optimized
        running_mean = running_mean[-self.k:]  # Throw away everything but the last entries k entries
        running_mean = running_mean / torch.arange(1, self.k + 1, device=x.device)
        lower = sorted_increasing[-self.k:]
        upper = torch.cat([sorted_increasing[-(self.k-1):], torch.tensor([float('inf')], device=x.device)])
        relevantIndices = torch.nonzero(torch.logical_and(upper + tol > running_mean, running_mean + tol >= lower))[0]
        r = int(relevantIndices[0]) # Should have only one element, otherwise its a numerical problem -> pick first


        # With r, we can now compute the norm
        d = x.numel()
        x_right = 1/(r+1) * torch.sum(sorted_increasing[:d-(self.k-r)+1]).pow(2)
        x_left = torch.sum(sorted_increasing[-(self.k-1-r):].pow(2)) if r < self.k - 1 else 0
        x_norm = torch.sqrt(x_left + x_right)
        return float(x_norm)


class KSparsePolytope(Constraint):
    """
    # Polytopes with vertices v \in {0, +/- r}^n such that exactly k entries are nonzero
    # This is exactly the intersection of B_1(r*k) with B_inf(r)
    """

    def __init__(self, n, K=1, diameter=None, radius=None):
        super().__init__(n)

        self.k = min(K, n)

        if diameter is None and radius is None:
            raise ValueError("Neither diameter nor radius given")
        elif diameter is None:
            self._radius = radius
            self._diameter = 2.0 * radius * math.sqrt(self.k)
        elif radius is None:
            self._radius = diameter / (2.0 * math.sqrt(self.k))
            self._diameter = diameter
        else:
            raise ValueError("Both diameter and radius given")

    @torch.no_grad()
    def lmo(self, x):
        """Returns v in KSparsePolytope w/ radius r minimizing v*x"""
        super().lmo(x)
        v = torch.zeros_like(x)
        maxIndices = torch.topk(torch.abs(x.flatten()), k=self.k).indices
        v.view(-1)[maxIndices] = -self._radius * torch.sign(x.view(-1)[maxIndices])
        return v

    @torch.no_grad()
    def shift_inside(self, x):
        """Projects x to the KSparsePolytope with radius r.
        NOTE: This is a valid projection, although not the one mapping to minimum distance points.
        """
        super().shift_inside(x)
        x_norm = self.k_sparse_norm(x)
        return self._radius * x.div(x_norm) if x_norm > self._radius else x

    @torch.no_grad()
    def euclidean_project(self, x):
        super().euclidean_project(x)
        raise NotImplementedError(f"Projection not implemented for K-sparse polytope.")

    @torch.no_grad()
    def k_sparse_norm(self, x):
        """Computes the k-sparse-norm of x"""
        Linf = torch.norm(x, p=float('inf'))
        L1k = torch.norm(x/self.k, p=1)
        return float(max(Linf, L1k))


class L0Ball(Constraint):
    """
    Constraint class for the n-dim L0-Ball(k). This has been separated from the LpBall class on purpose.
    Keep in mind that this is not a convex set, this is mainly used for projecting onto.
    """

    def __init__(self, n, k=1):
        super().__init__(n)
        self.k = k

        assert 1 <= k == int(k), f"Invalid k {k}"

    @torch.no_grad()
    def lmo(self, x):
        """Returns v with norm(v, self.p) <= r minimizing v*x"""
        super().lmo(x)
        raise NotImplementedError(f"LMO not implemented for L_0 Ball.")

    @torch.no_grad()
    def shift_inside(self, x):
        """Simply calls the euclidean projection.
        """
        super().shift_inside(x)
        return self.euclidean_project(x)

    @torch.no_grad()
    def euclidean_project(self, x):
        """Projects x to the closest (i.e. in L2-norm) point on the Ball."""
        super().euclidean_project(x)
        z = torch.zeros_like(x)
        indices = torch.topk(torch.abs(x).view(-1), k=self.k).indices
        z.view(-1)[indices] = x.view(-1)[indices]
        return z

class KNormBall(Constraint):
    """
        # Convex Hull of union of B_1(r) and B_inf(r/k)
    """

    def __init__(self, n, K=1, diameter=None, radius=None):
        super().__init__(n)

        self.k = min(K, n)

        if diameter is None and radius is None:
            raise ValueError("Neither diameter nor radius given")
        elif diameter is None:
            self._radius = radius
            self.rhombus = LpBall(n, ord=1, diameter=None, radius=self._radius)
            self.cube = LpBall(n, ord=float('inf'), diameter=None, radius=self._radius/self.k)
            self._diameter = max(self.rhombus._diameter, self.cube._diameter)
        elif radius is None:
            self._diameter = diameter
            self._radius = 0.5 * self._diameter if math.sqrt(n)/self.k <= 1 else 0.5 * self._diameter * self.k/math.sqrt(n)
            self.rhombus = LpBall(n, ord=1, diameter=None, radius=self._radius)
            self.cube = LpBall(n, ord=float('inf'), diameter=None, radius=self._radius/self.k)
        else:
            raise ValueError("Both diameter and radius given")

    @torch.no_grad()
    def lmo(self, x):
        """Returns v in KNormBall w/ radius r minimizing v*x"""
        super().lmo(x)
        rhombus_candidate = self.rhombus.lmo(x)
        cube_candidate = self.cube.lmo(x)

        rhombus_value = torch.dot(rhombus_candidate.flatten(), x.flatten())
        cube_value = torch.dot(cube_candidate.flatten(), x.flatten())
        return rhombus_candidate if cube_value > rhombus_value else cube_candidate

    @torch.no_grad()
    def shift_inside(self, x):
        """Projects x to the KNormBall with radius r.
        NOTE: This is a valid projection, although not the one mapping to minimum distance points.
        """
        super().shift_inside(x)
        k_norm = self.k_norm(x)
        return self._radius * x.div(k_norm) if k_norm > self._radius else x

    @torch.no_grad()
    def euclidean_project(self, x):
        super().euclidean_project(x)
        raise NotImplementedError(f"Projection not implemented for KNormBall.")

    @torch.no_grad()
    def k_norm(self, x):
        return float(torch.sum(torch.topk(torch.abs(x.flatten()), k=self.k).values))