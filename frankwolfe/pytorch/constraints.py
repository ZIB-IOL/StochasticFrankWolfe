# ===========================================================================
# Project:      FrankWolfe 2020
# File:         constraints.py
# Description:  Contains LMO-oracle classes for Pytorch
# ===========================================================================
import torch
import torch.nn.functional as F
import math

tolerance = 1e-10

#### HELPER FUNCTIONS ####
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
    in_ord_rec = 0.5 if in_ord == 1 else 1.0/in_ord
    out_ord_rec = 0.5 if out_ord == 1 else 1.0/out_ord

    return r * N**(out_ord_rec - in_ord_rec)

def get_lp_complementary_order(ord):
    """Get the complementary order"""
    ord = float(ord)
    if ord == float('inf'): return 1
    elif ord == 1: return float('inf')
    elif ord >= 2: return 1.0/(1.0 - 1.0/ord)
    else: raise NotImplementedError(f"Order {ord} not supported.")

#### LMO BASE CLASSES ####
class Constraint:
    """
    Parent/Base class for constraints
    :param n: dimension of constraint parameter space
    """
    def __init__(self, n):
        self.n = n
        self._diameter, self._radius = None, None

    def get_diameter(self):
        return self._diameter

    def get_radius(self):
        try: return self._radius
        except: raise ValueError("Tried to get radius from a constraint without one")

    def lmo(self, x):
        assert x.numel() == self.n, f"shape {x.shape} does not match dimension {self.n}"

    def shift_inside(self, x):
        assert x.numel() == self.n, f"shape {x.shape} does not match dimension {self.n}"

    def euclidean_project(self, x):
        assert x.numel() == self.n, f"shape {x.shape} does not match dimension {self.n}"

class Unconstrained(Constraint):
    """
    Parent/Base class for unconstrained parameter spaces
    :param n: dimension of unconstrained parameter space
    """
    def __init__(self, n):
        super().__init__(n)
        self._diameter = float('inf')

    def lmo(self, x):
        super().__init__(x)
        raise NotImplementedError("No lmo for unconstrained parameters")

    def shift_inside(self, x):
        super().__init__(x)
        return x

    def euclidean_project(self, x):
        super().__init__(x)
        return x

#### LMO CLASSES ####
class LpBall(Constraint):
    """
    LMO class for the n-dim Lp-Ball (p=ord) with L2-diameter diameter or radius.
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
            self._diameter = 2*convert_lp_radius(radius, self.n, in_ord=self.p, out_ord=2)
        elif radius is None:
            self._radius = convert_lp_radius(diameter/2.0, self.n, in_ord=2, out_ord=self.p)
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
            return torch.full_like(x, fill_value=self._radius).masked_fill_(x > 0, -self._radius)
        else:
            sgn_x = torch.sign(x).masked_fill_(x == 0, 1.0)
            absxqp = torch.pow(torch.abs(x), self.q / self.p)
            x_norm = float(torch.pow(torch.norm(x, p=self.q), self.q / self.p))
            if x_norm > tolerance:
                return -self._radius / x_norm * sgn_x * absxqp
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
                running_mean = (torch.cumsum(sorted, 0) - self._radius) / torch.arange(1, sorted.numel() + 1, device=x.device)
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
        L1Norm = float(torch.norm(x, p=1))
        LinfNorm = float(torch.norm(x, p=float('inf')))
        if L1Norm > self._radius * self.k or LinfNorm > self._radius:
            x_norm = max(L1Norm, LinfNorm)
            x_unit = x.div(x_norm)
            factor = min(math.floor(1. / float(torch.norm(x_unit, p=float('inf')))), self.k)
            assert 1 <= factor <= self.k
            return factor * self._radius * x_unit
        else:
            return x

    @torch.no_grad()
    def euclidean_project(self, x):
        super().euclidean_project(x)
        raise NotImplementedError(f"Projection not implemented for K-sparse polytope.")