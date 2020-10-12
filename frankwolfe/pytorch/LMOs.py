# ===========================================================================
# Project:      FrankWolfe 2020
# File:         LMOs.py
# Description:  Contains LMO-oracle classes for Pytorch
# ===========================================================================
import torch
import torch.nn.functional as F
import math
tolerance = 1e-10

#### HELPER FUNCTIONS ####
@torch.no_grad()
def get_avg_init_norm(layer, param_type=None, ord=2, repetitions=100):
    output = 0
    for _ in range(repetitions):
        layer.reset_parameters()
        output += torch.norm(getattr(layer, param_type), p=ord).item()

    return float(output) / repetitions

def convert_radius(r, N, in_ord=2, out_ord='inf'):
    """
    Convert between radius of Lp balls such that the ball of order out_order
    has the same L2 diameter as the ball with radius r of order in_order
    in N dimensions
    """
    in_ord = float('inf') if in_ord == 'inf' else in_ord
    out_ord = float('inf') if out_ord =='inf' else out_ord

    in_ord_rec = 0.5 if in_ord == 1 else 1/in_ord
    out_ord_rec = 0.5 if out_ord == 1 else 1/out_ord

    return r * N**(out_ord_rec - in_ord_rec)

def complementary_order(ord):
    ord = float('inf') if ord == 'inf' else ord

    if ord == float('inf'):
        return 1
    elif ord == 1:
        return float('inf')
    elif ord >= 2:
        return 1 / (1 - 1/ord)
    else:
        raise NotImplementedError(f"Order {ord} not supported.")


#### LMO CLASSES ####
class LpBall:
    """
    oracle class for the Lp Ball (p=ord) with with L2-diameter diameter
    """
    def __init__(self, diameter=None, radius=None, width=None, ord=2):
        self.p = float('inf') if ord=='inf' else ord

        assert diameter is None or radius is None, "Both diameter and radius given."
        assert (diameter is None and radius is None) or width is None, "Specify either diameter/radius or initialization width."

        self.d = diameter
        self.r = radius
        self.width = width
        self.initializerDependent = not (width is None)

    @torch.no_grad()
    def set_diameters(self, model):
        self._diameter_dict = dict()
        self._radius_dict = dict()

        for layer in model.modules():
            if hasattr(layer, 'reset_parameters'):
                for param_type in [entry for entry in ['weight', 'bias'] if (hasattr(layer, entry) and
                                                                             type(getattr(layer, entry)) != type(None))]:
                    param = getattr(layer, param_type)
                    shape = param.shape

                    avg_norm = get_avg_init_norm(layer, param_type=param_type, ord=2)
                    if avg_norm == 0.0:
                        # Catch case that weight/bias is 0-initialized (e.g. BatchNorm does this)
                        avg_norm = 1.0
                    self._diameter_dict[shape] = 2.0 * self.width * avg_norm
                    self._radius_dict[shape] = convert_radius(self.width * avg_norm, torch.numel(param), in_ord=2,
                                                              out_ord=self.p)

    @torch.no_grad()
    def get_radius(self, shape):
        if self.initializerDependent:
            return self._radius_dict[shape]
        else:
            return self.r or convert_radius(self.d/2, shape.numel(), in_ord=2, out_ord=self.p)

    @torch.no_grad()
    def get_diameter(self, shape):
        if self.initializerDependent:
            return self._diameter_dict[shape]
        else:
            return self.d or 2*convert_radius(self.r, shape.numel(), in_ord=self.p, out_ord=2)


    @torch.no_grad()
    def oracle(self, x):
        """Returns v with norm(v, self.p) <= r minimizing v*x"""
        r = self.get_radius(x.shape)
        q = complementary_order(self.p)

        if self.p == 1:
            v = torch.zeros_like(x)
            maxIdx = torch.argmax(torch.abs(x))
            v.view(-1)[maxIdx] = -r * torch.sign(x.view(-1)[maxIdx])
            return v
        elif self.p == 2:
            x_norm = float(torch.norm(x, p=2))
            if x_norm > tolerance:
                return -r * x.div(x_norm)
            else:
                return torch.zeros_like(x)
        elif self.p == float('inf'):
            return torch.full_like(x, fill_value=r).masked_fill_(x > 0, -r)
        else:
            sgn_x = torch.sign(x).masked_fill_(x == 0, 1.0)
            absxqp = torch.pow(torch.abs(x), q/self.p)
            x_norm = float(torch.pow(torch.norm(x, p=q), q/self.p))
            if x_norm > tolerance:
                return -r/x_norm * sgn_x * absxqp
            else:
                return torch.zeros_like(x)

    @torch.no_grad()
    def shift_inside(self, x):
        """Projects x to the LpBall with radius r.
        NOTE: This is a valid projection, although not the one mapping to minimum distance points.
        """
        r = self.get_radius(x.shape)
        x_norm = torch.norm(x, p=self.p)
        return r * x.div(x_norm) if x_norm > r else x

    @torch.no_grad()
    def project(self, x):
        """Projects x to the closest (i.e. in L2-norm) point on the LpBall (p = 1, 2, inf) with radius r."""
        r = self.get_radius(x.shape)

        if self.p == 1:
            x_norm = torch.norm(x, p=1)
            if x_norm > r:
                sorted = torch.sort(torch.abs(x.flatten()), descending=True).values
                running_mean = (torch.cumsum(sorted, 0) - r) / torch.arange(1, sorted.numel() + 1, device=x.device)
                is_less_or_equal = sorted <= running_mean
                # This works b/c if one element is True, so are all later elements
                idx = is_less_or_equal.numel() - is_less_or_equal.sum() - 1
                return torch.sign(x) * torch.max(torch.abs(x) - running_mean[idx], torch.zeros_like(x))
            else:
                return x
        elif self.p == 2:
            x_norm = torch.norm(x, p=2)
            return r * x.div(x_norm) if x_norm > r else x
        elif self.p == float('inf'):
            return torch.clamp(x, min=-r, max=r)
        else:
            raise NotImplementedError(f"Projection not implemented for order {self.p}")

class SparseBall:
    """
    # Polytopes with vertices v \in {0, +/- r}^n such that exactly k entries are nonzero
    # This is exactly the intersection of B_1(r*k) with B_inf(r)
    """
    def __init__(self, diameter=None, radius=None, width=None, k=1):
        self.k = k

        assert diameter is None or radius is None, "Both diameter and radius given"
        assert int(k) == k, "k must be integral"
        assert (diameter is None and radius is None) or width is None, "Specify either diameter/radius or initialization width."


        self.d = diameter
        self.r = radius # This is the L2 radius and needs to be converted
        self.width = width
        self.initializerDependent = not (width is None)

    def convert_sparseball_radius(self, r, N, in_ord, out_ord):
        if in_ord == 'SparseBall' and out_ord == 'SparseBall':
            return r
        elif in_ord == 'SparseBall':
            radius_L2 = math.sqrt(self.k) * r
            return convert_radius(r=radius_L2, N=N, in_ord=2, out_ord=out_ord)
        elif out_ord == 'SparseBall':
            radius_L2 = convert_radius(r=r, N=N, in_ord=in_ord, out_ord=2)
            return radius_L2 / math.sqrt(self.k)
        else:
            return convert_radius(r=r, N=N, in_ord=in_ord, out_ord=out_ord)

    @torch.no_grad()
    def set_diameters(self, model):
        self._diameter_dict = dict()
        self._radius_dict = dict()

        for layer in model.modules():
            if hasattr(layer, 'reset_parameters'):
                for param_type in [entry for entry in ['weight', 'bias'] if (hasattr(layer, entry) and
                                                                             type(getattr(layer, entry)) != type(None))]:
                    param = getattr(layer, param_type)
                    shape = param.shape

                    avg_norm = get_avg_init_norm(layer, param_type=param_type, ord=2)
                    if avg_norm == 0.0:
                        # Catch unlikely case that weight/bias is 0-initialized (e.g. BatchNorm does this)
                        avg_norm = 1.0
                    self._diameter_dict[shape] = 2.0 * self.width * avg_norm
                    self._radius_dict[shape] = self.convert_sparseball_radius(self.width * avg_norm, torch.numel(param), in_ord=2,
                                                              out_ord="SparseBall")

    @torch.no_grad()
    def get_radius(self, shape):
        if self.initializerDependent:
            return self._radius_dict[shape]
        else:
            return self.r or self.convert_sparseball_radius(self.d/2, shape.numel(), in_ord=2, out_ord='SparseBall')

    @torch.no_grad()
    def get_diameter(self, shape):
        if self.initializerDependent:
            return self._diameter_dict[shape]
        else:
            return self.d or 2*self.convert_sparseball_radius(self.r, shape.numel(), in_ord='SparseBall', out_ord=2)

    @torch.no_grad()
    def oracle(self, x):
        """Returns v in Sparseball w/ radius r minimizing v*x"""
        r = self.get_radius(x.shape)
        v = torch.zeros_like(x)
        maxIndices = torch.topk(torch.abs(x.flatten()), k=self.k).indices
        v.view(-1)[maxIndices] = -r * torch.sign(x.view(-1)[maxIndices])
        return v

    @torch.no_grad()
    def shift_inside(self, x):
        """Projects x to the SparseBall with radius r.
        NOTE: This is a valid projection, although not the one mapping to minimum distance points.
        """
        r = self.get_radius(x.shape)
        L1Norm = float(torch.norm(x, p=1))
        LinfNorm = float(torch.norm(x, p=float('inf')))
        if L1Norm > r*self.k or LinfNorm > r:
            x_norm = max(L1Norm, LinfNorm)
            x_unit = x.div(x_norm)
            factor = min(math.floor(1./float(torch.norm(x_unit, p=float('inf')))), self.k)
            assert 1 <= factor <= self.k
            return factor * r * x_unit
        else:
            return x

    @torch.no_grad()
    def project(self, x):
        """Projects x to the closest (i.e. in L2-norm) point on the Sparseball with radius r."""
        raise NotImplementedError(f"Projection not implemented for SparseBall.")

class KBall:

    def __init__(self, radius=1, k=2.0):
        self.k, self.radius = k, radius

        self.rhombus = LpBall(radius=radius, ord=1)
        self.cube = LpBall(radius=radius/self.k, ord=float('inf'))

    @torch.no_grad()
    def get_diameter(self, shape):
        return max(self.rhombus.get_diameter(shape), self.cube.get_diameter(shape))

    @torch.no_grad()
    def oracle(self, x):
        rhombus_candidate = self.rhombus.oracle(x)
        cube_candidate = self.cube.oracle(x)

        rhombus_value = torch.dot(rhombus_candidate.flatten(), x.flatten())
        cube_value = torch.dot(cube_candidate.flatten(), x.flatten())
        return rhombus_candidate if cube_value > rhombus_value else cube_candidate

    @torch.no_grad()
    def shift_inside(self, x):
        k_norm = self.k_norm(x)
        return self.radius * x.div(k_norm) if k_norm > self.radius else x

    @torch.no_grad()
    def k_norm(self, x):
        return float(torch.sum(torch.topk(torch.abs(x.flatten()), k=self.k).values))

class ProbabilitySimplex:
    """
    oracle class for the probability simplex, i.e. {x \in R^n| x_i >= 0, x_1 + .. + x_n = lambdaVal}
    """

    def __init__(self, lambdaVal=1):
        self.lambdaVal = lambdaVal

    @torch.no_grad()
    def oracle(self, x):
        """Returns v in probability simplex minimizing v*x as follows:
        Returns tensor v of same shape as x where v_i = 1 if x_i = min{x_j}
        """
        v = torch.zeros_like(x)
        minIdx = torch.argmin(x)
        v.view(-1)[minIdx] = self.lambdaVal
        return v

    @torch.no_grad()
    def project(self, x):
        """Projects x to the ProbabilitySimplex with radius lambdaVal using softmax.
        NOTE: This is some projection, although not the one mapping to minimum distance points.
        """
        return self.lambdaVal * F.softmax(x.view(-1), dim=-1).view(x.shape)

class Permutahedron:
    """
    oracle class for the permutahedron, i.e. conv{sigma([n]) | sigma permutation of [n]}
    """

    def __init__(self):
        pass

    @torch.no_grad()
    def oracle(self, x):
        """Returns v in permutahedron minimizing v*x"""
        sortIndices = torch.argsort(x.view(-1), descending=True)
        v = torch.zeros_like(sortIndices)
        v[sortIndices] = torch.arange(start=1, step=1, end=v.shape[0] + 1, device=v.device)
        return v.view_as(x)

    @torch.no_grad()
    def project(self, x):
        """Projects x to the permutahedron.
        NOTE: This is some projection, although not the one mapping to minimum distance points.
        """
        return torch.argsort(x.view(-1)).view(x.shape).type(x.dtype) + 1
