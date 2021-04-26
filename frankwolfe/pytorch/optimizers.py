# ===========================================================================
# Project:      StochasticFrankWolfe 2020 / IOL Lab @ ZIB
# File:         pytorch/optimizers.py
# Description:  Pytorch implementation of Stochastic Frank Wolfe, AdaGradSFW and SGD with projection
# ===========================================================================
import torch
from collections import OrderedDict
# TODO: How do we handle unconstrained parameters?

class SFW(torch.optim.Optimizer):
    """Stochastic Frank Wolfe Algorithm
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate between 0.0 and 1.0
        rescale (string or None): Type of lr rescaling. Must be 'diameter', 'gradient' or None
        momentum (float): momentum factor, 0 for no momentum
    """

    def __init__(self, params, lr=0.1, rescale='diameter', momentum=0, dampening=0, global_constraint=None, distance_penalty=None):
        momentum = momentum or 0
        dampening = dampening or 0
        distance_penalty = distance_penalty or 0
        if rescale is None and not (0.0 <= lr <= 1.0):
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not (0.0 <= momentum <= 1.0):
            raise ValueError(f"Momentum must be between 0 and 1, got {momentum} of type {type(momentum)}.")
        if not (0.0 <= dampening <= 1.0):
            raise ValueError("fDampening must be between 0 and 1, got {dampening} of type {type(dampening)}.")
        if not 0.0 <= distance_penalty <= 1.0:
            raise ValueError("NPO distance penalty must be in [0,1] or None.")
        if rescale == 'None': rescale = None
        if not (rescale in ['diameter', 'gradient', None, 'sparse_gradient', 'normalized_gradient', 'sparse_normalized_gradient']):
            raise ValueError(f"Rescale type must be either 'diameter', 'gradient' or None, got {rescale} of type {type(rescale)}.")
        #if rescale == 'gradient' and distance_penalty > 0:
        #    raise NotImplementedError("Cannot use gradient rescaling and a distance_penalty > 0.")

        self.rescale = rescale
        self.global_constraint = global_constraint  # If not None, this points to the global constraint instance
        self.distance_penalty = distance_penalty

        self.effective_lr = lr  # Just to catch this as a metric


        defaults = dict(lr=lr, momentum=momentum, dampening=dampening)
        super(SFW, self).__init__(params, defaults)
        assert not (self.global_constraint and len(
            self.param_groups) > 1), "This does not work for multiple param_groups yet."

    @torch.no_grad()
    def reset_momentum(self):
        """Resets momentum, typically used directly after pruning"""
        for group in self.param_groups:
            momentum = group['momentum']
            if momentum > 0:
                for p in group['params']:
                    param_state = self.state[p]
                    if 'momentum_buffer' in param_state: del param_state['momentum_buffer']

    @torch.no_grad()
    def set_distance_penalty(self, penalty):
        """Resets the distance penalty"""
        self.distance_penalty = penalty

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self.global_constraint:
            # Do the iterative default step
            self.iterative_step()
        elif self.global_constraint:
            self.non_iterative_step()
        return loss

    @torch.no_grad()
    def iterative_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                # Add momentum
                momentum = group['momentum']
                dampening = group['dampening']
                if momentum > 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = d_p.detach().clone()
                    else:
                        param_state['momentum_buffer'].mul_(momentum).add_(d_p, alpha=1 - dampening)
                    d_p = param_state['momentum_buffer']

                # LMO solution
                if self.distance_penalty > 0:
                    v = p.constraint.npo(x=p, d_x=d_p, penalty=self.distance_penalty)
                else:
                    v = p.constraint.lmo(d_p)  # LMO optimal solution

                # Determine learning rate rescaling factor
                factor = 1
                if self.rescale == 'diameter':
                    # Rescale lr by diameter
                    factor = 1. / p.constraint.get_diameter()
                elif self.rescale == 'gradient':
                    # Rescale lr by gradient
                    factor = torch.norm(d_p, p=2) / torch.norm(p - v, p=2)

                lr = max(0.0, min(factor * group['lr'], 1.0))  # Clamp between [0, 1]

                # Update parameters
                p.mul_(1 - lr)
                p.add_(v, alpha=lr)

    @torch.no_grad()
    def non_iterative_step(self):
        group = self.param_groups[0]
        momentum = group['momentum']
        dampening = group['dampening']

        # Collect relevant parameters
        param_list = [p for p in group['params'] if p.grad is not None]

        # Add momentum, fill grad list with momentum_buffers and concatenate
        grad_list = []
        for p in param_list:
            d_p = p.grad
            if momentum > 0:
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = d_p.detach().clone()
                else:
                    param_state['momentum_buffer'].mul_(momentum).add_(d_p, alpha=1 - dampening)
                d_p = param_state['momentum_buffer']
            grad_list.append(d_p)
        grad_vec = torch.cat([g.view(-1) for g in grad_list])

        # LMO solution
        if self.distance_penalty > 0:
            v = self.global_constraint.npo(x=torch.cat([p.view(-1) for p in param_list]), d_x=grad_vec, penalty=self.distance_penalty)
        else:
            v = self.global_constraint.lmo(grad_vec)  # LMO optimal solution

        # Determine learning rate rescaling factor
        factor = 1
        if self.rescale == 'diameter':
            # Rescale lr by diameter
            factor = 1. / self.global_constraint.get_diameter()
        elif self.rescale == 'gradient':
            # Rescale lr by gradient
            factor = torch.norm(grad_vec, p=2) / torch.norm(torch.cat([p.view(-1) for p in param_list]) - v, p=2)
        elif self.rescale == 'sparse_gradient':
            factor = self.global_constraint.last_sparse_grad_norm / torch.norm(torch.cat([p.view(-1) for p in param_list]) - v, p=2)
        elif self.rescale == 'normalized_gradient':
            factor = torch.norm(grad_vec, p=2) / (torch.norm(torch.cat([p.view(-1) for p in param_list]) - v, p=2) / self.global_constraint.get_diameter())
        elif self.rescale == 'sparse_normalized_gradient':
            factor = self.global_constraint.last_sparse_grad_norm / (torch.norm(torch.cat([p.view(-1) for p in param_list]) - v, p=2) / self.global_constraint.get_diameter())

        lr = max(0.0, min(factor * group['lr'], 1.0))  # Clamp between [0, 1]

        # Update parameters
        for p in param_list:
            numberOfElements = p.numel()
            p.mul_(1 - lr)
            p.add_(v[:numberOfElements].view(p.shape), alpha=lr)
            v = v[numberOfElements:]
        self.effective_lr = lr  # Just to catch this as a metric


class SGD(torch.optim.Optimizer):
    """Modified SGD which allows projection via Constraint class"""

    def __init__(self, params, lr=0.1, momentum=0, dampening=0, nesterov=False,
                 weight_decay=0, weight_decay_ord=2, global_constraint=None, gradient_callback=None):
        momentum = momentum or 0
        dampening = dampening or 0
        weight_decay = weight_decay or 0
        weight_decay_ord = float(weight_decay_ord) if weight_decay_ord is not None else 2
        if not weight_decay_ord >= 1:
            raise ValueError(f"Invalid weight_decay order: {weight_decay_ord}.")
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not (0.0 <= momentum <= 1.0):
            raise ValueError("Momentum must be between 0 and 1.")
        if not (0.0 <= dampening <= 1.0):
            raise ValueError("Dampening must be between 0 and 1.")
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and (momentum == 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum and zero dampening")

        self.global_constraint = global_constraint  # If not None, this points to the constraint instance
        self.gradient_callback = gradient_callback # If not None, call this function right before the update step

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, weight_decay_ord=weight_decay_ord, nesterov=nesterov)
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def reset_momentum(self):
        """Resets momentum, typically used directly after pruning"""
        for group in self.param_groups:
            momentum = group['momentum']
            if momentum > 0:
                for p in group['params']:
                    param_state = self.state[p]
                    if 'momentum_buffer' in param_state: del param_state['momentum_buffer']

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grad_list = []  # Collect gradients if we need to call the gradient_callback
        param_dict = OrderedDict()
        for i in range(len(self.param_groups)):
            param_dict[i] = []

        # For Linf regularization, we need to compute the maximal element out of all parameters
        for group in self.param_groups:
            if group['weight_decay'] > 0 and group['weight_decay_ord'] == float('inf'):
                group['maxParam'] = max(float(torch.max(torch.abs(p)))
                                        for p in group['params'] if p.grad is not None)

        for idx, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            weight_decay_ord = group['weight_decay_ord']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                if weight_decay > 0:
                    if weight_decay_ord == 1:
                        # L1 regularization
                        d_p = d_p.add(torch.sign(p), alpha=weight_decay)
                    elif weight_decay_ord == 2:
                        # L2 regularization
                        d_p = d_p.add(p, alpha=weight_decay)
                    elif weight_decay_ord == float('inf'):
                        # Linf regularization
                        maxParam = group['maxParam']
                        d_p = d_p.add(torch.sign(p) * (torch.abs(p) == maxParam), alpha=weight_decay)
                    else:
                        # Arbitrary Lp regularization when p ist not 1, 2 or inf
                        d_p = d_p.add(torch.sign(p) * torch.abs(p).pow(weight_decay_ord - 1), alpha=weight_decay)
                if momentum > 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                if self.gradient_callback is not None:
                    grad_list.append(d_p.view(-1))
                    param_dict[idx].append(p)
                else:
                    p.add_(d_p, alpha=-group['lr'])

                    # Project if necessary
                    if not self.global_constraint:  # We have to project afterwards
                        if hasattr(p, 'constraint'):
                            p.copy_(p.constraint.euclidean_project(p))

        if self.gradient_callback is not None:
            d_p = self.gradient_callback(torch.cat(grad_list))
            for idx in param_dict:
                group = self.param_groups[idx]
                for p in param_dict[idx]:
                    numberOfElements = p.numel()
                    p.add_(d_p[:numberOfElements].view(p.shape), alpha=-group['lr'])
                    d_p = d_p[numberOfElements:]
                    # Project if necessary
                    if not self.global_constraint:  # We have to project afterwards
                        if hasattr(p, 'constraint'):
                            p.copy_(p.constraint.euclidean_project(p))

        if self.global_constraint:
            # Project entire gradient vector
            assert len(self.param_groups) == 1, "This does not work for multiple param_groups yet."
            param_dict = [p for p in self.param_groups[0]['params'] if p.grad is not None]
            p_proj = self.global_constraint.euclidean_project(torch.cat([p.view(-1) for p in param_dict]))
            for p in param_dict:
                numberOfElements = p.numel()
                p.copy_(p_proj[:numberOfElements].view(p.shape))
                p_proj = p_proj[numberOfElements:]

        return loss


class AdaGradSFW(torch.optim.Optimizer):
    """AdaGrad Stochastic Frank-Wolfe algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        inner_steps (integer, optional): number of inner iterations (default: 2)
        lr (float, optional): learning rate (default: 1e-2)
        delta (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        momentum (float, optional): momentum factor
    """

    def __init__(self, params, inner_steps=2, lr=1e-2, delta=1e-8, momentum=0.9):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("Momentum must be between [0, 1].")
        if not 0.0 <= delta:
            raise ValueError("Invalid delta value: {}".format(delta))
        if not int(inner_steps) == inner_steps and not 0.0 <= inner_steps:
            raise ValueError("Number of inner iterations needs to be a positive integer: {}".format(inner_steps))

        self.K = inner_steps

        defaults = dict(lr=lr, delta=delta, momentum=momentum)
        super(AdaGradSFW, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['sum'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def reset_momentum(self):
        """Resets momentum, typically used directly after pruning"""
        for group in self.param_groups:
            momentum = group['momentum']
            if momentum > 0:
                for p in group['params']:
                    param_state = self.state[p]
                    if 'momentum_buffer' in param_state: del param_state['momentum_buffer']

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad
                param_state = self.state[p]
                momentum = group['momentum']

                if momentum > 0:
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = d_p.detach().clone()
                    else:
                        param_state['momentum_buffer'].mul_(momentum).add_(d_p, alpha=1 - momentum)
                        d_p = param_state['momentum_buffer']

                param_state['sum'].addcmul_(d_p, d_p, value=1)  # Holds the cumulative sum
                H = torch.sqrt(param_state['sum']).add(group['delta'])

                y = p.detach().clone()
                for _ in range(self.K):
                    d_Q = d_p.addcmul(H, y - p, value=1. / group['lr'])
                    y_v_diff = y - p.constraint.lmo(d_Q)
                    gamma = group['lr'] * torch.div(torch.sum(torch.mul(d_Q, y_v_diff)),
                                                    torch.sum(H.mul(torch.mul(y_v_diff, y_v_diff))))
                    gamma = max(0.0, min(gamma, 1.0))  # Clamp between [0, 1]

                    y.add_(y_v_diff, alpha=-gamma)  # -gamma needed as we want to add v-y, not y-v
                p.copy_(y)
        return loss


class Prox_SGD(torch.optim.SGD):
    """Straightforward implementation of Proximal SGD. Takes as input the same as torch.optim.SGD, but no weight_decay."""
    def __init__(self, params, prox_operator, global_constraint=False, **kwargs):
        assert ('weight_decay' not in kwargs) or (kwargs['weight_decay'] == 0), "Nonzero weight decay to Prox_SGD given."
        # Convert none values to default values
        checkArgList = ['momentum', 'dampening']
        for arg in checkArgList:
            if arg in kwargs:
                kwargs[arg] = kwargs[arg] or 0
        super().__init__(params, **kwargs)
        self.prox_operator = prox_operator
        self.global_constraint = global_constraint

    @torch.no_grad()
    def reset_momentum(self):
        """Resets momentum, typically used directly after pruning"""
        for group in self.param_groups:
            momentum = group['momentum']
            if momentum > 0:
                for p in group['params']:
                    param_state = self.state[p]
                    if 'momentum_buffer' in param_state: del param_state['momentum_buffer']

    @torch.no_grad()
    def step(self, closure=None):
        # Perform an SGD step, then apply the proximal operator to all parameters
        super().step(closure=closure)

        if not self.global_constraint:
            # Do the iterative default step
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.copy_(self.prox_operator(x=p, lr=group['lr']))
        elif self.global_constraint:
            group = self.param_groups[0]
            param_list = [p for p in group['params'] if p.grad is not None]
            result = self.prox_operator(x=torch.cat([p.view(-1) for p in group['params'] if p.grad is not None]), lr=group['lr'])
            # Update parameters
            for p in param_list:
                numberOfElements = p.numel()
                p.copy_(result[:numberOfElements].view(p.shape))
                result = result[numberOfElements:]



class ProximalOperator:
    """Static class containing proximal operators, each function returns a function, i.e. the proximal operator."""
    @staticmethod
    def soft_thresholding(weight_decay=0.001):
        """Implements Soft-Thresholding aka Proximal SGD with L1 weight decay"""
        @torch.no_grad()
        def operator(x, lr):
            return torch.sign(x)*torch.nn.functional.relu(torch.abs(x)-lr*weight_decay)
        return operator

    @staticmethod
    def knorm_soft_thresholding(weight_decay=0.001, k=1):
        """Implements soft thresholding for regularization term wd*(||x||_1 -||x||_[k])"""
        @torch.no_grad()
        def operator(x, lr):
            indices = torch.topk(torch.abs(x.flatten()), k=k).indices
            z = torch.sign(x)*torch.nn.functional.relu(torch.abs(x)-lr*weight_decay)
            z.view(-1)[indices] = x.view(-1)[indices]
            return z
        return operator
