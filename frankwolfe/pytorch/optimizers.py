# ===========================================================================
# Project:      StochasticFrankWolfe 2020 / IOL Lab @ ZIB
# File:         pytorch/optimizers.py
# Description:  Pytorch implementation of Stochastic Frank Wolfe, AdaGradSFW and SGD with projection
# ===========================================================================
import torch


# TODO: How do we handle unconstrained parameters?
# TODO: Is modifying a large tensor through indexing when dealing with global constraints a good idea?
# -> Max: will check this

class SFW(torch.optim.Optimizer):
    """Stochastic Frank Wolfe Algorithm
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        learning_rate (float): learning rate between 0.0 and 1.0
        rescale (string or None): Type of learning_rate rescaling. Must be 'diameter', 'gradient' or None
        momentum (float): momentum factor, 0 for no momentum
    """

    def __init__(self, params, learning_rate=0.1, rescale='diameter', momentum=0, dampening=0, global_constraint=None):
        momentum = momentum or 0
        dampening = dampening or 0
        if not (0.0 <= learning_rate <= 1.0):
            raise ValueError("Invalid learning rate: {}".format(learning_rate))
        if not (0.0 <= momentum <= 1.0):
            raise ValueError("Momentum must be between 0 and 1.")
        if not (0.0 <= dampening <= 1.0):
            raise ValueError("Dampening must be between 0 and 1.")
        if rescale == 'None': rescale = None
        if not (rescale in ['diameter', 'gradient', None]):
            raise ValueError("Rescale type must be either 'diameter', 'gradient' or None.")

        self.rescale = rescale
        self.global_constraint = global_constraint  # If not None, this points to the global constraint instance
        assert not (self.global_constraint and len(
            self.param_groups) > 1), "This does not work for multiple param_groups yet."

        defaults = dict(lr=learning_rate, momentum=momentum, dampening=dampening)
        super(SFW, self).__init__(params, defaults)

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
                v = p.constraint.lmo(d_p)  # LMO optimal solution

                # Determine learning rate rescaling factor
                if self.rescale == 'diameter':
                    # Rescale lr by diameter
                    factor = 1. / p.constraint.get_diameter()
                elif self.rescale == 'gradient':
                    # Rescale lr by gradient
                    factor = torch.norm(d_p, p=2) / torch.norm(p - v, p=2)
                else:
                    # No rescaling
                    factor = 1
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
        v = self.global_constraint.lmo(grad_vec)

        # Determine learning rate rescaling factor
        if self.rescale == 'diameter':
            # Rescale lr by diameter
            factor = 1. / self.global_constraint.get_diameter()
        elif self.rescale == 'gradient':
            # Rescale lr by gradient
            factor = torch.norm(grad_vec, p=2) / torch.norm(torch.cat([p.view(-1) for p in param_list]) - v, p=2)
        else:
            # No rescaling
            factor = 1
        lr = max(0.0, min(factor * group['lr'], 1.0))  # Clamp between [0, 1]

        # Update parameters
        for p in param_list:
            numberOfElements = p.numel()
            p.mul_(1 - lr)
            p.add_(v[:numberOfElements].view(p.shape), alpha=lr)
            v = v[numberOfElements:]


class SGD(torch.optim.Optimizer):
    """Modified SGD which allows projection via Constraint class"""

    def __init__(self, params, lr=0.1, momentum=0, dampening=0, nesterov=False,
                 weight_decay=0, weight_decay_ord=2, global_constraint=None):
        momentum = momentum or 0
        dampening = dampening or 0
        weight_decay = weight_decay or 0
        weight_decay_ord = float(weight_decay_ord)  # Catch string 'inf' input
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

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, weight_decay_ord=weight_decay_ord, nesterov=nesterov)
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

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

        # For Linf regularization, we need to compute the maximal element out of all parameters
        for group in self.param_groups:
            if group['weight_decay'] > 0 and group['weight_decay_ord'] == float('inf'):
                group['maxParam'] = max(float(torch.max(torch.abs(p)))
                                        for p in group['params'] if p.grad is not None)

        for group in self.param_groups:
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

                p.add_(d_p, alpha=-group['lr'])

                # Project if necessary
                if not self.global_constraint:  # We have to project afterwards
                    if hasattr(p, 'constraint'):
                        if not p.constraint.is_unconstrained():
                            p.copy_(p.constraint.euclidean_project(p))

        if self.global_constraint:
            # Project entire gradient vector
            assert len(self.param_groups) == 1, "This does not work for multiple param_groups yet."
            param_list = [p for p in self.param_groups[0]['params'] if p.grad is not None]
            p_proj = self.global_constraint.euclidean_project(torch.cat([p.view(-1) for p in param_list]))
            for p in param_list:
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
        learning_rate (float, optional): learning rate (default: 1e-2)
        delta (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        momentum (float, optional): momentum factor
    """

    def __init__(self, params, inner_steps=2, learning_rate=1e-2, delta=1e-8, momentum=0.9):
        if not 0.0 <= learning_rate:
            raise ValueError("Invalid learning rate: {}".format(learning_rate))
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("Momentum must be between [0, 1].")
        if not 0.0 <= delta:
            raise ValueError("Invalid delta value: {}".format(delta))
        if not int(inner_steps) == inner_steps and not 0.0 <= inner_steps:
            raise ValueError("Number of inner iterations needs to be a positive integer: {}".format(inner_steps))

        self.K = inner_steps

        defaults = dict(lr=learning_rate, delta=delta, momentum=momentum)
        super(AdaGradSFW, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['sum'] = torch.zeros_like(p, memory_format=torch.preserve_format)

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
