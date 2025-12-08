"""
Advanced optimizers: Lion, Ranger, and others
Implements ADVICE 07/12 - Modern Optimizers
"""
import torch
from torch.optim.optimizer import Optimizer
import math


class Lion(Optimizer):
    """
    Lion optimizer (EvoLved Sign Momentum)
    Paper: https://arxiv.org/abs/2302.06675
    
    Lion uses sign of gradients and is more memory efficient than Adam.
    Typically use learning_rate = lr_adam / 3, weight_decay = wd_adam * 3
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                
                # Momentum update
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss


class Ranger(Optimizer):
    """
    Ranger optimizer = RAdam + Lookahead
    Combines rectified adaptive learning rates with lookahead mechanism
    """
    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, betas=(0.9, 0.999), 
                 eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha parameter: {alpha}")
        
        defaults = dict(lr=lr, alpha=alpha, k=k, betas=betas, eps=eps, 
                       weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        # Lookahead slow weights
        self.buffer = [[p.clone().detach() for p in group['params']] 
                       for group in self.param_groups]
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Ranger does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # RAdam variance rectification
                rho_inf = 2 / (1 - beta2) - 1
                rho = rho_inf - 2 * state['step'] * (beta2 ** state['step']) / bias_correction2
                
                if rho > 4:
                    # Adaptive learning rate
                    rect = math.sqrt(
                        (rho - 4) * (rho - 2) * rho_inf / 
                        ((rho_inf - 4) * (rho_inf - 2) * rho)
                    )
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    step_size = group['lr'] * rect / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    # Use SGD-like step when variance not reliable
                    step_size = group['lr'] / bias_correction1
                    p.add_(exp_avg, alpha=-step_size)
                
                # Weight decay
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])
        
        # Lookahead step
        for group, buffer_group in zip(self.param_groups, self.buffer):
            k = group['k']
            alpha = group['alpha']
            
            if self.state[group['params'][0]]['step'] % k == 0:
                for p, slow_p in zip(group['params'], buffer_group):
                    slow_p.add_(p - slow_p, alpha=alpha)
                    p.copy_(slow_p)
        
        return loss


def get_optimizer(name, model_parameters, lr, weight_decay):
    """
    Factory function to get optimizer by name
    
    Args:
        name: 'adamw', 'lion', 'ranger'
        model_parameters: model.parameters()
        lr: learning rate
        weight_decay: weight decay factor
    
    Returns:
        Optimizer instance
    """
    optimizers = {
        'adamw': lambda: torch.optim.AdamW(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay
        ),
        'lion': lambda: Lion(
            model_parameters,
            lr=lr / 3,  # Lion typically uses lower LR
            weight_decay=weight_decay * 3  # But higher weight decay
        ),
        'ranger': lambda: Ranger(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay
        )
    }
    
    if name not in optimizers:
        raise ValueError(f"Optimizer {name} not supported. Choose from {list(optimizers.keys())}")
    
    return optimizers[name]()
