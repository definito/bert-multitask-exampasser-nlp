from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # 1- Update first and second moments of the gradients
                # 2- Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3- Update parameters (p.data).
                # 4- After that main gradient-based update, update again using weight decay
                #    (incorporating the learning rate again).
                
                ### TODO
                if len(state) == 0:
                    # Initialize timestep
                    state['step'] = 0
                    # Initialize initial 1st moment vector
                    state['m'] = torch.zeros_like(p.data)
                    # Initialize initial 2nd moment vector
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                correct_bias = group['correct_bias']
                # print("------")
                # print(alpha)
                # print(beta1,beta2)
                # print(eps)
                # print(weight_decay)
                # print(correct_bias)
                

                # t <- t + 1
                state['step'] += 1

                # Update first and second moments of the gradients
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Apply bias correction
                if correct_bias:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                else:
                    bias_correction1 = 1
                    bias_correction2 = 1

                # Compute the `alpha_t`
                alpha_t = alpha * (math.sqrt(bias_correction2) / bias_correction1)

                # Update parameters (p.data).
                # ! applied RGD here, not for assignment 1---
                # print("using RGD")
                # p.data -= alpha_t * torch.exp(m / (v.sqrt().add_(eps)))
                # finished RGD ---
                p.data -= alpha_t * m / (v.sqrt().add_(eps))
                
                # Apply weight decay
                if weight_decay != 0:
                    # p.data.add_(p.data, alpha = -alpha * weight_decay)
                    p.data = p.data -alpha * weight_decay * p.data
                    # print(alpha)
        return loss
