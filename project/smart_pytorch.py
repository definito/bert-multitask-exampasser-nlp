from typing import Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from itertools import count 

def check_device(*tensors):
    for tensor in tensors:
        if tensor is not None and not tensor.is_cuda:
            print(f"Warning: Tensor {tensor} is not on GPU, it's on {tensor.device}")


def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d

def inf_norm(x):
    return torch.norm(x, p=float('inf'), dim=-1, keepdim=True)

class SMARTLoss(nn.Module):
    
    def __init__(
        self,
        eval_fn: Callable,
        loss_fn: Callable,
        loss_last_fn: Callable = None, 
        norm_fn: Callable = inf_norm, 
        num_steps: int = 1,
        step_size: float = 1e-3, 
        epsilon: float = 1e-6,
        noise_var: float = 1e-5
    ) -> None:
        super().__init__()
        self.eval_fn = eval_fn 
        self.loss_fn = loss_fn
        self.loss_last_fn = default(loss_last_fn, loss_fn)
        self.norm_fn = norm_fn
        self.num_steps = num_steps 
        self.step_size = step_size
        self.epsilon = epsilon 
        self.noise_var = noise_var
        
    def forward(self, embed: Tensor, state: Tensor) -> Tensor:
        noise = torch.randn_like(embed, requires_grad=True) * self.noise_var

        # Indefinite loop with counter 
        for i in count():
            # Compute perturbed embed and states 
            embed_perturbed = embed + noise 
            state_perturbed = self.eval_fn(embed_perturbed)
            # Return final loss if last step (undetached state)
            if i == self.num_steps: 
                return self.loss_last_fn(state_perturbed, state) 
            # Compute perturbation loss (detached state)
            loss = self.loss_fn(state_perturbed, state.detach())
            # Compute noise gradient ∂loss/∂noise
            noise_gradient, = torch.autograd.grad(loss, noise)
            # Move noise towards gradient to change state as much as possible 
            step = noise + self.step_size * noise_gradient 
            # Normalize new noise step into norm induced ball 
            step_norm = self.norm_fn(step)
            noise = step / (step_norm + self.epsilon)
            # Reset noise gradients for next step
            noise = noise.detach().requires_grad_()
            check_device(embed, noise, state_perturbed)
            
class SMARTLossSPair(nn.Module):
    
    def __init__(
        self,
        eval_fn: Callable,
        loss_fn: Callable,
        loss_last_fn: Callable = None, 
        norm_fn: Callable = inf_norm, 
        num_steps: int = 1,
        step_size: float = 1e-3, 
        epsilon: float = 1e-6,
        noise_var: float = 1e-5
    ) -> None:
        super().__init__()
        self.eval_fn = eval_fn 
        self.loss_fn = loss_fn
        self.loss_last_fn = default(loss_last_fn, loss_fn)
        self.norm_fn = norm_fn
        self.num_steps = num_steps 
        self.step_size = step_size
        self.epsilon = epsilon 
        self.noise_var = noise_var
        
    def forward(self, embed_1: Tensor, embed_2: Tensor, state: Tensor) -> Tensor:
        noise_1 = torch.randn_like(embed_1, requires_grad=True) * self.noise_var
        noise_2 = torch.randn_like(embed_2, requires_grad=True) * self.noise_var
        
        # Indefinite loop with counter 
        for i in count():
            # Compute perturbed embed and states 
            embed_perturbed_1 = embed_1 + noise_1 
            embed_perturbed_2 = embed_2 + noise_2 
            state_perturbed = self.eval_fn(embed_perturbed_1, embed_perturbed_2)
            # Return final loss if last step (undetached state)
            if i == self.num_steps: 
                return self.loss_last_fn(state_perturbed, state) 
            # Compute perturbation loss (detached state)
            loss = self.loss_fn(state_perturbed, state.detach())
            # Compute noise gradient ∂loss/∂noise
            noise_gradient_1, noise_gradient_2 = torch.autograd.grad(loss, [noise_1, noise_2])
            # Move noise towards gradient to change state as much as possible 
            step_1 = noise_1 + self.step_size * noise_gradient_1
            step_2 = noise_2 + self.step_size * noise_gradient_2
            # Normalize new noise step into norm induced ball 
            step_norm_1 = self.norm_fn(step_1)
            step_norm_2 = self.norm_fn(step_2)

            noise_1 = step_1 / (step_norm_1 + self.epsilon)
            noise_2 = step_2 / (step_norm_2 + self.epsilon)

            # Reset noise gradients for next step
            noise_1 = noise_1.detach().requires_grad_()
            noise_2 = noise_2.detach().requires_grad_()