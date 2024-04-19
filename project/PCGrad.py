import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random
import logging


class PCGrad():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        logging.warning('PC Grad Initialized')
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''
        if objectives is None or len(objectives) == 0:
            raise ValueError("Objectives must be a non-empty list of loss tensors.")

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        logging.info("Resolving conflicting losses")
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        g_j_norm_squared = [torch.norm(g_j)**2 for g_j in grads]
        
        # Shuffle once before the loop
        random.shuffle(grads)
        
        for g_i in pc_grad:
            for idx, g_j in enumerate(grads):
                g_i_g_j = torch.dot(g_i, g_j)
                # If dot product is <0 the vectors are pointing in opposite direction. 
                if g_i_g_j < 0:
                    # In summary, this line of code adjusts the gradient g_i based on the dot product with g_j and projects it away from the direction of g_j 
                    # by subtracting a scaled version of g_j. This projection step helps to resolve conflicting gradients by reducing their alignment.
                    g_i -= (g_i_g_j) * g_j / g_j_norm_squared[idx]
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction == 'mean':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''
        
        grad, shape, has_grad = [], [], []
        with torch.no_grad():  # close gradient compute
            for group in self._optim.param_groups:
                for p in group['params']:
                    # if p.grad is None: continue
                    # tackle the multi-head scenario
                    if p.grad is None:
                        shape.append(p.shape)
                        grad.append(torch.zeros_like(p).to(p.device))
                        has_grad.append(torch.zeros_like(p).to(p.device))
                        continue
                    shape.append(p.grad.shape)
                    grad.append(p.grad.clone())
                    has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad



class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat)