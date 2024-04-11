import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import math
from timm.models.layers import DropPath
import random
from functools import reduce
from operator import mul

def ortho_penalty(t):
    return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean() * 1e-6

def cossim_penalty(t):
    t = F.normalize(t)
    return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean() * 1e-6

def tensor_init(p, init, gain=1, std=1, a=1, b=1):
    if init == 'ortho':
        nn.init.orthogonal_(p)
    elif init == 'uniform':
        nn.init.uniform_(p, a=a, b=b)
    elif init == 'normal':
        nn.init.normal_(p, std=std)
    elif init == 'zero':
        nn.init.zeros_(p)
    elif init == 'he_uniform':
        nn.init.kaiming_uniform_(p, a=a)
    elif init == 'he_normal':
        nn.init.kaiming_normal_(p, a=a)
    elif init == 'xavier_uniform':
        nn.init.xavier_uniform_(p, gain=gain)
    elif init == 'xavier_normal':
        nn.init.xavier_normal_(p, gain=gain)
    elif init == 'trunc_normal':
        nn.init.trunc_normal_(p, std=std)
    else:
        assert NotImplementedError

def tensor_prompt(x, y=None, z=None, w=None, init='xavier_uniform', gain=1, std=1, a=1, b=1):
    if y is None:
        p = torch.nn.Parameter(torch.FloatTensor(x), requires_grad=True)
    elif z is None:
        p = torch.nn.Parameter(torch.FloatTensor(x,y), requires_grad=True)
    elif w is None:
        p = torch.nn.Parameter(torch.FloatTensor(x,y,z), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(x,y,z,w), requires_grad=True)

    if p.dim() > 2:
        tensor_init(p[0], init, gain=gain, std=std, a=a, b=b)
        for i in range(1, x): p.data[i] = p.data[0]
    else:
        tensor_init(p, init)
    
    return p

class PromptPool(nn.Module):
    def __init__(self, prompt_param, embed_dim, key_dim=768, prompt_deep=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.key_d = key_dim
        self._init_smart(embed_dim, prompt_param)
        self.prompt_deep = prompt_deep
        self.bn = nn.BatchNorm1d(self.e_pool_size, affine=False)
        
        if not self.prompt_deep:
            self.e_layers = [-1]

        # e prompt init
        for e in self.e_layers:
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, embed_dim, init='ortho')
            k = tensor_prompt(self.e_pool_size, self.key_d, init='ortho')
            a = tensor_prompt(self.e_pool_size, self.key_d, init='ortho')

            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    def _init_smart(self, embed_dim, prompt_param):

        # prompt basic param
        self.e_pool_size = prompt_param[0]
        self.e_p_length = prompt_param[1]

        # prompt locations
        self.e_layers = list(range(0,12))
        
        # ablations
        self.attention = True

    @torch.no_grad()
    def update_center(self, output, momentum=0.9):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(output, dim=0, keepdim=True)

        # ema update
        self.center = self.center * momentum + batch_center * (1 - momentum)

    def compute_aq_k(self, x_query, l=-1):
        K = getattr(self,f'e_k_{l}')
        p = getattr(self,f'e_p_{l}')

        if self.attention:
            ##########
            # with attention and cosine sim
            ##########
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            A = getattr(self,f'e_a_{l}')
            a_query = torch.einsum('bd,kd->bkd', x_query, F.softmax(A, dim=1))
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = F.normalize(K, dim=1)
            q = F.normalize(a_query, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            self.bn.eval()
            aq_k = self.bn(aq_k)
            
        else:
            ##########
            # cosine sim
            ##########
            # # (b x 1 x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = F.normalize(K, dim=1)
            q = F.normalize(x_query, dim=1)
            aq_k = torch.einsum('bd,kd->bk', q, n_K)

        return aq_k

    def forward(self, x_query, l):
        # e prompts
        e_valid = False
        loss = 0
                
        if l in self.e_layers:
            e_valid = True
            B, C = x_query.shape

            K = getattr(self,f'e_k_{l}')
            p = getattr(self,f'e_p_{l}')

            aq_k = self.compute_aq_k(x_query)
            # (b x 1 x k x 1) * [1 x k x plen x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length/2)
            
            if self.training:
                K = getattr(self,f'e_k_{l}')
                A = getattr(self,f'e_a_{l}')
                p = getattr(self,f'e_p_{l}')
                
                loss += ortho_penalty(K)
                loss += ortho_penalty(A)
                loss += ortho_penalty(p.flatten(1,2))

        # combine prompts for prefix tuning
        if e_valid:
            p_return = P_
        else:
            p_return = None

        # return
        return p_return, loss
    

class VPT(nn.Module):
    def __init__(self, prompt_param, embed_dim, key_dim=768, prompt_deep=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.key_d = key_dim
        self.e_pool_size = prompt_param[0]
        self.e_p_length = prompt_param[1]
        self.prompt_deep = prompt_deep
        
        self.e_layers = [-1]
        if self.prompt_deep:
            self.e_layers = self.e_layers + list(range(0,12))

        # e prompt init
        for e in self.e_layers:
            e_l = self.e_p_length
            val = math.sqrt(6. / float(3 * 196 + embed_dim))  # noqa

            # xavier_uniform initialization
            p = tensor_prompt(self.e_p_length, embed_dim)
            nn.init.uniform_(p, -val, val)

            setattr(self, f'e_p_{e}', p)

    def forward(self, x_query, l):
        # e prompts
        e_valid = False
        loss = 0
                
        if l in self.e_layers:
            e_valid = True
            p_ = getattr(self,f'e_p_{l}')

        # combine prompts for prefix tuning
        if e_valid:
            p_return = p_
        else:
            p_return = None

        # return
        return p_return, loss

    
class StochasticAdapter(nn.Module):
    def __init__(self, embed_dim, low_dim=64, init_value=0.1, scale_train=False, drop_path=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.low_dim = low_dim
        self.scale_train = scale_train
        self.adapt_layers = list(range(0,12))
        self.non_linear = nn.ReLU()
        self.drop_path = drop_path
        self.d = nn.ParameterList([tensor_prompt(2, self.embed_dim, self.low_dim, init='he_uniform') for i in self.adapt_layers])
        self.u = nn.ParameterList([tensor_prompt(2, self.low_dim, self.embed_dim, init='zero') for i in self.adapt_layers])
        self.s = nn.ParameterList([nn.Parameter(init_value * torch.ones(2), requires_grad = self.scale_train) for i in self.adapt_layers])
            
        self.drop_paths1 = nn.ModuleList([DropPath(drop_path) for i in self.adapt_layers])
        self.drop_paths2 = nn.ModuleList([DropPath(drop_path) for i in self.adapt_layers])
                    
    def forward(self, x, l, f, aq_k=None):
        loss = 0
        
        if l in self.adapt_layers:
            D_ = self.d[l][f]
            U_ = self.u[l][f]
            S = self.s[l][f]
            x = torch.einsum('bnd,ds->bns', x, D_)
            x = self.non_linear(x)
            x = torch.einsum('bns,sd->bnd', x, U_)
            x = x.mul_(S)
            
            if f == 0:
                x = self.drop_paths1[l](x)
            else:
                x = self.drop_paths2[l](x)

        else:
            x = 0
            
        return x

class AdaptFormer(nn.Module):
    def __init__(self, embed_dim, low_dim=64, init_value=0.1, scale_train=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.low_dim = low_dim
        self.scale_train = scale_train
        self.adapt_layers = list(range(0,12))
        self.non_linear = nn.ReLU()
        self.d = nn.ModuleList([nn.Linear(self.embed_dim, self.low_dim) for i in self.adapt_layers])
        self.u = nn.ModuleList([nn.Linear(self.low_dim, self.embed_dim) for i in self.adapt_layers])
        self.s = nn.ParameterList([nn.Parameter(init_value * torch.ones(1), requires_grad = self.scale_train) for i in self.adapt_layers])
                    
        for i in self.adapt_layers: 
            nn.init.kaiming_uniform_(self.d[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.u[i].weight)
            nn.init.zeros_(self.d[i].bias)
            nn.init.zeros_(self.u[i].bias)

    def forward(self, x, l, f=0, aq_k=None):
        if l in self.adapt_layers and f == 1:
            x = self.d[l](x)
            x = self.non_linear(x)
            x = self.u[l](x)
            x = x.mul_(self.s[l])
        else:
            x = 0
            
        return x

class Lora(nn.Module):
    def __init__(self, embed_dim, low_dim=64, init_value=32, scale_train=False, drop_path=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.low_dim = low_dim
        self.scale_train = scale_train
        self.lora_layers = list(range(0,12))
        self.drop_path = drop_path
        self.d_init = 'he_uniform'
        self.u_init = 'zero'

        self.d = nn.ParameterList([tensor_prompt(2, self.embed_dim, self.low_dim, init=self.d_init) for i in self.lora_layers])
        self.u = nn.ParameterList([tensor_prompt(2, self.low_dim, self.embed_dim, init=self.d_init) for i in self.lora_layers])

        r = embed_dim / low_dim
        self.s = nn.ParameterList([nn.Parameter(init_value * torch.ones(2), requires_grad = self.scale_train) for i in self.lora_layers]) / r
                    
        self.drop_paths1 = nn.ModuleList([DropPath(drop_path) for i in self.lora_layers])
        self.drop_paths2 = nn.ModuleList([DropPath(drop_path) for i in self.lora_layers])
                    
    def forward(self, x, l, f):
        if l in self.lora_layers:
            D_ = self.d[l][f]
            U_ = self.u[l][f]
            S = self.s[l][f]
            x = torch.einsum('bnd,ds->bns', x, D_)
            x = torch.einsum('bns,sd->bnd', x, U_)
            x = x.mul_(S)
            
            if f == 0:
                x = self.drop_paths1[l](x)
            else:
                x = self.drop_paths2[l](x)
        return x