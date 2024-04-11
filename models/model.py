import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import models
import math
from torchvision.models import resnet50
from models import prompted_vit

import hyptorch.nn as hypnn
from hyptorch.pmath import dist_matrix

class ResNet50(nn.Module):
    def __init__(self, pretrained=True, bn_freeze = True):
        super(ResNet50, self).__init__()

        self.model = resnet50(pretrained)
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)
        self.model.fc = nn.Identity()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x
        x = x.view(x.size(0), -1)
        
        return x

class CustomSequential(nn.Sequential):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def forward(self, input):
        for module in self:
            dim = len(input.shape)
            if isinstance(module, self.bn_types) and dim > 2:
                perm = list(range(dim - 1)); perm.insert(1, dim - 1)
                inv_perm = list(range(dim)) + [1]; inv_perm.pop(1)
                input = module(input.permute(*perm)).permute(*inv_perm)
            else:
                input = module(input)
        return input

def _build_norm(norm, hidden_dim, **kwargs):
    if norm == 'bn':
        norm = nn.BatchNorm1d(hidden_dim, **kwargs)
    elif norm == 'bn_noaffine':
        norm = nn.BatchNorm1d(hidden_dim, affine=False)
    elif norm == 'syncbn':
        norm = nn.SyncBatchNorm(hidden_dim, **kwargs)
    elif norm == 'csyncbn':
        norm = CSyncBatchNorm(hidden_dim, **kwargs)
    elif norm == 'psyncbn':
        norm =  PSyncBatchNorm(hidden_dim, **kwargs)
    elif norm == 'ln':
        norm = nn.LayerNorm(hidden_dim, **kwargs)
    else:
        assert norm is None, "unknown norm type {}".format(norm)
    return norm

def _build_act(act):
    if act == 'relu':
        act = nn.ReLU(inplace=False)
    elif act == 'gelu':
        act = nn.GELU()
    else:
        assert False, "unknown act type {}".format(act)
    return act
    
def _build_mlp(nlayers, in_dim=384, hidden_dim=2048, out_dim=128, act='gelu', bias=True, norm=None, output_norm=None):
    """
    build a mlp
    """
    
    norm_func = _build_norm(norm, hidden_dim)
    act_func = _build_act(act)
    layers = []
    for layer in range(nlayers):
        dim1 = in_dim if layer == 0 else hidden_dim
        dim2 = out_dim if layer == nlayers - 1 else hidden_dim

        layers.append(nn.Linear(dim1, dim2, bias=bias))

        if layer < nlayers - 1:
            if norm_func is not None:
                layers.append(norm_func)
            layers.append(act_func)
        elif output_norm is not None:
            output_norm_func = _build_norm(output_norm, out_dim)
            layers.append(act_func)
            layers.append(output_norm_func)
            
    mlp = CustomSequential(*layers)
            
    return mlp

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=1)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_model(args):
    prompt_config = (args.use_prompt, args.prompt_type, args.prefix_style, args.prompt_deep, args.component_size, args.prompt_length)
    adapter_config = (args.use_adapter, args.adapter_type, args.adapter_dim, args.adapter_droppath)
    lora_config = (args.use_lora, args.lora_dim, args.lora_droppath)

    body = timm.create_model(args.model, pretrained=True, pooling = args.pooling, 
                            prompt_config = prompt_config, adapter_config=adapter_config, lora_config=lora_config)

    if args.hyp_c > 0:
        last = hypnn.ToPoincare(c=args.hyp_c, ball_dim=args.emb, riemannian=False, clip_r=args.clip_r)
    else:
        last = NormLayer()
        
    bdim = body.embed_dim
        
    if not args.emb_mlp:
        last_layer = nn.Sequential(nn.Linear(bdim, args.emb), last)
    else:
        last_layer = nn.Sequential(_build_mlp(nlayers=3, in_dim=bdim, out_dim=args.emb), last)

    rm_head(body)

    if args.freeze:
        freeze_nlayer = len(body.blocks)
        print('freeze weights')
        freeze(body, freeze_nlayer)

    model = HeadSwitch(body, last_layer, last)

    model.cuda().train()

    return model

class HeadSwitch(nn.Module):
    def __init__(self, body, last_layer, norm):
        super(HeadSwitch, self).__init__()
        self.body = body
        self.last_layer = last_layer
        self.norm = norm
        
    def forward(self, x, skip_head=False, q=None):
        x, x_query, loss = self.body(x, q)
        if type(x) == tuple:
            x = x[0]
        
        if skip_head:
            x = self.norm(x)
        else:
            x = self.last_layer(x)
        
        if self.training:
            return x, loss
        else:
            return x

class NormLayer(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

def freeze(model, num_block):
    def fr(m):
        for param in m.parameters():
            param.requires_grad = False

    fr(model.patch_embed)
    model.pos_embed.requires_grad = False
    model.cls_token.requires_grad = False

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            m.eval()
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(False)

    if hasattr(model, 'dist_token'):
        model.dist_token.requires_grad = False
    for i in range(num_block):
        fr(model.blocks[i])

def rm_head(m):
    names = set(x[0] for x in m.named_children())
    target = {"head", "fc", "head_dist"}
    for x in names & target:
        m.add_module(x, nn.Identity())
