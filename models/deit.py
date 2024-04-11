""" DeiT - Data-efficient Image Transformers

DeiT model defs and weights from https://github.com/facebookresearch/deit, original copyright below

paper: `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

paper: `DeiT III: Revenge of the ViT` - https://arxiv.org/abs/2204.07118

Modifications copyright 2021, Ross Wightman
"""
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from functools import partial

import torch
from torch import nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.models.vision_transformer import VisionTransformer, trunc_normal_, checkpoint_filter_fn

from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg, checkpoint_seq
from timm.models.registry import register_model
from models.vision_transformer import VisionTransformer, trunc_normal_, checkpoint_filter_fn

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # deit models (FB weights)
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'),
    'deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        input_size=(3, 384, 384), crop_pct=1.0),

    'deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        classifier=('head', 'head_dist')),
    'deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), crop_pct=1.0,
        classifier=('head', 'head_dist')),

    'deit3_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_224_1k.pth'),
    'deit3_small_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_384_1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_medium_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_medium_224_1k.pth'),
    'deit3_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_224_1k.pth'),
    'deit3_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_384_1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_large_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_224_1k.pth'),
    'deit3_large_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_384_1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_huge_patch14_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_huge_224_1k.pth'),

    'deit3_small_patch16_224_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_224_21k.pth',
        crop_pct=1.0),
    'deit3_small_patch16_384_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_384_21k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_medium_patch16_224_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_medium_224_21k.pth',
        crop_pct=1.0),
    'deit3_base_patch16_224_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_224_21k.pth',
        crop_pct=1.0),
    'deit3_base_patch16_384_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_384_21k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_large_patch16_224_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_224_21k.pth',
        crop_pct=1.0),
    'deit3_large_patch16_384_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_384_21k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_huge_patch14_224_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_huge_224_21k_v1.pth',
        crop_pct=1.0),
}


class VisionTransformerDistilled(VisionTransformer):
    """ Vision Transformer w/ Distillation Token and Head

    Distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, *args, **kwargs):
        weight_init = kwargs.pop('weight_init', '')
        super().__init__(*args, **kwargs, weight_init='skip')
        assert self.global_pool in ('token',)       

        self.num_prefix_tokens = 2        
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))      
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens + self.prompt_length, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.distilled_training = False  # must set this True to train w/ distillation token

        self.init_weights(weight_init)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    def init_weights(self, mode=''):
        trunc_normal_(self.dist_token, std=.02)
        super().init_weights(mode=mode)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed|dist_token',
            blocks=[
                (r'^blocks\.(\d+)', None),
                (r'^norm', (99999,))]  # final norm w/ last block
        )

    @torch.jit.ignore
    def get_classifier(self):
        return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def forward_features(self, x, q=None):
        loss = torch.zeros(1).cuda()
        B = x.shape[0]
        x = self.patch_embed(x)
        
        x_query = (x.mean(1).detach() + x.max(1)[0].detach())/2

        if self.prompt is not None and not self.prefix_style:
            prompt_tokens, prompt_loss = self.prompt(x_query=x_query, l=-1)
            prompt_tokens = prompt_tokens.expand(B, -1, -1)
            x = torch.cat((prompt_tokens, x), dim=1)
            loss += prompt_loss

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        p_list, prompt_loss = None, 0
        for i, blk in enumerate(self.blocks):
            prompt_tokens, prefix_tokens, prompt_loss = None, None, 0
            if self.prompt is not None and self.prompt_deep:
                if i in self.prompt.e_layers:
                    if self.prefix_style:
                        prefix_tokens, prompt_loss = self.prompt(x_query=x_query, l=i)
                    else:
                        prompt_tokens, prompt_loss = self.prompt(x_query=x_query, l=i)
                        prompt_tokens = prompt_tokens.expand(B, -1, -1)
                        x = torch.cat((x[:, :1, :], prompt_tokens, x[:, (1+self.prompt_length):]), dim=1)
                    loss += prompt_loss
                                                        
            if self.adapter is None:
                x = blk.forward_attn(x, prefix=prefix_tokens, lora=self.lora, l=i)
                x = blk.forward_ffn(x)
            else:               
                x_attn = blk.forward_attn(x, prefix=prefix_tokens, lora=self.lora, l=i)
                x_adapt1 = self.adapter(blk.norm1(x), l=i, f=0)
                x_attn = x_attn + x_adapt1
                
                x_ffn = blk.forward_ffn(x_attn)
                x_adapt2 = self.adapter(blk.norm2(x_attn), l=i, f=1)
                x = x_ffn + x_adapt2

        x = self.norm(x)
        return x, x_query, loss


def _create_deit(variant, pretrained=False, distilled=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    model_cls = VisionTransformerDistilled if distilled else VisionTransformer
    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    model = build_model_with_cfg(
        model_cls, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=partial(checkpoint_filter_fn, adapt_layer_scale=True),
        pretrained_strict = False,
        **kwargs)
    return model


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_deit('deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_deit('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_deit('deit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    """ DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_deit('deit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_deit(
        'deit_tiny_distilled_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_deit(
        'deit_small_distilled_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_deit(
        'deit_base_distilled_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_deit(
        'deit_base_distilled_patch16_384', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def deit3_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-3 small model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_small_patch16_384(pretrained=False, **kwargs):
    """ DeiT-3 small model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_small_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_medium_patch16_224(pretrained=False, **kwargs):
    """ DeiT-3 medium model @ 224x224 (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_medium_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_base_patch16_224(pretrained=False, **kwargs):
    """ DeiT-3 base model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_base_patch16_384(pretrained=False, **kwargs):
    """ DeiT-3 base model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_large_patch16_224(pretrained=False, **kwargs):
    """ DeiT-3 large model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_large_patch16_384(pretrained=False, **kwargs):
    """ DeiT-3 large model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_huge_patch14_224(pretrained=False, **kwargs):
    """ DeiT-3 base model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_huge_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_small_patch16_224_in21ft1k(pretrained=False, **kwargs):
    """ DeiT-3 small model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-21k pretrained weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_small_patch16_224_in21ft1k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_small_patch16_384_in21ft1k(pretrained=False, **kwargs):
    """ DeiT-3 small model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-21k pretrained weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_small_patch16_384_in21ft1k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_medium_patch16_224_in21ft1k(pretrained=False, **kwargs):
    """ DeiT-3 medium model @ 224x224 (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_medium_patch16_224_in21ft1k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_base_patch16_224_in21ft1k(pretrained=False, **kwargs):
    """ DeiT-3 base model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-21k pretrained weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_base_patch16_224_in21ft1k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_base_patch16_384_in21ft1k(pretrained=False, **kwargs):
    """ DeiT-3 base model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-21k pretrained weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_base_patch16_384_in21ft1k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_large_patch16_224_in21ft1k(pretrained=False, **kwargs):
    """ DeiT-3 large model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-21k pretrained weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_large_patch16_224_in21ft1k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_large_patch16_384_in21ft1k(pretrained=False, **kwargs):
    """ DeiT-3 large model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-21k pretrained weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_large_patch16_384_in21ft1k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_huge_patch14_224_in21ft1k(pretrained=False, **kwargs):
    """ DeiT-3 base model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-21k pretrained weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_huge_patch14_224_in21ft1k', pretrained=pretrained, **model_kwargs)
    return model