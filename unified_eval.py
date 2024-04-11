
import argparse
import os
import sys
import datetime
import time
import math
import json
import numpy as np
import utils
import wandb
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from pathlib import Path
from tqdm import trange
import wandb
from functools import partial
import PIL
import multiprocessing
from losses import *

from sampler import UniqueClassSampler, BalancedSampler, ClassMiningSampler, DataClassBalancedSampler
from helpers import get_emb, evaluate_all
from dataset import All_dataset

import wandb
from functools import partial
import PIL
import multiprocessing
from losses import *
from models.model import init_model
        

import argparse
import os
import sys
import datetime
import time
import math
import json
import numpy as np
import utils
import wandb
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from pathlib import Path
from tqdm import trange
import wandb
from functools import partial
import PIL
import multiprocessing
from losses import *

from sampler import UniqueClassSampler, BalancedSampler, ClassMiningSampler, DataClassBalancedSampler
from helpers import get_emb, evaluate_all
from dataset import All_dataset

import wandb
from functools import partial
import PIL
import multiprocessing
from losses import *
from models.model import init_model
        
def get_args_parser():
    parser = argparse.ArgumentParser('iBOT', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='vit_small_patch16_224', type=str,
        choices=['resnet50', 'deit_small_distilled_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224', 'dino_vits16', 
                 'vit_small_patch16_224_dino', 'vit_base_patch16_clip_224'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--image_size', type=int, default=224, help="""Size of Global Image""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--window_size', default=7, type=int, help="""Size of window - default 7.
        This config is only valid for Swin Transofmer and is ignoired for vanilla ViT architectures.""")

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--clip_grad', type=float, default=0.1, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size', default=360, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=1e-5, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=0, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr_scale', type=float, default=0.1, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'adamp', 'adam', 'sgd'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--load_from', default=None, help="""Path to load checkpoints to resume training.""")
    parser.add_argument('--pooling', default = 'cls', type=str, choices=['cls', 'avg', 'max', 'avgmax', 'gem'], help = 'ViT Pooling')

    # Augementation parameters
    parser.add_argument('--global_crops_number', type=int, default=1, help="""Number of global
        views to generate. Default is to use two global crops. """)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.14, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--lr_decay', default = None, type=str, help = 'Learning decay step setting')
    
    # Metric Learning parameters
    parser.add_argument('--t', type=float, default=0.1)
    parser.add_argument('--IPC', type=int, default=0)
    parser.add_argument('--norm_freeze', type=utils.bool_flag, default=True)
    parser.add_argument('--freeze', type=utils.bool_flag, default=True)
    parser.add_argument('--resize_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--save_emb', type=utils.bool_flag, default=False)
    parser.add_argument('--best_recall', type=int, default=0)
    parser.add_argument('--loss', default='PA', type=str)
    parser.add_argument('--topk', default=30, type=int)
    parser.add_argument('--sampler', default='Unique', type=str)
    
    # Proxy Anchor parameters
    parser.add_argument('--alpha', type=float, default=32)
    parser.add_argument('--mrg', type=float, default=0.1)
    
    parser.add_argument('--memory', type=utils.bool_flag, default=False)

    # Misc
    parser.add_argument('--dataset', default='All', nargs='+', type=str, 
                        choices=["CUB", "Cars", "SOP", "Inshop", "NAbird", "Dogs", "Flowers", "Aircraft", "Standard", "All"], help='Please specify dataset to train')
    parser.add_argument('--test_dataset', default='All', nargs='+', type=str, 
                        choices=["CUB", "Cars", "SOP", "Inshop", "NAbird", "Dogs", "Flowers", "Aircraft", "Standard", "All"], help='Please specify dataset to train')
    parser.add_argument('--data_path', default='./', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    
    parser.add_argument('--output_dir', default="./logs/", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--run_name', default="", type=str, help='Wandb run name')
    parser.add_argument('--saveckp_freq', default=40, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--wandb_save', default=True, type=utils.bool_flag, help='Save and upload checkpoint to wandb')
    parser.add_argument('--wandb', default=True, type=utils.bool_flag, help='wandb flag')
    parser.add_argument('--wandb_entity', default=None, type=str, help='Wandb Entity')
    parser.add_argument('--eval_freq', default=1,type=int, help='k-NN Evaluation and Linear Probing Evaluation for every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=None, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    
    ## Few-shot Experiment
    parser.add_argument('--num_shot', default=None, type=int)
    parser.add_argument('--fewshot_dir', default="./few_shot/", type=str, help='few-shot data pkl file')
    parser.add_argument('--skip_head', default=False, type=utils.bool_flag, help='Eval no embedding layer')
    parser.add_argument('--pretrained_eval', default=False, type=utils.bool_flag, help='Eval no embedding layer')

    ## prompts
    parser.add_argument('--use_prefix', default=False, type=utils.bool_flag, help="""use prefix or prompt""")
    parser.add_argument('--prompt_lambda', default=0, type=float, help="""prompt_lambda.""")
    parser.add_argument('--component_size', default=0, type=int, help="""component_size.""")
    parser.add_argument('--prompt_length', default=0, type=int, help="""prompt_length.""")
    parser.add_argument('--prompt_type', default=None, type=str, help="""prompt_type.""")
    
    ## adapters
    parser.add_argument('--num_adapter', default=0, type=int, help="""Number of Parallel Adapter""")
    parser.add_argument('--adapter_dim', default=0, type=int, help="""Parallel Adapter dimension""")
    parser.add_argument('--use_adapter', default=False, type=utils.bool_flag, help="""use parallel adapter""")
    parser.add_argument('--adapter_droppath', default=0, type=float, help="""use parallel adapter drop path""")
    parser.add_argument('--adapter_gate', default=False, type=utils.bool_flag, help="""use parallel adapter drop path""")

    ## embedding layer
    parser.add_argument('--num_emb', default=1, type=int, help="""Number of Parallel Adapter""")
    parser.add_argument('--emb', default=128, type=int, help="""Dimensionality of output for [CLS] token.""")
    
    ## Hyperbolic Vision Trnasformer
    parser.add_argument('--hyp_c', type=float, default=0.1)
    parser.add_argument('--clip_r', type=float, default=2.3)

    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)

    world_size = utils.get_world_size()
    print('world size')
    if args.model.startswith("vit"):
        mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        
    # if args.dataset[0] == "All":
    # dataset_list = ["CUB", "Cars", "SOP", "Inshop"]    
    dataset_list = ["CUB", "Cars", "SOP", "Inshop", "NAbird", "Dogs", "Flowers", "Aircraft"]        
    num_worker = args.num_workers if args.num_workers is not None else multiprocessing.cpu_count() // world_size
        
    method = ["Triplet", "MS", "Margin" ,"PA", "ProxyNCApp", "CosFace", "ArcFace", "SoftTriple", "Hyp", "CurricularFace"]
    IPC=[4, 4, 4, 0, 4, 0, 0, 0 ,4, 0]
    hyp_c=[0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0]
    batch_size=[180, 180, 120, 180, 180, 180, 180, 180, 180, 180]
    lr=[3e-5, 3e-5, 3e-5, 3e-5, 3e-5, 3e-5, 3e-5, 3e-5, 3e-5, 3e-5]
    
    for method_idx in range(8,9):
        models = []
        args.hyp_c = hyp_c[method_idx]
        print('Start Ensemble Evaluation of {}'.format(method[method_idx]))
        for i, dataset in enumerate(dataset_list):
            models.append(nn.parallel.DistributedDataParallel(init_model(args), device_ids=[args.gpu], broadcast_buffers=True, find_unused_parameters=True))
            path = './univ_metric/specific_model_checkpoint/specific_models_chekpoints/vit_small_patch16_224_Finetuned_Specific_{}_{}_IPC{}_DIM128_LR3e-5_checkpoint_best.pth'.format(
                dataset, method[method_idx],IPC[method_idx]
            )
            state_dict = torch.load(path, map_location="cpu")['stduent']
            models[i].load_state_dict(state_dict, strict=False)
                    
        # Set DataLoader
        train_tr = utils.MultiTransforms(mean_std, args.global_crops_number)
        ds_train = All_dataset(args.data_path, dataset_list, "train", train_tr, fewshot_dir=args.fewshot_dir, shot=args.num_shot)
        nb_classes = len(list(set(ds_train.ys)))
        
        recall_unified, recalls_query_sep, recalls_both_sep = evaluate_all(model=models, ds=All_dataset, ds_list = dataset_list,
                                                                    path=args.data_path, mean_std=mean_std, world_size=world_size,
                                                                    eval_mode=["unified"], hyp_c=hyp_c[method_idx], skip_head=args.skip_head, ensemble=True)
        
        
