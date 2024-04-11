
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
    parser.add_argument('--model', default='dino_vits16', type=str,
        choices=['deit_small_distilled_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224', 
                'vit_large_patch16_224', 'vit_small_patch16_224_dino', 'vit_base_patch16_clip_224'],
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
    parser.add_argument('--clip_grad', type=float, default=1.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=180, type=int,
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
    parser.add_argument('--data_path', default='./Data/', type=str,
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
    parser.add_argument('--use_prompt', default=False, type=utils.bool_flag, help="""use prefix or prompt""")
    parser.add_argument('--prompt_type', default='pool', type=str, help="""prompt_type.""")
    parser.add_argument('--prefix_style', default=False, type=utils.bool_flag, help="""use prefix or prompt""")
    parser.add_argument('--prompt_deep', default=False, type=utils.bool_flag, help="""prompt in hidden states.""")
    parser.add_argument('--prompt_lambda', default=0, type=float, help="""prompt_lambda.""")
    parser.add_argument('--component_size', default=0, type=int, help="""component_size.""")
    parser.add_argument('--prompt_length', default=0, type=int, help="""prompt_length.""")
    
    ## adapters
    parser.add_argument('--use_adapter', default=False, type=utils.bool_flag, help="""use parallel adapter""")
    parser.add_argument('--adapter_type', default='stochastic', type=str, help="""prompt_type.""")
    parser.add_argument('--adapter_dim', default=0, type=int, help="""Parallel Adapter dimension""")
    parser.add_argument('--adapter_droppath', default=0, type=float, help="""use parallel adapter drop path""")

    ## Lora
    parser.add_argument('--use_lora', default=False, type=utils.bool_flag, help="""use Lora""")
    parser.add_argument('--lora_dim', default=0, type=int, help="""Lora dimension""")
    parser.add_argument('--lora_droppath', default=0, type=float, help="""use Lora drop path""")

    ## embedding layer
    parser.add_argument('--emb', default=128, type=int, help="""Dimensionality of output for [CLS] token.""")
    parser.add_argument('--emb_mlp', default=False, type=utils.bool_flag, help="""Dimensionality of output for [CLS] token.""")
    
    ## Hyperbolic Vision Trnasformer
    parser.add_argument('--hyp_c', type=float, default=0)
    parser.add_argument('--clip_r', type=float, default=2.3)

    return parser

def train_one_epoch(model, sup_metric_loss, data_loader, optimizer, mean_std,
                    lr_schedule, epoch, fp16_scaler, world_size, args):    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
        
    if args.pretrained_eval:
        if args.test_dataset[0] == "All":
            args.test_dataset = ["CUB", "Cars", "SOP", "Inshop", "NAbird", "Dogs", "Flowers", "Aircraft"]
        elif args.test_dataset[0] == "Standard":
            args.test_dataset = ["CUB", "Cars", "SOP", "Inshop"]
        recall_unified, recalls_query_sep, recalls_both_sep = evaluate_all(model=model.module, ds=All_dataset, ds_list = args.test_dataset,
                                                            path=args.data_path, mean_std=mean_std, world_size=world_size,
                                                            eval_mode=["unified", "query_sep", "both_sep"], hyp_c=args.hyp_c, skip_head=args.skip_head)
        
    model.train()    
    if args.norm_freeze:
        for m in model.modules(): 
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                m.eval()
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

    print('fine-tune norm:', model.module.body.norm.weight.requires_grad)

    for i, param_group in enumerate(optimizer.param_groups):
        if not "proxies" in param_group["name"]:
            if epoch < args.warmup_epochs:
                for param in param_group["params"]:
                    param.requires_grad = False
            else:
                if args.freeze and "pretrained_params" in param_group["name"]:
                    pass
                else:
                    for param in param_group["params"]:
                        param.requires_grad = True

    for it, (x, y, _, _) in enumerate(metric_logger.log_every(data_loader, 20, header)):        
        it = len(data_loader) * epoch + it  
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = args.lr * param_group["lr_scale"] * lr_schedule[it]
        
        x = torch.cat([im.cuda(non_blocking=True) for im in x])
        y = y.cuda(non_blocking=True).repeat(args.global_crops_number)
        
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            z, prompt_loss = model(x)
            prompt_loss *= args.prompt_lambda
            
            if args.loss == 'SupCon' and args.IPC >0:
                z = z.view(len(x) // args.IPC, args.IPC, args.emb)
                            
            if world_size > 1:
                z = utils.all_gather(z, args.local_rank)
                y = utils.all_gather(y, args.local_rank)
            
            metric_loss = sup_metric_loss(z, y)
            loss = metric_loss + prompt_loss
                   
        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(False):
            if fp16_scaler is None:
                
                loss.backward()
                if args.clip_grad > 0:
                    utils.clip_gradients_value(model, 1, losses=[sup_metric_loss])
                    
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                if args.clip_grad > 0:
                    fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    utils.clip_gradients_value(model, 1, losses=[sup_metric_loss])
                    
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
                
        torch.cuda.synchronize()
        metric_logger.update(Total_Loss=loss.item())
        metric_logger.update(Metric_loss=metric_loss.item())
        metric_logger.update(Prompt_loss=prompt_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        
    metric_logger.synchronize_between_processes()
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    rh_model = 0
    if epoch % args.eval_freq == 0 and epoch >= args.warmup_epochs and epoch > 0:
        if args.test_dataset[0] == "All":
            args.test_dataset = ["CUB", "Cars", "SOP", "Inshop", "NAbird", "Dogs", "Flowers", "Aircraft"]
        elif args.test_dataset[0] == "Standard":
            args.test_dataset = ["CUB", "Cars", "SOP", "Inshop"]
        eval_dict = evaluate_all(model=model.module, ds=All_dataset, ds_list = args.test_dataset,
                                                            path=args.data_path, mean_std=mean_std, world_size=world_size,
                                                            eval_mode=["unified", "both_sep"], hyp_c=args.hyp_c, skip_head=args.skip_head)
        # return_dict.update({"R@1_unified": recall_unified})
        return_dict.update(eval_dict)
        
    return return_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)

    if utils.is_main_process() and args.wandb:
        wandb.init(project="univ_metric", name="{}_{}_{}_{}".format(','.join(str(s) for s in args.dataset), args.model, args.emb, args.run_name), config=args)

    world_size = utils.get_world_size()
    print('world size')
    if args.model in ["vit_small_patch16_224", "vit_base_patch16_224", "vit_large_patch16_224"]:
        mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        
    if args.dataset[0] == "All":
        dataset_list = ["CUB", "Cars", "SOP", "Inshop", "NAbird", "Dogs", "Flowers", "Aircraft"]
    elif args.dataset[0] == "Standard":
        dataset_list = ["CUB", "Cars", "SOP", "Inshop"]
    else:
        dataset_list = args.dataset
        
    num_worker = args.num_workers if args.num_workers is not None else multiprocessing.cpu_count() // world_size
        
    model = init_model(args)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=True, find_unused_parameters=True)
        
    # Set DataLoader
    train_tr = utils.MultiTransforms(mean_std, args.global_crops_number)
    ds_train = All_dataset(args.data_path, dataset_list, "train", train_tr, fewshot_dir=args.fewshot_dir, shot=args.num_shot)
    nb_classes = len(list(set(ds_train.ys)))
    
    # Set Metric Learning Loss
    if args.loss == 'MS':
        sup_metric_loss = MSLoss(sz_embed = args.emb).cuda()
    elif args.loss == 'Triplet':
        sup_metric_loss = TripletLoss(sz_embed = args.emb).cuda()
    elif args.loss == 'Margin':
        sup_metric_loss = MarginLoss(sz_embed = args.emb).cuda()
    elif args.loss == 'PA':
        sup_metric_loss = PALoss(nb_classes=nb_classes, sz_embed = args.emb).cuda()
    elif args.loss == 'ProxyNCApp':
        sup_metric_loss = ProxyNCApp(nb_classes=nb_classes, sz_embed = args.emb).cuda()
    elif args.loss == 'SoftTriple':
        sup_metric_loss = SoftTripleLoss(nb_classes=nb_classes, sz_embed = args.emb).cuda()        
    elif args.loss == 'PNCA':
        sup_metric_loss = PNCALoss(nb_classes=nb_classes, sz_embed = args.emb).cuda()     
    elif args.loss =='ArcFace':
        sup_metric_loss = ArcFaceLoss(nb_classes=nb_classes, sz_embed = args.emb).cuda()
    elif args.loss =='CosFace':
        sup_metric_loss = CosFaceLoss(nb_classes=nb_classes, sz_embed = args.emb).cuda()
    elif args.loss =='SupCon':
        sup_metric_loss = SupCon(hyp_c=args.hyp_c, IPC=args.IPC).cuda()
    elif args.loss =='CurricularFace':
        sup_metric_loss = CurricularFace(nb_classes=nb_classes, sz_embed = args.emb, alpha=args.alpha, mrg = args.mrg).cuda()
    elif args.loss == 'Proto':
        sup_metric_loss = ProtoNet(nb_classes=nb_classes, sz_embed = args.emb, IPC=args.IPC * args.global_crops_number,  alpha=args.alpha, mrg = args.mrg).cuda()
        
    params_groups = utils.get_params_groups(model, sup_metric_loss, weight_decay=args.weight_decay)
        
    if args.IPC > 0:
        if args.sampler == 'Unique':
            sampler = UniqueClassSampler(ds_train.ys, args.batch_size, args.IPC, rank=args.local_rank, world_size=world_size)
        elif args.sampler == 'DCB':
            sampler = DataClassBalancedSampler(ds_train.ys, ds_train.ds_ID, args.batch_size, args.IPC, rank=args.local_rank, world_size=world_size)
        elif args.sampler == 'Class':
            sampler = ClassMiningSampler(ds_train.ys, args.batch_size, sup_metric_loss, args.IPC, num_class_NN=5, rank=args.local_rank, world_size=world_size)
    else:
        sampler = torch.utils.data.DistributedSampler(ds_train, shuffle=True)
        
    if args.memory:
        sup_metric_loss = XBM(sup_metric_loss, sz_embed = args.emb, queue_size=4096).cuda()


    data_loader = DataLoader(
        dataset=ds_train,
        sampler=sampler,
        batch_size=args.batch_size // world_size,
        num_workers=num_worker,
        pin_memory=True,
        drop_last=True,
    )
        
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "adamp":
        from adamp import AdamP
        optimizer = AdamP(params_groups)  # to use with ViTs
    elif args.optimizer == "adam":
        from adamp import AdamP
        optimizer = torch.optim.Adam(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, momentum=0.9, lr=args.lr)  # lr is set by scheduler
        
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    if args.lr_decay == 'cosine':
        lr_schedule = utils.cosine_scheduler(1, args.min_lr_scale, args.epochs, len(data_loader))
    elif args.lr_decay is not None:
        lr_schedule = utils.step_scheduler(int(args.lr_decay), args.epochs, len(data_loader), gamma=0.5)
    else:
        lr_schedule = utils.step_scheduler(args.epochs, args.epochs, len(data_loader), gamma=1)

    cudnn.benchmark = True
    best_recall = 0
    for epoch in range(args.epochs):
        if sampler is not None and args.IPC > 0:
            sampler.set_epoch(epoch)
            if args.sampler == 'Class':
                sampler.compute_proxy_neighbor()
            
        # ============ training one epoch of iBOT ... ============
        train_stats = train_one_epoch(model, sup_metric_loss, data_loader, optimizer, mean_std,
                                      lr_schedule, epoch, fp16_scaler, world_size, args)

        # ============ writing logs ... ============
        save_dict = {
            'stduent': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }
        
        dataset_name = args.dataset[0] if len(args.dataset) == 1 else '_'.join(str(s) for s in args.dataset)
        log_folder = "{}/{}".format(args.output_dir, dataset_name)
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        if epoch % args.eval_freq == 0 and epoch > 0:
            is_best = train_stats['Unified_R@1'] > best_recall 
            best_recall = train_stats['Unified_R@1'] if is_best else best_recall
            if is_best:
                best_model_path = '{}/{}_{}_checkpoint_best.pth'.format(log_folder, args.model, args.run_name)
                utils.save_on_master(save_dict, best_model_path)
                if utils.is_main_process() and args.wandb_save and args.wandb:
                    wandb.save(best_model_path)
                
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if utils.is_main_process():
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            with (Path("{}/{}_{}_log.txt".format(log_folder, args.model, args.run_name))).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if args.wandb:
                for k, v in train_stats.items():
                    if isinstance(v, list):
                        train_stats[k] = v[0]
                wandb.log(train_stats, step=epoch)
    
    if utils.is_main_process():
        with (Path("{}/{}_{}_log.txt".format(log_folder, args.model, args.run_name))).open("a") as f:
            f.write(json.dumps({'best_recall':best_recall}) + "\n")
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

