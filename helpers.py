import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import PIL
import multiprocessing
import utils

from hyptorch.pmath import dist_matrix, mobius_matvec, poincare_mean

from pytorch_metric_learning.utils.accuracy_calculator import precision_at_k, r_precision, mean_average_precision, get_label_match_counts


class MultiSample:
    def __init__(self, transform, n=2):
        self.transform = transform
        self.num = n

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num))
    
def get_emb(model, ds, ds_list, path, mean_std, ds_mode="eval", skip_head=False, world_size=1):
    resize, crop = 256, 224
    
    eval_tr = T.Compose(
        [
            T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(crop),
            T.ToTensor(),
            T.Normalize(*mean_std),
        ]
    )
    ds_eval = ds(path, ds_list, ds_mode, eval_tr)
    
    if world_size == 1:
        sampler = None
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(ds_eval)
        
    dl_eval = DataLoader(
        dataset=ds_eval,
        batch_size=1000,
        shuffle=False,
        num_workers=multiprocessing.cpu_count() // world_size,
        pin_memory=True,
        drop_last=False,
        sampler=sampler,
    )
    model.eval()
    x, y, index, ds_ID = eval_dataset(model, dl_eval, skip_head)
    y, index, ds_ID = y.cuda(), index.cuda(), ds_ID.cuda()
    if world_size > 1:
        all_x = [torch.zeros_like(x) for _ in range(world_size)]
        all_y = [torch.zeros_like(y) for _ in range(world_size)]
        all_index = [torch.zeros_like(index) for _ in range(world_size)]
        all_ds_ID = [torch.zeros_like(ds_ID) for _ in range(world_size)]
        torch.distributed.all_gather(all_x, x)
        torch.distributed.all_gather(all_y, y)
        torch.distributed.all_gather(all_index, index)
        torch.distributed.all_gather(all_ds_ID, ds_ID)
        x, y, index, ds_ID = torch.cat(all_x), torch.cat(all_y), torch.cat(all_index), torch.cat(all_ds_ID)

    model.train()
    return x, y, index, ds_ID

def get_emb_multi(models, ds, ds_list, path, mean_std, ds_mode="eval", skip_head=False, world_size=1, hyp_c=None):
    resize, crop = 256, 224
    
    eval_tr = T.Compose(
        [
            T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(crop),
            T.ToTensor(),
            T.Normalize(*mean_std),
        ]
    )
    ds_eval = ds(path, ds_list, ds_mode, eval_tr)
    
    if world_size == 1:
        sampler = None
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(ds_eval)
        
    dl_eval = DataLoader(
        dataset=ds_eval,
        batch_size=1000,
        shuffle=False,
        num_workers=multiprocessing.cpu_count() // world_size,
        pin_memory=True,
        drop_last=False,
        sampler=sampler,
    )
    
    multi_x = []
    for i, model in enumerate(models):
        print('Feature Extraction of {}-th model'.format(i))
        model.eval()
        x, y, index, ds_ID = eval_dataset(model, dl_eval, skip_head)
        y, index, ds_ID = y.cuda(), index.cuda(), ds_ID.cuda()
        if world_size > 1:
            all_x = [torch.zeros_like(x) for _ in range(world_size)]
            all_y = [torch.zeros_like(y) for _ in range(world_size)]
            all_index = [torch.zeros_like(index) for _ in range(world_size)]
            all_ds_ID = [torch.zeros_like(ds_ID) for _ in range(world_size)]
            torch.distributed.all_gather(all_x, x)
            torch.distributed.all_gather(all_y, y)
            torch.distributed.all_gather(all_index, index)
            torch.distributed.all_gather(all_ds_ID, ds_ID)
            x, y, index, ds_ID = torch.cat(all_x), torch.cat(all_y), torch.cat(all_index), torch.cat(all_ds_ID)
        multi_x.append(x)
        
    if hyp_c == 0:
        x = torch.stack(multi_x).mean(0)
    else:
        x = poincare_mean(torch.stack(multi_x), dim=0, c=hyp_c)
    model.train()
    return x, y, index, ds_ID

def eval_dataset(model, dl, skip_head=False):
    all_x, all_y, all_index, all_ds_ID = [], [], [], []
    for x, y, index, ds_ID in dl:
        with torch.no_grad():
            x = x.cuda(non_blocking=True)
            all_x.append(model(x, skip_head))
        all_y.append(y)
        all_index.append(index)
        all_ds_ID.append(ds_ID)
        
    return torch.cat(all_x), torch.cat(all_y), torch.cat(all_index), torch.cat(all_ds_ID)

def evaluate_all(model, ds, ds_list, path, mean_std, ds_mode="eval", 
                metric=["R@1"], eval_mode=["unified", "both_sep"],
                world_size=1, hyp_c=0, skip_head=False, ensemble=False):

    print("Start Evaluation")
    if not ensemble:
        emb_head_query = get_emb(model, ds, ds_list, path, mean_std, ds_mode="query", skip_head=skip_head, world_size=world_size)
        emb_head_gal = get_emb(model, ds, ds_list, path, mean_std, ds_mode="gallery", skip_head=skip_head, world_size=world_size)
    else:
        emb_head_query = get_emb_multi(model, ds, ds_list, path, mean_std, ds_mode="query", skip_head=skip_head, world_size=world_size, hyp_c=hyp_c)
        emb_head_gal = get_emb_multi(model, ds, ds_list, path, mean_std, ds_mode="gallery", skip_head=skip_head, world_size=world_size, hyp_c=hyp_c)
    ds_ID_set = {"CUB":0, "Cars":1, "SOP":2, "Inshop":3, "NAbird":4, "Dogs":5, "Flowers":6, "Aircraft":7}
    
    eval_dict = {}

    recall_unified = None
    if "unified" in eval_mode:
        print("\033[91m" + "Query Unified, Gallery Unified" + '\033[0m')
        emb_head_query_unified = (emb_head_query[0], emb_head_query[1], emb_head_query[2])
        emb_head_gal_unified = (emb_head_gal[0], emb_head_gal[1], emb_head_gal[2])
        accuarcy_dict = get_accuracy(*emb_head_query_unified, *emb_head_gal_unified, hyp_c, metric=metric)
        for key in metric:
            eval_dict['Unified_' + key] = accuarcy_dict[key]
            
    recalls_both_sep = None
    if "both_sep" in eval_mode:
        print("\033[91m" + "Query Seperated, Gallery Seperated" + '\033[0m')
        accuracy_both_sep = {}
        for each_dataset in ds_list:
            print(each_dataset, end=': ')
            q_ds_ID = emb_head_query[-1]
            g_ds_ID = emb_head_gal[-1]
            q_in_ds = (q_ds_ID == ds_ID_set[each_dataset])
            g_in_ds = (g_ds_ID == ds_ID_set[each_dataset])  
                       
            emb_head_query_dataset = (emb_head_query[0][q_in_ds], emb_head_query[1][q_in_ds], emb_head_query[2][q_in_ds])
            emb_head_gal_dataset = (emb_head_gal[0][g_in_ds], emb_head_gal[1][g_in_ds], emb_head_gal[2][g_in_ds])
            accuarcy_dict = get_accuracy(*emb_head_query_dataset, *emb_head_gal_dataset, each_dataset, hyp_c, metric=metric)
            for key in metric:
                accuracy_both_sep[each_dataset + '_' + key] = accuarcy_dict[key]
                
        eval_dict.update(accuracy_both_sep)
        harmonics = {}
        for key in metric:
            harmonic_metric = len(ds_list) / sum([1/np.array(accuracy_both_sep[ds+'_'+key]) for ds in ds_list])
            eval_dict['Harmonic_' + key] = harmonic_metric
            harmonics[key] = harmonic_metric
        print('Harmonic: {}'.format(harmonics))
            
    return eval_dict

def pdist(A, B, squared = False, eps = 1e-12):
    D = A.pow(2).sum(1) + (-2) * B.mm(A.t())
    D = (B.pow(2).sum(1) + D.t()).clamp(min=eps)
    
    if not squared:
        D = D.sqrt()
        
    if torch.equal(A,B):
        D = D.clone()
        D[range(len(A)), range(len(A))] = 0
        
    return D

# def get_accuracy(xq, yq, index_q, xg, yg, index_g, each_dataset=None, hyp_c=0, metric=None):
#     if each_dataset == "SOP":
#         k_list = [1, 10, 100, 1000]
#     elif each_dataset == "Inshop":
#         k_list = [1, 10, 20, 30]
#     else:
#         k_list = [1, 2, 4, 8]

#     import pdb; pdb.set_trace()
#     # def cal_part_sim()



#     def part_dist_and_match(xq, yq, index_q, xg, yg, index_g, k, label_counts, metric):
#         sim = F.linear(xq, xg)

#         if metric == 'recall_at_k':
#             pos_sim = torch.where(torch.logical_and(yg == yq.unsqueeze(-1), index_g != index_q.unsqueeze(-1)), sim, -torch.ones_like(sim) * float('inf'))
#             neg_sim = torch.where(torch.logical_and(yg != yq.unsqueeze(-1), index_g != index_q.unsqueeze(-1)), sim, -torch.ones_like(sim) * float('inf'))
#             thresh = torch.max(pos_sim, dim=1)[0]
#             match_counter = torch.sum(torch.sum(neg_sim > thresh.unsqueeze(-1), dim=1) < k).item()
#         else:
#             sim = torch.where(index_g != index_q.unsqueeze(-1), sim, -torch.ones_like(sim) * float('inf'))
#             sorted_labels = yg[sim.sort(descending=True)[1]]

#             if metric == 'precision_at_k':
#                 match_counter = precision_at_k(sorted_labels, yq[:, None], k, False, False, torch.eq) * len(yq)
#             elif metric =='map_at_r':
#                 match_counter = mean_average_precision(sorted_labels, yq[:, None], False, label_counts, False, False, torch.eq, at_r=True) * len(yq)
#             elif metric =='r_precision':
#                 match_counter = r_precision(sorted_labels, yq[:, None], False, label_counts, False, False, torch.eq, at_r=True) * len(yq)
        
#         return match_counter
    
#     def calcaulate_accuarcy(xq, yq, index_q, xg, yg, index_g, k, metric=None, split_size = 2000):
#         match_counter = 0
#         splits = range(0, len(xq), split_size)
#         label_counts = get_label_match_counts(yq, yg, torch.eq)

#         if split_size < len(xq):
#             for i in range(0, len(splits)-1):
#                 split_xq, split_yq, split_index_q = xq[splits[i]:splits[i+1]], yq[splits[i]:splits[i+1]], index_q[splits[i]:splits[i+1]]
#                 match_counter += part_dist_and_match(split_xq, split_yq, split_index_q, xg, yg, index_g, k, label_counts=label_counts)

#         split_xq, split_yq, split_index_q = xq[splits[-1]:], yq[splits[-1]:], index_q[splits[-1]:]
#         match_counter += part_dist_and_match(split_xq, split_yq, split_index_q, xg, yg, index_g, k, label_counts, metric)
#         return match_counter / len(xq)

#         recall = [calcaulate_accuarcy(xq, yq, index_q, xg, yg, index_g, k, metric) for k in k_list]

#     print(recall)
#     return [round(rec, 4) for rec in recall]

def recall_at_k(knn_labels, gt_labels, k):
    """
    gt_labels : [nb_samples] (target labels)
    knn_labels : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(gt_labels, knn_labels):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(gt_labels))

def get_accuracy(xq, yq, index_q, xg, yg, index_g, each_dataset=None, hyp_c=0, metric=None):
    if each_dataset == "SOP":
        k_list = [1, 10, 100, 1000]
    elif each_dataset == "Inshop":
        k_list = [1, 10, 20, 30]
    else:
        k_list = [1, 2, 4, 8]

    def get_part_maksed_knn(xq, yq, index_q, xg, yg, index_g, max_k):
        sim = F.linear(xq, xg)
        sim = torch.where(index_g != index_q.unsqueeze(-1), sim, -torch.ones_like(sim) * float('inf'))
        part_topk_labels = yg[sim.topk(max_k)[1]]
        return part_topk_labels
    
    def calcaulate_accuarcy(xq, yq, index_q, xg, yg, index_g, k=1, metric=None, split_size = 2000):
        accuarcy_dict = {}
        
        label_counts  = get_label_match_counts(yq, yg, torch.eq)
        max_k = torch.max(label_counts[1]).item()
        splits = range(0, len(xq), split_size)

        topk_labels = []
        if split_size < len(xq):
            for i in range(0, len(splits)-1):
                split_query = xq[splits[i]:splits[i+1]], yq[splits[i]:splits[i+1]], index_q[splits[i]:splits[i+1]]
                topk_labels.append(get_part_maksed_knn(*split_query, xg, yg, index_g, max_k))
            split_query = xq[splits[-1]:], yq[splits[-1]:], index_q[splits[-1]:]
            topk_labels.append(get_part_maksed_knn(*split_query, xg, yg, index_g, max_k))
        topk_labels = torch.cat(topk_labels, dim=0)

        if 'P@1' in metric:
            accuarcy_dict['R@1'] = precision_at_k(topk_labels, yq[:, None], k, False, False, torch.eq)
        if 'R@1' in metric:
            accuarcy_dict['R@1'] = recall_at_k(topk_labels, yq[:, None], k)
        if 'MAP@R' in metric:
            accuarcy_dict['MAP@R'] = mean_average_precision(topk_labels, yq[:, None], False, label_counts, False, False, torch.eq, at_r=True)
        if 'RP' in metric:
            accuarcy_dict['RP'] = r_precision(topk_labels, yq[:, None], False, label_counts, False, False, torch.eq)
        
        return accuarcy_dict

    accuarcy_dict = calcaulate_accuarcy(xq, yq, index_q, xg, yg, index_g, 1, metric)
    print(accuarcy_dict)
    return accuarcy_dict