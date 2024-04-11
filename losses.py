import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

from pytorch_metric_learning import miners, losses, distances
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
import torch.distributed as dist

from hyptorch.pmath import dist_matrix

def ortho_penalty(t):
    return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean() * 1e-6
    
class TripletLoss(nn.Module):
    def __init__(self, sz_embed):
        torch.nn.Module.__init__(self)
        self.sz_embed = sz_embed
        self.loss_func = losses.TripletMarginLoss(distance = distances.CosineSimilarity())
        self.miner = miners.BatchHardMiner()
    
    def forward(self, X, T):
        X = F.normalize(X)
        hard_pairs = self.miner(X, T)
        loss = self.loss_func(X, T, hard_pairs)
        return loss
    
class MSLoss(nn.Module):
    def __init__(self, sz_embed):
        torch.nn.Module.__init__(self)
        self.sz_embed = sz_embed
        self.loss_func = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
        self.miner = miners.DistanceWeightedMiner(cutoff=0.5, nonzero_loss_cutoff=1.4)
    
    def forward(self, X, T):
        X = F.normalize(X)
        hard_pairs = self.miner(X, T)
        loss = self.loss_func(X, T, hard_pairs)
        return loss
    
class MarginLoss(nn.Module):
    def __init__(self, sz_embed):
        torch.nn.Module.__init__(self)
        self.sz_embed = sz_embed
        self.loss_func = losses.MarginLoss(margin=0.2, nu=0, beta=1.2)
        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
    
    def forward(self, X, T):
        X = F.normalize(X)
        hard_pairs = self.miner(X, T)
        loss = self.loss_func(X, T, hard_pairs)
        return loss
    
class PALoss(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
        self.proxies = torch.nn.Parameter(torch.randn(self.nb_classes, self.sz_embed).cuda())
        nn.init.kaiming_uniform_(self.proxies, mode='fan_out')
        
    def forward(self, X, T, P=None):
        if P is None:
            P = self.proxies
        else:
            P = P[:self.nb_classes]
                
        cos = F.linear(F.normalize(X), F.normalize(P))  # Calcluate cosine similarity
        P_one_hot = F.one_hot(T, num_classes = self.nb_classes).float()        
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
                
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        
        loss = (pos_term + neg_term)
        
        return loss
    
class PNCALoss(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.alpha = alpha
        self.mrg = mrg
        
        self.proxies = torch.nn.Parameter(torch.randn(self.nb_classes, self.sz_embed).cuda())
        nn.init.kaiming_uniform_(self.proxies, mode='fan_out')
    
    def forward(self, X, T, P=None):
        if P is None:
            P = self.proxies
        else:
            P = P[:self.nb_classes]
            
        cos = F.linear(F.normalize(X), F.normalize(P))  # Calcluate cosine similarity
        P_one_hot = F.one_hot(T, num_classes = self.nb_classes).float()        
        N_one_hot = 1 - P_one_hot
        
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(1)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(1)

        loss = (torch.log(1 + P_sim_sum) + torch.log(1 * N_sim_sum)).sum() / len(X)
        
        return loss
    
    
class ProxyNCApp(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, alpha = 9):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.alpha = alpha
        self.proxies = torch.nn.Parameter(torch.randn(self.nb_classes, self.sz_embed).cuda())
        nn.init.kaiming_uniform_(self.proxies, mode='fan_out')
        
    def forward(self, X, T, P=None):
        if P is None:
            P = self.proxies
        else:
            P = P[:self.nb_classes]
        
        D = 2 * (1-F.linear(F.normalize(X), F.normalize(P)))  # Calcluate cosine similarity
        P_one_hot = F.one_hot(T, num_classes = self.nb_classes).float()
        
        loss = torch.sum(-P_one_hot * F.log_softmax(-D, -1), -1)
        loss = loss.mean()
        
        return loss
    
class CosFaceLoss(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.35, alpha = 64):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.alpha = alpha
        self.mrg = mrg
        
        self.proxies = torch.nn.Parameter(torch.randn(self.nb_classes, self.sz_embed).cuda())
        nn.init.kaiming_uniform_(self.proxies, mode='fan_out')
        
    def forward(self, X, T, P=None):
        if P is None:
            P = self.proxies
        else:
            P = P[:self.nb_classes]
        
        cos = F.linear(F.normalize(X), F.normalize(P))  # Calcluate cosine similarity
        P_one_hot = F.one_hot(T, num_classes = self.nb_classes).float()
        N_one_hot = 1 - P_one_hot
        
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * cos)

        P_sim = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp))
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(1, keepdim=True)

        loss = (torch.log(1 + P_sim * N_sim_sum)).sum() / len(X)
        
        return loss
    
class ArcFaceLoss(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.5, alpha=64.0):
        torch.nn.Module.__init__(self)
        self.sz_embed = sz_embed
        self.nb_classes = nb_classes
        self.criterion = nn.CrossEntropyLoss()

        self.mrg = mrg
        self.alpha = alpha
        
        self.proxies = torch.nn.Parameter(torch.randn(self.nb_classes, self.sz_embed).cuda())
        nn.init.kaiming_uniform_(self.proxies, mode='fan_out')

        self.cos_m = math.cos(mrg)
        self.sin_m = math.sin(mrg)
        self.th = math.cos(math.pi - mrg)
        self.mm = math.sin(math.pi - mrg) * mrg

    def forward(self, X, T):
        # input is not l2 normalized
        cosine = F.linear(F.normalize(X), F.normalize(self.proxies)).float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).float()

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = F.one_hot(T, num_classes = self.nb_classes).float()

        logit = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logit *= self.alpha

        loss = self.criterion(logit, T)

        return loss
    
class SupCon(torch.nn.Module):
    def __init__(self, tau=0.1, hyp_c=0.1, IPC=1):
        torch.nn.Module.__init__(self)
        self.tau = tau
        self.hyp_c = hyp_c
        self.IPC = IPC
        
        if hyp_c == 0:
            self.dist_f = lambda x, y: x @ y.t()
        else:
            self.dist_f = lambda x, y: -dist_matrix(x, y, c=hyp_c)
            
    def compute_loss(self, x0, x1):
        bsize = x0.shape[0]
        target = torch.arange(bsize).cuda()
        eye_mask = torch.eye(bsize).cuda() * 1e9
        logits00 = self.dist_f(x0, x0) / self.tau - eye_mask
        logits01 = self.dist_f(x0, x1) / self.tau
        logits = torch.cat([logits01, logits00], dim=1)
        logits -= logits.max(1, keepdim=True)[0].detach()
        loss = F.cross_entropy(logits, target)
        return loss
    
    def forward(self, X, T):
        # x0 and x1 - positive pair
        # tau - temperature
        # hyp_c - hyperbolic curvature, "0" enables sphere mode
        loss = 0
        step = 0
        for i in range(self.IPC):
            for j in range(self.IPC):
                if i != j:
                    loss += self.compute_loss(X[:, i], X[:, j])
                step += 1
        loss /= step
        return loss
    
class CurricularFace(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.3, alpha = 32, mrg_train=False):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.alpha = alpha
        self.mrg = mrg
        
        self.proxies = torch.nn.Parameter(torch.randn(self.nb_classes, self.sz_embed).cuda())
        nn.init.kaiming_uniform_(self.proxies, mode='fan_out')
        
        self.momentum = 0.99
        self.register_buffer('pos_sim', torch.zeros(self.nb_classes))
        
        self.cos_m = math.cos(mrg)
        self.sin_m = math.sin(mrg)
        self.th = math.cos(math.pi - mrg)
        self.mm = math.sin(math.pi - mrg) * mrg

    def update_sim(self, cos, T):
        P_one_hot = F.one_hot(T, num_classes = self.nb_classes).float()        
        N_one_hot = 1 - P_one_hot
        
        pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        cur_pos_sim = torch.where(P_one_hot == 1, cos, torch.zeros_like(cos)).sum(dim=0).detach() / (P_one_hot==1).sum(0)
        self.pos_sim[pos_proxies] = self.momentum * self.pos_sim[pos_proxies] + (1-self.momentum) * cur_pos_sim[pos_proxies]
        
    def forward(self, X, T):
        # input is not l2 normalized
        cosine = F.linear(F.normalize(X), F.normalize(self.proxies)).float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).float()

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = F.one_hot(T, num_classes = self.nb_classes).float()

        pos_phi = torch.gather(phi, 1, T.unsqueeze(-1))
        neg_cosine = torch.where(pos_phi >= cosine, cosine, cosine * (cosine + self.pos_sim[T].unsqueeze(-1)))
        logit = (one_hot * phi) + ((1.0 - one_hot) * neg_cosine)
        logit *= self.alpha

        loss = F.cross_entropy(logit, T)
        
        self.update_sim(cosine, T)

        return loss

class XBM(nn.Module):
    def __init__(self, loss, sz_embed, queue_size=8192):
        torch.nn.Module.__init__(self)
        self.sz_embed = sz_embed
        self.loss_func = losses.CrossBatchMemory(loss.loss_func, sz_embed, memory_size=queue_size, miner=loss.miner)
            
    def forward(self, X, T):
        X = F.normalize(X)
        loss = self.loss_func(X, T)
        return loss

class BroadMemory(nn.Module):
    def __init__(
        self,
        criterion,
        queue_size=8192,
        compensate=False,
    ):
        super(BroadMemory, self).__init__()
        self.criterion = criterion
        self.queue_size = queue_size
        self.compensate = compensate
        
        feature_mb = torch.zeros(0, self.criterion.sz_embed)
        label_mb = torch.zeros(0, dtype=torch.int64)
        self.register_buffer("feature_mb", feature_mb)
        self.register_buffer("label_mb", label_mb)

        if self.compensate:
            proxy_mb = torch.zeros(0, self.criterion.sz_embed)
            self.register_buffer("proxy_mb", proxy_mb)

    def update(self, input, label):
        self.feature_mb = torch.cat([self.feature_mb, input.data], dim=0)
        self.label_mb = torch.cat([self.label_mb, label.data], dim=0)
        if self.compensate:
            self.proxy_mb = torch.cat(
                [self.proxy_mb, self.criterion.proxies.data[label].clone()], dim=0
            )

        over_size = self.feature_mb.shape[0] - self.queue_size
        if over_size > 0:
            self.feature_mb = self.feature_mb[over_size:]
            self.label_mb = self.label_mb[over_size:]
            if self.compensate:
                self.proxy_mb = self.proxy_mb[over_size:]

    def forward(self, X, T):
        # input is not l2 normalized
        if self.compensate:
            proxy_now = self.criterion.proxies.data[self.label_mb]
            delta_proxy = proxy_now - self.proxy_mb
            
            update_feature_mb = (
                self.feature_mb
                + (
                    self.feature_mb.norm(p=2, dim=1, keepdim=True)
                    / self.proxy_mb.norm(p=2, dim=1, keepdim=True)
                )
                * delta_proxy
            )
        else:
            update_feature_mb = self.feature_mb

        large_X = torch.cat([update_feature_mb, X.data], dim=0)
        large_T = torch.cat([self.label_mb, T], dim=0)

        batch_loss = self.criterion(X, T)
        broad_loss = self.criterion(large_X, large_T)
        self.update(X, T)

        return batch_loss + broad_loss
    
class SoftTripleLoss(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, centers_per_class=10, la=20, gamma=0.1, mrg=0.01):
        torch.nn.Module.__init__(self)
        self.loss_func = losses.SoftTripleLoss(nb_classes, sz_embed, centers_per_class, la, gamma, mrg)
    
    def forward(self, X, T):
        X = F.normalize(X)
        loss = self.loss_func(X, T)
        return loss
    
class ProtoNet(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 4, IPC=1):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.alpha = alpha
        self.IPC = IPC
        self.mrg = mrg

    def forward(self, X, T):
            
        sorted_X = X[T.sort()[1]]
        splitted_X = sorted_X.split(self.IPC)
        proto = torch.stack([x.mean(0) for x in splitted_X]).to(X.device)
        new_T = torch.arange(len(T)//self.IPC).repeat(self.IPC).sort()[0].to(X.device)
        
        cos = F.linear(F.normalize(sorted_X), F.normalize(proto))
        
        loss = F.cross_entropy(cos * self.alpha, new_T)

        # P_one_hot = F.one_hot(new_T, num_classes = len(new_T) // self.IPC).float()
        # N_one_hot = 1 - P_one_hot
        
        # pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        # neg_exp = torch.exp(self.alpha * cos)

        # P_sim = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp))
        # N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(1, keepdim=True)

        # loss = (torch.log(1 + P_sim * N_sim_sum)).sum() / len(X)
        
        return loss


# class ProtoNet(torch.nn.Module):
#     def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32, IPC=1):
#         torch.nn.Module.__init__(self)
#         # Proxy Anchor Initialization
#         self.nb_classes = nb_classes
#         self.sz_embed = sz_embed
#         self.alpha = alpha
#         self.mrg = mrg
#         self.IPC = IPC
        
#     def forward(self, X, T, P=None):
            
#         sorted_X = X[T.sort()[1]]
#         splitted_X = sorted_X.split(self.IPC)
#         proto = torch.stack([x.mean(0) for x in splitted_X]).to(X.device)
#         new_T = torch.arange(len(T)//self.IPC).repeat(self.IPC).sort()[0].to(X.device)
            
#         cos = F.linear(F.normalize(sorted_X), F.normalize(proto))  # Calcluate cosine similarity
#         P_one_hot = F.one_hot(new_T, num_classes = len(new_T) // self.IPC)      
#         N_one_hot = 1 - P_one_hot
        
#         pos_exp = torch.exp(-self.alpha * (cos - 0.6))
#         neg_exp = torch.exp(self.alpha * (cos - 0.4))

#         P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(1)
#         N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(1)

#         loss = (torch.log(1 + P_sim_sum) + torch.log(1 * N_sim_sum)).sum() / len(X)
        
#         return loss