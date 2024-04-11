import torch
import random
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
import numpy as np
import collections


def get_labels_to_indices(labels):
    """
    Creates labels_to_indices, which is a dictionary mapping each label
    to a numpy array of indices that will be used to index into self.dataset
    """
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    labels_to_indices = collections.defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)
    for k, v in labels_to_indices.items():
        labels_to_indices[k] = np.array(v, dtype=np.int)
    return labels_to_indices


def safe_random_choice(input_data, size):
    replace = len(input_data) < size
    return np.random.choice(input_data, size=size, replace=replace).tolist()


class UniqueClassSampler(Sampler):
    """
    BatchSampler that ensures a fixed amount of images per class are sampled in the minibatch
    """
    def __init__(self, targets, batch_size, images_per_class=3, rank=0, world_size=1):
        self.targets = targets
        self.batch_size = batch_size
        self.images_per_class = images_per_class
        self.rank = rank
        self.world_size = world_size
        self.reverse_index, self.ignored = self._build_reverse_index()
        self.epoch = 0

    def __iter__(self):
        num_batches = len(self.targets) // (self.world_size * self.batch_size)
        ret = []
        i = 0
        while num_batches > 0:
            ret.extend(self.sample_batch(i))
            num_batches -= 1
            i += 1
        return iter(ret) 

    def _build_reverse_index(self):
        reverse_index = {}
        ignored = []
        for i, target in enumerate(self.targets):
            if target not in reverse_index:
                reverse_index[target] = []
            reverse_index[target].append(i)
        return reverse_index, ignored

    def sample_batch(self, batch_idx):
        np.random.seed(batch_idx * 10000 + self.epoch)
        num_classes = self.batch_size * self.world_size // self.images_per_class
        replace = num_classes > len(list(set(self.targets)))
        sampled_classes = np.random.choice(list(self.reverse_index.keys()), num_classes, replace=replace)
        size = num_classes // self.world_size
        sampled_classes = sampled_classes[size * self.rank : size * (self.rank + 1)]

        np.random.seed(random.randint(0, 1e6))
        sampled_indices = []
        for cls_idx in sampled_classes:
            # Need replace = True for datasets with non-uniform distribution of images per class
            sampled_indices.extend(np.random.choice(self.reverse_index[cls_idx], self.images_per_class, replace=True))
        return sampled_indices

    def __len__(self):
        return len(self.targets) // self.world_size
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        
class BalancedSampler(Sampler):
    """
    BatchSampler that ensures a fixed amount of images per class are sampled in the minibatch
    """
    def __init__(self, targets, batch_size, images_per_class=3, world_size=1):
        self.targets = targets
        self.batch_size = batch_size
        self.images_per_class = images_per_class
        self.world_size = world_size
        self.reverse_index, self.ignored = self._build_reverse_index()

    def __iter__(self):
        num_batches = len(self.targets) // (self.world_size * self.batch_size)
        ret = []
        while num_batches > 0:
            ret.extend(np.random.permutation(self.sample_batch()))
            num_batches -= 1
        return iter(ret) 

    def _build_reverse_index(self):
        reverse_index = {}
        ignored = []
        for i, target in enumerate(self.targets):
            if target not in reverse_index:
                reverse_index[target] = []
            reverse_index[target].append(i)
        return reverse_index, ignored

    def sample_batch(self):
        # Real batch size is self.images_per_class * (self.batch_size // self.images_per_class)
        num_classes = self.batch_size // self.images_per_class
        sampled_classes = np.random.choice(list(self.reverse_index.keys()), num_classes, replace=False)

        sampled_indices = []
        for cls_idx in sampled_classes:
            # Need replace = True for datasets with non-uniform distribution of images per class
            sampled_indices.extend(np.random.choice(self.reverse_index[cls_idx], self.images_per_class, replace=True))
        return sampled_indices

    def __len__(self):
        return len(self.targets) // self.world_size
    
    def set_epoch(self, epoch):
        self.epoch = epoch

class DataClassBalancedSampler(Sampler):
    """
    BatchSampler that ensures a fixed amount of images per class are sampled in the minibatch
    """
    def __init__(self, targets, ds_IDs, batch_size, images_per_class=3, rank=0, world_size=1):
        self.targets = targets
        self.ds_IDs = ds_IDs
        self.ds_set = list(set(ds_IDs))
        self.batch_size = batch_size
        self.images_per_class = images_per_class
        self.rank = rank
        self.world_size = world_size
        self.reverse_index, self.ignored = self._build_reverse_index()
        self.epoch = 0

    def __iter__(self):
        num_batches = len(self.targets) // (self.world_size * self.batch_size)
        ret = []
        i = 0
        while num_batches > 0:
            ret.extend(self.sample_batch(i))
            num_batches -= 1
            i += 1
        return iter(ret) 

    def _build_reverse_index(self):
        reverse_index = {}
        ignored = []
        for i, target in enumerate(self.targets):
            if target not in reverse_index:
                reverse_index[target] = []
            reverse_index[target].append(i)
        return reverse_index, ignored

    def sample_batch(self, batch_idx):
        np.random.seed(batch_idx * 10000 + self.epoch)
        num_classes = self.batch_size * self.world_size // (self.images_per_class * len(self.ds_set))
        sampled_indices = []
        for ds_type in self.ds_set:
            ds_target_set = list(set(np.array(self.targets)[np.array(self.ds_IDs) == ds_type]))
            replace = num_classes > len(ds_target_set)
            sampled_classes = np.random.choice(ds_target_set, num_classes, replace=replace)
            size = num_classes // self.world_size
            sampled_classes = sampled_classes[size * self.rank : size * (self.rank + 1)]

            np.random.seed(random.randint(0, 1e6))
            for cls_idx in sampled_classes:
                # Need replace = True for datasets with non-uniform distribution of images per class
                sampled_indices.extend(np.random.choice(self.reverse_index[cls_idx], self.images_per_class, replace=True))
        return sampled_indices

    def __len__(self):
        return len(self.targets) // self.world_size
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        
        
class ClassMiningSampler(Sampler):
    """
    BatchSampler that ensures a fixed amount of images per class are sampled in the minibatch
    """
    def __init__(self, targets, batch_size, proxy_loss, images_per_class=3, num_class_NN=1, rank=0, world_size=1):
        self.targets = targets
        self.batch_size = batch_size
        self.images_per_class = images_per_class
        self.rank = rank
        self.world_size = world_size
        self.reverse_index, self.ignored = self._build_reverse_index()
        self.proxy_loss = proxy_loss
        self.num_class_NN = num_class_NN
        self.epoch = 0
        self.compute_proxy_neighbor()

    def __iter__(self):
        num_batches = len(self.targets) // (self.world_size * self.batch_size)
        ret = []
        i = 0
        while num_batches > 0:
            ret.extend(self.sample_batch(i))
            num_batches -= 1
            i += 1
        return iter(ret) 

    def _build_reverse_index(self):
        reverse_index = {}
        ignored = []
        for i, target in enumerate(self.targets):
            if target not in reverse_index:
                reverse_index[target] = []
            reverse_index[target].append(i)
        return reverse_index, ignored
    
    def compute_proxy_neighbor(self):
        proxies = self.proxy_loss.proxies.data
        proxy_sim = F.linear(proxies, proxies)
        self.proxy_NN_matrix = proxy_sim.topk(self.num_class_NN, dim=1)[1]

    def sample_batch(self, batch_idx):
        np.random.seed(batch_idx * 10000 + self.epoch)
        num_classes = self.batch_size * self.world_size // self.images_per_class // (1+self.num_class_NN)
        replace = num_classes > len(list(set(self.targets)))
        sampled_classes = np.random.choice(list(self.reverse_index.keys()), num_classes, replace=replace)
        size = num_classes // self.world_size
        sampled_classes = sampled_classes[size * self.rank : size * (self.rank + 1)]
        neighbor_classes = self.proxy_NN_matrix[sampled_classes].flatten(0,1)
        sampled_classes = neighbor_classes.cpu().numpy()
        
        np.random.seed(random.randint(0, 1e6))
        sampled_indices = []
        for cls_idx in sampled_classes:
            # Need replace = True for datasets with non-uniform distribution of images per class
            sampled_indices.extend(np.random.choice(self.reverse_index[cls_idx], self.images_per_class, replace=True))
        return sampled_indices

    def __len__(self):
        return len(self.targets) // self.world_size
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        
# class NNBatchSampler(Sampler):
#     """
#     BatchSampler that ensures a fixed amount of images per class are sampled in the minibatch
#     """
#     def __init__(self, data_source, model, seen_dataloader, batch_size, nn_per_image = 5, using_feat = True, is_norm = True):
#         self.batch_size = batch_size
#         self.nn_per_image = nn_per_image
#         self.using_feat = using_feat
#         self.is_norm = is_norm
#         self.num_samples = data_source.__len__()
#         self.nn_matrix, self.dist_matrix = self._build_nn_matrix(model, seen_dataloader)

#     def __iter__(self):
#         for _ in range(len(self)):
#             yield self.sample_batch()
            
#     def _predict_batchwise(self, model, seen_dataloader):
#         device = "cuda"
#         model_is_training = model.training
#         model.eval()

#         ds = seen_dataloader.dataset
#         A = [[] for i in range(len(ds[0]))]
#         with torch.no_grad():
#             # extract batches (A becomes list of samples)
#             for batch in tqdm(seen_dataloader):
#                 for i, J in enumerate(batch):
#                     # i = 0: sz_batch * images
#                     # i = 1: sz_batch * labels
#                     # i = 2: sz_batch * indices
#                     if i == 0:
#                         # move images to device of model (approximate device)
#                         if self.using_feat:
#                             J, _ = model(J.cuda())
#                         else:
#                             _, J = model(J.cuda())
                            
#                         if self.is_norm:
#                             J = F.normalize(J, p=2, dim=1)
                            
#                     for j in J:
#                         A[i].append(j)
                        
#         model.train()
#         model.train(model_is_training) # revert to previous training state

#         return [torch.stack(A[i]) for i in range(len(A))]
    
#     def _build_nn_matrix(self, model, seen_dataloader):
#         # calculate embeddings with model and get targets
#         X, T, _ = self._predict_batchwise(model, seen_dataloader)
        
#         # get predictions by assigning nearest 8 neighbors with cosine
#         K = self.nn_per_image * 1
#         nn_matrix = []
#         dist_matrix = []
#         xs = []
        
#         for x in X:
#             if len(xs)<5000:
#                 xs.append(x)
#             else:
#                 xs.append(x)            
#                 xs = torch.stack(xs,dim=0)

#                 dist_emb = xs.pow(2).sum(1) + (-2) * X.mm(xs.t())
#                 dist_emb = X.pow(2).sum(1) + dist_emb.t()

#                 ind = dist_emb.topk(K, largest = False)[1].long().cpu()
#                 dist = dist_emb.topk(K, largest = False)[0]
#                 nn_matrix.append(ind)
#                 dist_matrix.append(dist.cpu())
#                 xs = []
#                 del ind

#         # Last Loop
#         xs = torch.stack(xs,dim=0)
#         dist_emb = xs.pow(2).sum(1) + (-2) * X.mm(xs.t())
#         dist_emb = X.pow(2).sum(1) + dist_emb.t()
#         ind = dist_emb.topk(K, largest = False)[1]
#         dist = dist_emb.topk(K, largest = False)[0]
#         nn_matrix.append(ind.long().cpu())
#         dist_matrix.append(dist.cpu())
#         nn_matrix = torch.cat(nn_matrix, dim=0)
#         dist_matrix = torch.cat(dist_matrix, dim=0)
        
#         return nn_matrix, dist_matrix


#     def sample_batch(self):
#         num_image = self.batch_size // self.nn_per_image
#         sampled_queries = np.random.choice(self.num_samples, num_image, replace=False)
#         sampled_indices = self.nn_matrix[sampled_queries].view(-1)

#         return sampled_indices

#     def __len__(self):
#         return self.num_samples // self.batch_size