from __future__ import print_function
from __future__ import division

import torch
import torchvision
import PIL.Image
import numpy as np, os, sys, pandas as pd, csv, copy
import scipy.io
from pathlib import Path

import random
from collections import Counter
import pickle

class All_dataset(torch.utils.data.Dataset):
    def __init__(self, root, ds_list, mode, transform = None, fewshot_dir=None, shot=None):
        self.root = root
        self.ds_list = ds_list
        self.mode = mode
        self.transform = transform
        self.start_y = 0
        self.start_idx = 0
        self.ys, self.im_paths, self.I, self.ds_ID = [], [], [], []
        
        self.shot = shot
        self.fewshot_dir = fewshot_dir
        if self.fewshot_dir is not None:
            Path(self.fewshot_dir).mkdir(parents=True, exist_ok=True)
                
        for ds_name in ds_list:
            if ds_name == 'CUB':
                self.init_CUB()
            elif ds_name == 'Cars':
                self.init_Cars()
            elif ds_name == 'SOP':
                self.init_SOP()
            elif ds_name == 'NAbird':
                self.init_NABird()
            elif ds_name == 'Dogs':
                self.init_dogs()
            elif ds_name == 'Flowers':
                self.init_flowers()
            elif ds_name == 'Aircraft':
                self.init_aircraft()
                
        if "Inshop" in ds_list:
            self.init_Inshop()
                
        self.class_size = torch.Tensor([v for k,v in sorted(Counter(self.ys).items())])

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        def img_load(index):
            im = PIL.Image.open(self.im_paths[index]).convert('RGB')
            # convert gray to rgb
            if self.transform is not None:
                im = self.transform(im)
            return im

        im = img_load(index)
        target = self.ys[index]
        ds_ID = self.ds_ID[index]

        return im, target, index, ds_ID
    
    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]
                    
    def init_CUB(self):
        root = self.root + '/CUB_200_2011'
        if self.mode == 'train':
            classes = range(0,100)
        elif self.mode == 'eval' or self.mode == 'query' or self.mode == 'gallery':
            classes = range(100,200)
        else:
            classes = range(0,200)
        
        if self.mode != 'train' or self.shot is None:
            index = 0
            for i in torchvision.datasets.ImageFolder(root = 
                    os.path.join(root, 'images')).imgs:
                # i[1]: label, i[0]: root
                y = i[1]
                # fn needed for removing non-images starting with `._`
                fn = os.path.split(i[0])[1]
                if y in classes and fn[:2] != '._':
                    self.ys += [int(y) + self.start_y] 
                    self.I += [index + self.start_idx]
                    self.im_paths.append(os.path.join(root, i[0]))
                    self.ds_ID += [0]
                    index += 1
        else:            
            preprocessed = os.path.join(self.fewshot_dir, "dataset_{}_shot_{}.pkl".format('CUB', self.shot))
            if not os.path.exists(preprocessed):
                index = 0
                class_index = {}                
                for i in torchvision.datasets.ImageFolder(root = 
                        os.path.join(root, 'images')).imgs:
                    # i[1]: label, i[0]: root
                    y = i[1]
                    # fn needed for removing non-images starting with `._`
                    fn = os.path.split(i[0])[1]
                    if y in classes and fn[:2] != '._':
                        if y not in class_index:
                            class_index[y] = []
                        class_index[y].append([y, index, os.path.join(root, i[0]), 0])
                        index += 1
                        
                data = {"ys":[], "I":[], "im_paths":[], "ds_ID":[]}
                for label, items in class_index.items():
                    if len(items) >= self.shot:
                        sampled_items = random.sample(items, self.shot)
                    else:
                        sampled_items = random.choices(items, k=self.shot)
                            
                    for j in range(self.shot):
                        data["ys"].append(sampled_items[j][0])
                        data["I"].append(sampled_items[j][1])
                        data["im_paths"].append(sampled_items[j][2])
                        data["ds_ID"].append(sampled_items[j][3])                      
                    
                print("Saving preprocessed few-shot data to {}".format(preprocessed))
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
                    
            else:            
                print("Loading preprocessed few-shot data from {}".format(preprocessed))
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                
            self.ys += [int(y) + self.start_y for y in data["ys"]]
            self.I += [index + self.start_idx for index in data["I"]]
            self.im_paths += data["im_paths"]
            self.ds_ID += data["ds_ID"]

        self.start_y = len(list(set(self.ys)))
        self.start_idx = len(list(set(self.I)))
        
    def init_Cars(self):
        root = self.root + '/cars196'
        if self.mode == 'train':
            classes = range(0,98)
        elif self.mode == 'eval' or self.mode == 'query' or self.mode == 'gallery':
            classes = range(98,196)
        else:
            classes = range(0,196)
            
        annos_fn = 'cars_annos.mat'
        cars = scipy.io.loadmat(os.path.join(root, annos_fn))
        ys = [int(a[5][0] - 1) for a in cars['annotations'][0]]
        im_paths = [a[0][0] for a in cars['annotations'][0]]
        
        if self.mode != 'train' or self.shot is None:
            index = 0
            for im_path, y in zip(im_paths, ys):
                if y in classes: # choose only specified classes
                    self.ys += [int(y) + self.start_y] 
                    self.I += [index + self.start_idx]
                    self.im_paths.append(os.path.join(root, im_path))
                    self.ds_ID += [1]
                    index += 1
                
        else:
            preprocessed = os.path.join(self.fewshot_dir, "dataset_{}_shot_{}.pkl".format('Cars', self.shot))
            if not os.path.exists(preprocessed):
                index = 0
                class_index = {}                
                for im_path, y in zip(im_paths, ys):
                    if y in classes: # choose only specified classes
                        if y not in class_index:
                            class_index[y] = []
                        class_index[y].append([y, index, os.path.join(root, im_path), 1])
                        index += 1
                        
                data = {"ys":[], "I":[], "im_paths":[], "ds_ID":[]}
                for label, items in class_index.items():
                    if len(items) >= self.shot:
                        sampled_items = random.sample(items, self.shot)
                    else:
                        sampled_items = random.choices(items, k=self.shot)
                            
                    for j in range(self.shot):
                        data["ys"].append(sampled_items[j][0])
                        data["I"].append(sampled_items[j][1])
                        data["im_paths"].append(sampled_items[j][2])
                        data["ds_ID"].append(sampled_items[j][3])              
                    
                print("Saving preprocessed few-shot data to {}".format(preprocessed))
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
                    
            else:            
                print("Loading preprocessed few-shot data from {}".format(preprocessed))
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                
            self.ys += [int(y) + self.start_y for y in data["ys"]]
            self.I += [index + self.start_idx for index in data["I"]]
            self.im_paths += data["im_paths"]
            self.ds_ID += data["ds_ID"]
                
                
        self.start_y = len(list(set(self.ys)))
        self.start_idx = len(list(set(self.I)))
        
    def init_SOP(self):
        root = self.root + '/Stanford_Online_Products'
        if self.mode == 'train':
            classes = range(0,11318)
        elif self.mode == 'eval' or self.mode == 'query' or self.mode == 'gallery':
            classes = range(11318,22634)
        else:
            classes = range(0,22634)
            
        if self.mode == 'train' or self.mode == 'eval' or self.mode == 'query' or self.mode == 'gallery':
            if self.mode != 'train' or self.shot is None:
                metadata = open(os.path.join(root, 'Ebay_train.txt' if classes == range(0, 11318) else 'Ebay_test.txt'))
                for i, (index, y, _, path) in enumerate(map(str.split, metadata)):
                    if i > 0:
                        if int(y)-1 in classes:
                            self.ys += [int(y)-1 + self.start_y] 
                            self.I += [int(index)-1 + self.start_idx]
                            self.im_paths.append(os.path.join(root, path))
                            self.ds_ID += [2]
                        
            else:
                preprocessed = os.path.join(self.fewshot_dir, "dataset_{}_shot_{}.pkl".format('SOP', self.shot))
                if not os.path.exists(preprocessed):
                    class_index = {}                
                    metadata = open(os.path.join(root, 'Ebay_train.txt'))
                    for i, (index, y, _, path) in enumerate(map(str.split, metadata)):
                        if i > 0:
                            if int(y)-1 in classes:
                                if y not in class_index:
                                    class_index[y] = []
                                class_index[y].append([int(y)-1, int(index)-1, os.path.join(root, path), 2])
                            
                    data = {"ys":[], "I":[], "im_paths":[], "ds_ID":[]}
                    for label, items in class_index.items():
                        if len(items) >= self.shot:
                            sampled_items = random.sample(items, self.shot)
                        else:
                            sampled_items = random.choices(items, k=self.shot)
                                
                        for j in range(self.shot):
                            data["ys"].append(sampled_items[j][0])
                            data["I"].append(sampled_items[j][1])
                            data["im_paths"].append(sampled_items[j][2])
                            data["ds_ID"].append(sampled_items[j][3])             
                        
                    print("Saving preprocessed few-shot data to {}".format(preprocessed))
                    with open(preprocessed, "wb") as file:
                        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
                        
                else:            
                    print("Loading preprocessed few-shot data from {}".format(preprocessed))
                    with open(preprocessed, "rb") as file:
                        data = pickle.load(file)
                    
                self.ys += [int(y) + self.start_y for y in data["ys"]]
                self.I += [index + self.start_idx for index in data["I"]]
                self.im_paths += data["im_paths"]
                self.ds_ID += data["ds_ID"]
                
        else:
            for txt_file in ['Ebay_train.txt', 'Ebay_test.txt']:
                metadata = open(os.path.join(root, txt_file))
                for i, (image_id, class_id, _, path) in enumerate(map(str.split, metadata)):
                    if i > 0:
                        if int(class_id)-1 in classes:
                            self.ys += [int(class_id)-1 + self.start_y] 
                            self.I += [int(image_id)-1 + self.start_idx]
                            self.im_paths.append(os.path.join(root, path))
                            self.ds_ID += [2]
                
        self.start_y = len(list(set(self.ys)))
        self.start_idx = len(list(set(self.I)))
        
    def init_Inshop(self):
        root = self.root + '/Inshop_Clothes'
        data_info = np.array(pd.read_table(root +'/Eval/list_eval_partition.txt', header=1, delim_whitespace=True))[:,:]
        #Separate into training dataset and query/gallery dataset for testing.
        train, query, gallery = data_info[data_info[:,2]=='train'][:,:2], data_info[data_info[:,2]=='query'][:,:2], data_info[data_info[:,2]=='gallery'][:,:2]

        #Generate conversions
        t_lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in train[:,1]])))}
        train[:,1] = np.array([t_lab_conv[int(x.split('_')[-1])] for x in train[:,1]])
        
        qg_lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in np.concatenate([query[:,1], gallery[:,1]])])))}
        query[:,1]   = np.array([qg_lab_conv[int(x.split('_')[-1])] for x in query[:,1]])
        gallery[:,1] = np.array([qg_lab_conv[int(x.split('_')[-1])] for x in gallery[:,1]])
        
        mode2data = {'train': train, 'query': query, 'gallery': gallery}
        if self.mode == 'all':
            for mode in mode2data.keys():
                for image_id, (img_path, class_id) in enumerate(mode2data[mode]):
                    if mode == 'train':
                        self.ys += [int(class_id) + self.start_y]
                    else:
                        self.ys += [int(class_id) + self.start_y + 3997]
                    if self.mode == 'gallery':
                        self.I += [int(image_id) + self.start_idx + len(query)]
                    else:
                        self.I += [int(image_id) + self.start_idx]
                    self.im_paths.append(os.path.join(root, 'Img', img_path))
                    self.ds_ID += [3]
        else:
            if self.mode != 'train' or self.shot is None:
                for image_id, (img_path, class_id) in enumerate(mode2data[self.mode]):
                    self.ys += [int(class_id) + self.start_y]
                    if self.mode == 'gallery':
                        self.I += [int(image_id) + self.start_idx + len(query)]
                    else:
                        self.I += [int(image_id) + self.start_idx]
                    self.im_paths.append(os.path.join(root, 'Img', img_path))
                    self.ds_ID += [3]
                
            else:
                preprocessed = os.path.join(self.fewshot_dir, "dataset_{}_shot_{}.pkl".format('Inshop', self.shot))
                if not os.path.exists(preprocessed):
                    class_index = {}
                    for image_id, (img_path, class_id) in enumerate(mode2data[self.mode]):
                        if class_id not in class_index:
                            class_index[class_id] = []
                        class_index[class_id].append([int(class_id), int(image_id), os.path.join(root, 'Img', img_path), 3])
                            
                    data = {"ys":[], "I":[], "im_paths":[], "ds_ID":[]}
                    for label, items in class_index.items():
                        if len(items) >= self.shot:
                            sampled_items = random.sample(items, self.shot)
                        else:
                            sampled_items = random.choices(items, k=self.shot)
                                
                        for j in range(self.shot):
                            data["ys"].append(sampled_items[j][0])
                            data["I"].append(sampled_items[j][1])
                            data["im_paths"].append(sampled_items[j][2])
                            data["ds_ID"].append(sampled_items[j][3])       
                            
                    print("Saving preprocessed few-shot data to {}".format(preprocessed))
                    with open(preprocessed, "wb") as file:
                        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
                        
                else:            
                    print("Loading preprocessed few-shot data from {}".format(preprocessed))
                    with open(preprocessed, "rb") as file:
                        data = pickle.load(file)
                    
                self.ys += [int(y) + self.start_y for y in data["ys"]]
                self.I += [index + self.start_idx for index in data["I"]]
                self.im_paths += data["im_paths"]
                self.ds_ID += data["ds_ID"]
                    
        self.start_y = len(list(set(self.ys)))
        self.start_idx = len(list(set(self.I)))
        
    def init_NABird(self):
        root = self.root + '/nabirds'
        if self.mode == 'train':
            classes = range(0, 278)
        elif self.mode == 'eval' or self.mode == 'query' or self.mode == 'gallery':
            classes = range(278, 555)
        else:
            classes = range(0, 555)
                
        image_paths = pd.read_csv(os.path.join(root,'images.txt'),sep=' ',names=['img_id','filepath'])
        image_class_labels = pd.read_csv(os.path.join(root,'image_class_labels.txt'),sep=' ',names=['img_id','target'])
        label_list = list(set(image_class_labels['target']))
        label_list = sorted(label_list)
        label_map = {k: i for i, k in enumerate(label_list)}
        train_test_split = pd.read_csv(os.path.join(root, 'train_test_split.txt'), sep=' ', names=['img_id', 'is_training_img'])
        data = image_paths.merge(image_class_labels, on='img_id')
        data = data.merge(train_test_split, on='img_id')
        
        if self.mode != 'train' or self.shot is None:
            index = 0
            for i, row in data.iterrows():
                file_path = os.path.join(os.path.join(root,'images'), row['filepath'])
                y = int(label_map[row['target']])
                if y in classes: # choose only specified classes
                    self.ys += [int(y) + self.start_y]
                    self.I += [index + self.start_idx]
                    self.im_paths.append(os.path.join(root, file_path))
                    self.ds_ID += [4]
                    index += 1
                
        else:
            index = 0
            preprocessed = os.path.join(self.fewshot_dir, "dataset_{}_shot_{}.pkl".format('NABird', self.shot))
            if not os.path.exists(preprocessed):
                class_index = {}                
                for i, row in data.iterrows():
                    file_path = os.path.join(os.path.join(root,'images'), row['filepath'])
                    y = int(label_map[row['target']])
                    if y in classes: # choose only specified classes
                        if y not in class_index:
                            class_index[y] = []
                        class_index[y].append([int(y), int(index), os.path.join(root, file_path), 4])
                        index += 1
                        
                data = {"ys":[], "I":[], "im_paths":[], "ds_ID":[]}
                for label, items in class_index.items():
                    if len(items) >= self.shot:
                        sampled_items = random.sample(items, self.shot)
                    else:
                        sampled_items = random.choices(items, k=self.shot)
                            
                    for j in range(self.shot):
                        data["ys"].append(sampled_items[j][0])
                        data["I"].append(sampled_items[j][1])
                        data["im_paths"].append(sampled_items[j][2])
                        data["ds_ID"].append(sampled_items[j][3])          
                    
                print("Saving preprocessed few-shot data to {}".format(preprocessed))
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
                    
            else:            
                print("Loading preprocessed few-shot data from {}".format(preprocessed))
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                
            self.ys += [int(y) + self.start_y for y in data["ys"]]
            self.I += [index + self.start_idx for index in data["I"]]
            self.im_paths += data["im_paths"]
            self.ds_ID += data["ds_ID"]
                
        self.start_y = len(list(set(self.ys)))
        self.start_idx = len(list(set(self.I)))
                
    def init_dogs(self):
        root = self.root + '/dogs'
        if self.mode == 'train':
            classes = range(0, 60)
        elif self.mode == 'eval' or self.mode == 'query' or self.mode == 'gallery':
            classes = range(60, 120) 
        else:
            classes = range(0, 120)                
                
        anno_data = scipy.io.loadmat(os.path.join(root,'file_list.mat'))
        if self.mode != 'train' or self.shot is None:
            index = 0
            for file,label in zip(anno_data['file_list'],anno_data['labels']):
                file_path = os.path.join(os.path.join(root,'Images'),file[0][0])
                y = int(label[0])-1
                if y in classes: # choose only specified classes
                    self.ys += [int(y) + self.start_y] 
                    self.I += [index + self.start_idx]
                    self.im_paths.append(os.path.join(root, file_path))
                    self.ds_ID += [5]
                    index += 1
                
        else:
            index = 0
            preprocessed = os.path.join(self.fewshot_dir, "dataset_{}_shot_{}.pkl".format('Dogs', self.shot))
            if not os.path.exists(preprocessed):
                class_index = {}                
                for file,label in zip(anno_data['file_list'],anno_data['labels']):
                    file_path = os.path.join(os.path.join(root,'Images'),file[0][0])
                    y = int(label[0])-1
                    if y in classes: # choose only specified classes
                        if y not in class_index:
                            class_index[y] = []
                        class_index[y].append([int(y), int(index), os.path.join(root, file_path), 5])
                        index += 1
                        
                data = {"ys":[], "I":[], "im_paths":[], "ds_ID":[]}
                for label, items in class_index.items():
                    if len(items) >= self.shot:
                        sampled_items = random.sample(items, self.shot)
                    else:
                        sampled_items = random.choices(items, k=self.shot)
                            
                    for j in range(self.shot):
                        data["ys"].append(sampled_items[j][0])
                        data["I"].append(sampled_items[j][1])
                        data["im_paths"].append(sampled_items[j][2])
                        data["ds_ID"].append(sampled_items[j][3])             
                    
                print("Saving preprocessed few-shot data to {}".format(preprocessed))
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
                    
            else:            
                print("Loading preprocessed few-shot data from {}".format(preprocessed))
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                
            self.ys += [int(y) + self.start_y for y in data["ys"]]
            self.I += [index + self.start_idx for index in data["I"]]
            self.im_paths += data["im_paths"]
            self.ds_ID += data["ds_ID"]
                
        self.start_y = len(list(set(self.ys)))
        self.start_idx = len(list(set(self.I)))
    
    def init_flowers(self):
        root = self.root + '/flowers'
        if self.mode == 'train':
            classes = range(0, 51)
        elif self.mode == 'eval' or self.mode == 'query' or self.mode == 'gallery':
            classes = range(51, 102) 
        else:
            classes = range(0, 102)
                
        imagelabels = scipy.io.loadmat(os.path.join(root,'imagelabels.mat'))
        imagelabels = imagelabels['labels'][0]
        train_val_split = scipy.io.loadmat(os.path.join(root,'setid.mat'))
        train_data = train_val_split['trnid'][0].tolist()
        val_data = train_val_split['valid'][0].tolist()
        test_data = train_val_split['tstid'][0].tolist()
        images_root = os.path.join(root,'jpg')
        all_data = train_data + val_data + test_data


        if self.mode != 'train' or self.shot is None:
            index = 0
            for data in all_data:
                file_path = os.path.join(images_root,f'image_{str(data).zfill(5)}.jpg')
                y = int(imagelabels[int(data)-1])-1
                if y in classes: # choose only specified classes
                    self.ys += [int(y) + self.start_y] 
                    self.I += [index + self.start_idx]
                    self.im_paths.append(os.path.join(root, file_path))
                    self.ds_ID += [6]
                    index += 1

        else:
            index = 0
            preprocessed = os.path.join(self.fewshot_dir, "dataset_{}_shot_{}.pkl".format('Flowers', self.shot))
            if not os.path.exists(preprocessed):
                class_index = {}                
                for data in all_data:
                    file_path = os.path.join(images_root,f'image_{str(data).zfill(5)}.jpg')
                    y = int(imagelabels[int(data)-1])-1
                    if y in classes: # choose only specified classes
                        if y not in class_index:
                            class_index[y] = []
                        class_index[y].append([int(y), int(index), os.path.join(root, file_path), 6])
                        index += 1
                        
                data = {"ys":[], "I":[], "im_paths":[], "ds_ID":[]}
                for label, items in class_index.items():
                    if len(items) >= self.shot:
                        sampled_items = random.sample(items, self.shot)
                    else:
                        sampled_items = random.choices(items, k=self.shot)
                            
                    for j in range(self.shot):
                        data["ys"].append(sampled_items[j][0])
                        data["I"].append(sampled_items[j][1])
                        data["im_paths"].append(sampled_items[j][2])
                        data["ds_ID"].append(sampled_items[j][3])             
                    
                print("Saving preprocessed few-shot data to {}".format(preprocessed))
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
                    
            else:            
                print("Loading preprocessed few-shot data from {}".format(preprocessed))
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                
            self.ys += [int(y) + self.start_y for y in data["ys"]]
            self.I += [index + self.start_idx for index in data["I"]]
            self.im_paths += data["im_paths"]
            self.ds_ID += data["ds_ID"]
                
        self.start_y = len(list(set(self.ys)))
        self.start_idx = len(list(set(self.I)))
                
    def init_aircraft(self):
        root = os.path.join(self.root,'fgvc-aircraft-2013b','data')
        if self.mode == 'train':
            classes = range(0, 50)
        elif self.mode == 'eval' or self.mode == 'query' or self.mode == 'gallery':
            classes = range(50, 100) 
        else:
            classes = range(0, 100)
         
        data_file = os.path.join(root,'images_variant_trainvaltest.txt')
        if not os.path.isfile(data_file):
            filenames = ['images_variant_trainval.txt', 'images_variant_test.txt']
            with open(data_file, 'w') as data_file:
                for filename in filenames:
                    with open(filename) as file:
                        for line in file:
                            data_file.write(line)
                            
        data_file = os.path.join(root,'images_variant_trainvaltest.txt')    
        classes_names = set()
        with open(data_file,'r') as f:
            for line in f:
                class_name = '_'.join(line.split()[1:])
                classes_names.add(class_name)
        classes_names = sorted(list(classes_names))
        class_to_idx = {name:ind for ind,name in enumerate(classes_names)}
        
        if self.mode != 'train' or self.shot is None:
            index = 0
            with open(data_file, 'r') as f:
                images_root = os.path.join(root,'images')
                for line in f:
                    image_file = line.split()[0]
                    class_name = '_'.join(line.split()[1:])
                    file_path = os.path.join(images_root, f'{image_file}.jpg')
                    y = class_to_idx[class_name]
                    if y in classes: # choose only specified classes
                        self.ys += [int(y) + self.start_y] 
                        self.I += [index  + self.start_idx]
                        self.im_paths.append(os.path.join(root, file_path))
                        self.ds_ID += [7]
                        index += 1
                    
        else:
            index = 0
            preprocessed = os.path.join(self.fewshot_dir, "dataset_{}_shot_{}.pkl".format('Aircraft', self.shot))
            if not os.path.exists(preprocessed):
                class_index = {}                
                with open(data_file, 'r') as f:
                    images_root = os.path.join(root,'images')
                    for line in f:
                        image_file = line.split()[0]
                        class_name = '_'.join(line.split()[1:])
                        file_path = os.path.join(images_root, f'{image_file}.jpg')
                        y = class_to_idx[class_name]
                        if y in classes: # choose only specified classes
                            if y not in class_index:
                                class_index[y] = []
                            class_index[y].append([int(y), int(index), os.path.join(root, file_path), 7])
                            index += 1
                        
                data = {"ys":[], "I":[], "im_paths":[], "ds_ID":[]}
                for label, items in class_index.items():
                    if len(items) >= self.shot:
                        sampled_items = random.sample(items, self.shot)
                    else:
                        sampled_items = random.choices(items, k=self.shot)

                    for j in range(self.shot):
                        data["ys"].append(sampled_items[j][0])
                        data["I"].append(sampled_items[j][1])
                        data["im_paths"].append(sampled_items[j][2])
                        data["ds_ID"].append(sampled_items[j][3])        
                    
                print("Saving preprocessed few-shot data to {}".format(preprocessed))
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
                    
            else:            
                print("Loading preprocessed few-shot data from {}".format(preprocessed))
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    
        self.start_y = len(list(set(self.ys)))
        self.start_idx = len(list(set(self.I)))