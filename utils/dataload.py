import numpy as np
import os
from os.path import join

import torch
import torch.nn.functional as F
from torch.utils import data
from glob import glob
import networkx as nx
import pandas as pd
import json
import walker

class SingleIsovistsCubicasa(data.Dataset):

    def __init__(self, root, train=True, transform=None, filename='cubicasa_isovist_numpy'):
        self.root = os.path.expanduser(root)
        self.train = train
        self.filename=filename
        self.transform = transform

        if self.train:
            self.train_data = np.load(join(self.root, self.filename, 'x_train.npy'))
            self.data = self.train_data
            try:
                self.train_labels = np.load(join(self.root, self.filename, 'y_train.npy'))
            except:
                print('no label found, assign 0s')
                self.train_labels = np.zeros(self.train_data.shape[0])
            print(np.shape(self.train_data), np.shape(self.train_labels))
        else:
            self.eval_data = np.load(join(self.root, self.filename, 'x_eval.npy'))
            self.data = self.eval_data
            try:
                self.eval_labels = np.load(join(self.root, self.filename, 'y_eval.npy'))
            except:
                print('no label found, assign 0s')
                self.eval_labels = np.zeros(self.eval_data.shape[0])
            print(np.shape(self.eval_data), np.shape(self.eval_labels))

    def __getitem__(self, index):
        if self.train:
            isovist, target = self.train_data[index], self.train_labels[index]
        else:
            isovist, target = self.eval_data[index], self.eval_data[index]

        if self.transform is not None:
            isovist = self.transform(isovist)
        return isovist, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.eval_data)
        


class PairedIsovistsCubicasa(data.Dataset):

    def __init__(self, root, name=None, train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.filename = f'{name}.npy'
        self.transform = transform
        self.data = np.load(join(self.root, self.filename))
        self.train_data = self.data[0]
        self.train_labels = self.data[1]

    def __getitem__(self, index):
        data, target = self.train_data[index], self.train_labels[index]

        if self.transform is not None:
            data = self.transform(data)
            target = self.transform(target)

        return data, target

    def __len__(self):
        return len(self.train_data)


class CycleIsovistsCubicasa(data.Dataset):

    def __init__(self, root, name=None, train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.filename_x = f'x.npy'
        self.filename_y = f'y.npy'
        self.transform = transform
        # self.data = np.load(join(self.root, self.filename))
        self.train_data = np.load(join(self.root, self.filename_x))
        self.train_labels = np.load(join(self.root, self.filename_y))

    def __getitem__(self, index):
        data, target = self.train_data[index], self.train_labels[index]

        if self.transform is not None:
            data = self.transform(data)
            target = self.transform(target)

        return data, target

    def __len__(self):
        return len(self.train_data)
    
    
class SequenceIsovistCubicasa(data.Dataset):

    def __init__(self, root, name='sample_train', train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.filename = f'{name}.npy'
        self.transform = transform
        self.data = np.load(join(self.root, self.filename))
        self.locs, self.isovists = np.split(self.data, [2], axis=2)

    def __getitem__(self, index):
        locs = self.locs[index]
        isovists = self.isovists[index]

        if self.transform is not None:
            locs = self.transform(locs)
            isovists = self.transform(isovists)

        return locs, isovists
    
    def __len__(self):
        return len(self.data)
    
    def get_seq_length(self):
        return self.data.shape[1]
    

class SequenceSamplerCubicasa(data.Dataset):

    def __init__(self, root, seq_length=10, seq_num=5, p=0.25, q=0.001, train=True, transform=None, transform_isovist=None):
        self.root = os.path.expanduser(root)
        self.train = train
        if self.train:
            self.csvs = sorted(glob(join(self.root, 'training', '*.csv')))
            self.jsons = sorted(glob(join(self.root, 'training', '*.json')))
        else:
            self.csvs = sorted(glob(join(self.root, 'eval', '*.csv')))
            self.jsons = sorted(glob(join(self.root, 'eval', '*.json')))
        self.transform = transform
        self.transform_isovist = transform_isovist
        self.seq_length = seq_length
        self.seq_num = seq_num
        self.p = p
        self.q = q
        self.rng = np.random.default_rng()

    def read_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        scale = df.at[0, 'scale']
        isovist_rad = df.at[0, 'isovist_rad']
        data = (df.to_numpy()).astype(np.float32)
        loc = data[:, 2:4]*scale/isovist_rad
        isovist = data[:,4:]
        return scale, isovist_rad, loc, isovist
    
    def read_json(self, json_path):
        with open(json_path) as f:
            net = json.load(f)
        new_net = {}
        for key in net:
            new_net[int(key)] = net[key]
        G = nx.Graph(new_net)
        return G
    
    def get_rel_loc(self, nodes, loc, d):
        rel_loc = []
        p_loc = None
        for node in nodes:
            i_loc = loc[node]
            if p_loc is None:
                rel_loc.append(0)
            else:
                rel = (i_loc - p_loc).round(decimals=2)
                if (rel == np.array([d, 0.], dtype=np.float32)).all():
                    rel_loc.append(1)
                elif (rel == np.array([d, d], dtype=np.float32)).all():
                    rel_loc.append(2)
                elif (rel == np.array([0., d], dtype=np.float32)).all():
                    rel_loc.append(3)
                elif (rel == np.array([-d, d], dtype=np.float32)).all():
                    rel_loc.append(4)
                elif (rel == np.array([-d, 0.], dtype=np.float32)).all():
                    rel_loc.append(5)
                elif (rel == np.array([-d, -d], dtype=np.float32)).all():
                    rel_loc.append(6)
                elif (rel == np.array([0., -d], dtype=np.float32)).all():
                    rel_loc.append(7)
                elif (rel == np.array([d, -d], dtype=np.float32)).all():
                    rel_loc.append(8)
            p_loc = i_loc
        return rel_loc
    

    def __getitem__(self, index):
        loc_samples = []
        isovist_samples = []
        scale, isovist_rad, loc, isovists = self.read_csv(self.csvs[index])
        d = 100.0/isovist_rad
        G = self.read_json(self.jsons[index])
        node_list = np.asarray(G.nodes())
        graph_size = len(node_list)
        choice = self.rng.choice(graph_size, size=self.seq_num, replace=False)
        X = walker.random_walks(G, n_walks=1, walk_len=self.seq_length,
                                start_nodes=choice, p=self.p, q=self.q, verbose=False)
        for sample in X:
            sample = node_list[sample]
            loc_samples.append(np.stack(self.get_rel_loc(sample, loc, d)))
            isovist_samples.append(np.stack([isovists[i] for i in sample]))
        loc_samples = np.stack(loc_samples)
        isovist_samples = np.stack(isovist_samples)

        if self.transform is not None:
            loc_samples = self.transform(loc_samples)
            isovist_samples = self.transform(isovist_samples)
            if self.transform_isovist is not None:
                isovist_samples = self.transform_isovist(isovist_samples)
        return loc_samples, isovist_samples

    def __len__(self):
        return len(self.csvs)
    
    

class ToTanhRange(object):
    # transform to -1, 1
    def __call__(self, sample):
        sample = (sample - 0.5) * 2
        return sample

class ToTensor(object):
    #transform to tensor
    def __call__(self, sample):
        return torch.from_numpy(sample)

class Resize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, sample):
        sample.unsqueeze_(0)
        return F.interpolate(sample, self.size, mode='linear', align_corners=False).squeeze_(0)

class RandomRotate(object):
    # random shift 0, 90, 180, 270 degress
    # 0, 64, 128, 192
    def __init__(self, size):
        self.shift = int(round(size/4))
    def __call__(self, sample):
        dir = np.random.randint(4)
        return torch.roll(sample, dir * self.shift, dims=-1)
    
class RandomRotate_b(object):
    # random shift 0, 22.5, 45, 67.6 90, 180, 270 degress
    # 0, 64, 128, 192
    def __init__(self, size):
        self.shift = int(round(size/16))
    def __call__(self, sample):
        dir = np.random.randint(16)
        return torch.roll(sample, dir * self.shift, dims=-1)

class RandomFlip(object):
    # random flip the data
    def __call__(self, sample):
        if np.random.uniform(0, 1) < 0.5:
            return sample
        else:
            return torch.flip(sample, dims=(-1,))
