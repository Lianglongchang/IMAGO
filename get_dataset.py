from sklearn.preprocessing import minmax_scale
from scipy import sparse
import os
import pickle
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset


class multimodesDataset(Dataset):
    def __init__(self, num_modes, modes_features, labels):
        self.modes_features = modes_features
        self.labels = labels
        self.num_modes = num_modes
        
    def __getitem__(self, index):
        modes_features = []
        for i in range(self.num_modes):
            modes_features.append(self.modes_features[i][index])
        return modes_features, self.labels[index]
    
    def __len__(self):
        return self.modes_features[0].size(0)
    
class multimodesFullDataset(Dataset):
    def __init__(self, num_modes, modes_features):
        self.modes_features = modes_features
        self.num_modes = num_modes
        
    def __getitem__(self, index):
        modes_features = []
        for i in range(self.num_modes):
            modes_features.append(self.modes_features[i][index])
        return modes_features
    
    def __len__(self):
        return self.modes_features[0].size(0)

def get_ssl_datasets(args):
    'retrun self-supervised data'
    #=========load feature==========
    feature_file = args.dataset_dir + '/features.npy'
    with open(feature_file, 'rb') as f:
        Z = pickle.load(f)
    Z = minmax_scale(Z)
    
    #=========load PPMIs========
    ppmi = args.org + '_net_combined.mat'
    pf = os.path.join(args.dataset_dir, ppmi)
    N = sio.loadmat(pf, squeeze_me=True)
    X = N['Net'].todense()
    X = np.asarray(X)
    X = minmax_scale(X)
    #X = np.hstack((X,Z))

    full_X = torch.from_numpy(X)
    full_X = full_X.float()
    
    full_Z = torch.from_numpy(Z)
    full_Z = full_Z.float()
    modefeature_lens = [X.shape[1], Z.shape[1]]
    
    full_dataset = multimodesFullDataset(2, [full_X, full_Z])
    
    return full_dataset, modefeature_lens
    

def get_datasets(args):
    #===========load annot============
    Annot = sio.loadmat(args.dataset_dir + '/' + args.org + '_annot.mat', squeeze_me=True)
    
    #=========load feature==========
    feature_file = args.dataset_dir + '/features.npy'
    with open(feature_file, 'rb') as f:
        Z = pickle.load(f)
    Z = minmax_scale(Z)
    
    #=========load PPMIs========
    ppmi = args.org + '_net_combined.mat'
    pf = os.path.join(args.dataset_dir, ppmi)
    N = sio.loadmat(pf, squeeze_me=True)
    X = N['Net'].todense()
    X = np.asarray(X)
    X = minmax_scale(X)
    #X = np.hstack((X,Z))
    
    train_idx = Annot['indx'][args.aspect].tolist()['train'].tolist().tolist()
    valid_idx = Annot['indx'][args.aspect].tolist()['valid'].tolist().tolist()
    test_idx = Annot['indx'][args.aspect].tolist()['test'].tolist().tolist()
    train_idx.extend(valid_idx)
    
    X_train = X[train_idx]
    labels_train = np.array(Annot['GO'][args.aspect].tolist()['train'].tolist())
    labels_valid = np.array(Annot['GO'][args.aspect].tolist()['valid'].tolist())
    labels_train = np.vstack((labels_train, labels_valid))
    print('labels_train shape = ', labels_train.shape)
    
    X_test = X[test_idx]
    labels_test = np.array(Annot['GO'][args.aspect].tolist()['test'].tolist())
    
    X_train = torch.from_numpy(X_train)
    labels_train = torch.from_numpy(labels_train)
    X_test = torch.from_numpy(X_test)
    labels_test = torch.from_numpy(labels_test)
    
    X_train = X_train.float()
    X_test = X_test.float()
    labels_train = labels_train.float()
    labels_test = labels_test.float()
    
    Z_train = Z[train_idx]
    Z_train = torch.from_numpy(Z_train)
    Z_train = Z_train.float()
    
    Z_test = Z[test_idx]
    Z_test = torch.from_numpy(Z_test)
    Z_test = Z_test.float()
    
    train_dataset = multimodesDataset(2, [X_train, Z_train], labels_train)
    test_dataset = multimodesDataset(2, [X_test, Z_test], labels_test)
    modefeature_lens = [X_train.shape[1], Z_train.shape[1]]
    print('X_train shape = ', X_train.shape)
    
    return train_dataset, test_dataset, modefeature_lens
