# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import scipy.io as sio
import argparse
import os
from scipy import sparse

def _load_network(filename, mtrx='adj'):
    As = []
    if mtrx == 'adj':
        net_names = ['combined']
        graphs = []
        for name in net_names:
            graphs.append(nx.Graph(name=name))
        fRead = open(filename, 'r')
        fRead.readline()
        for line in fRead:
            splitted = line.strip().split()
            prot1 = str(splitted[0])
            # prot1 = prot1.split('.')[1]
            prot2 = str(splitted[1])
            # prot2 = prot2.split('.')[1]
            score = splitted[-1]
            score = float(score)
            if not graphs[0].has_node(prot1):
                graphs[0].add_node(prot1)
            if not graphs[0].has_node(prot2):
                graphs[0].add_node(prot2)
            if score > 0:
                graphs[0].add_edge(prot1, prot2, weight=score)
        fRead.close()
        # for ii in range(0, len(graphs)):
        #    graphs[ii] = nx.relabel_nodes(graphs[ii], string2uniprot)
        String = {}
        String['prot_IDs'] = list(graphs[0].nodes())
        String['nets'] = {}
        for ii in range(0, len(graphs)):
            String['nets'][net_names[ii]] = nx.adjacency_matrix(graphs[ii], nodelist=String['prot_IDs'])
            print (net_names[ii], graphs[ii].order(), graphs[ii].size())
            
        for g_name in String['nets']:
            A = String['nets'][g_name]
            A = A.todense()
            A = np.squeeze(np.asarray(A))
            if A.min() < 0:
                print ("### Negative entries in the matrix are not allowed!")
                A[A < 0] = 0
                print ("### Matrix converted to nonnegative matrix.")
                print
            if (A.T == A).all():
                pass
            else:
                print ("### Matrix not symmetric!")
                A = A + A.T
                print ("### Matrix converted to symmetric.")
            As.append(A)
            print('max_score=', np.max(A))
    else:
        print ("### Wrong mtrx type. Possible: {'adj', 'inc'}.")
    
    for A in As:
        A = A - np.diag(np.diag(A))
        A = A + np.diag(A.sum(axis=1) == 0)

    return As


def load_networks(filename, mtrx='adj'):
    """
    Function for loading Mashup files
    Files can be downloaded from:
        http://cb.csail.mit.edu/cb/mashup/
    """
    Nets = _load_network(filename, mtrx)

    return Nets


def _net_normalize(X):
    """
    Normalizing networks according to node degrees.
    """
    if X.min() < 0:
        print ("### Negative entries in the matrix are not allowed!")
        X[X < 0] = 0
        print ("### Matrix converted to nonnegative matrix.")
        print
    if (X.T == X).all():
        pass
    else:
        print ("### Matrix not symmetric.")
        X = X + X.T - np.diag(np.diag(X))
        print ("### Matrix converted to symmetric.")

    # normalizing the matrix
    deg = X.sum(axis=1).flatten()
    deg = np.divide(1., np.sqrt(deg))
    deg[np.isinf(deg)] = 0
    D = np.diag(deg)
    X = D.dot(X.dot(D))

    return X


def net_normalize(Net):
    """
    Normalize Nets or list of Nets.
    """
    if isinstance(Net, list):
        for i in range(len(Net)):
            Net[i] = _net_normalize(Net[i])
        print (Net[0].shape)
    else:
        Net = _net_normalize(Net)

    return Net


def _scaleSimMat(A):
    """Scale rows of similarity matrix"""
    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=0) == 0)
    col = A.sum(axis=0)
    A = A.astype(np.float)/col[:, None]

    return A


def RWR(A, K=3, alpha=0.98):
    """Random Walk on graph"""
    A = _scaleSimMat(A)
    # Random surfing
    n = A.shape[0]
    P0 = np.eye(n, dtype=float)
    P = P0.copy()
    M = np.zeros((n, n), dtype=float)
    for i in range(0, K):
        P = alpha*np.dot(P, A) + (1. - alpha)*P0
        M = M + P

    return M


def PPMI_matrix(M):
    """ Compute Positive Pointwise Mutual Information Matrix"""
    M = _scaleSimMat(M)
    n = M.shape[0]
    col = np.asarray(M.sum(axis=0), dtype=float)
    col = col.reshape((1, n))
    row = np.asarray(M.sum(axis=1), dtype=float)
    row = row.reshape((n, 1))
    D = np.sum(col)

    np.seterr(all='ignore')
    PPMI = np.log(np.divide(D*M, np.dot(row, col)))
    PPMI[np.isnan(PPMI)] = 0
    PPMI[PPMI < 0] = 0

    return PPMI


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', '--data_path', default= r'~/IMAGO/Dataset', type = str, help = "the data path")
    parser.add_argument('-snf', '--string_network_file', default= r'~/IMAGO/Dataset/mouse/10090.protein.links.detailed.v11.5.txt', type = str, help = "the input string PPI network file")
    parser.add_argument('-org', '--organism', default= 'mouse', type = str, help = "the output_directory")
    
    
    margs = parser.parse_args()
    filename = os.path.join(margs.data_path, margs.organism, margs.string_network_file)
    od = margs.data_path
    
    if not os.path.exists(od):
        os.mkdir(od)
    
    string_nets = ['combined']
    # Load STRING networks
    Nets = load_networks(filename)
    # Compute RWR + PPMI

    print ("### Writing output to file...")
    save_file = margs.data_path + '/' + margs.organism + '/' + margs.organism + '_net_combined.mat'
    sio.savemat(save_file, {'Net':sparse.csc_matrix(Nets[0])})
