import os
import os.path as osp
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip, Dataset
from torch_geometric.data import DataLoader, DenseDataLoader
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import pandas as pd
import torch
import torch_geometric
import numpy as np
from scipy.sparse import csr_matrix
import scipy.io
import mat73
from sklearn.model_selection import train_test_split

import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datapath",
        type=str,
        default="/cluster/home/alakshmanan/ra_work/24",
        help="which path to retreive the zip file from",
    )
    parser.add_argument(
        "--datatype",
        type=str,
        default="binary",
        choices=[
            "binary",
            "multiclass",
            "regression",
        ],
        help="which power grid to use in small letters",
    )
    parser.add_argument(
        "--gridtype",
        type=str,
        default="ieee24",
        choices=[
            "ieee24",
            "ieee39",
            "ieee118",
            "swissgrid",
            "ukgrid",
        ],
        help="which power grid to use in small letters",
    )
    parser.add_argument(
        "--train_test_split",
        type=lambda s: [float(item) for item in s.split(",")],
        default="80,10,10",
        help="Train Test Val split input with comma",
    )

    args = parser.parse_args()
    return args

class Powergrid(Dataset):
    datatype = ''
    split = ''
    #url = r'T:\03_Student_Projects\Rahel Wolfisberg\Rahel\IEEE39-Data.zip'
    #url = r'C:\Users\Rahel Wolfisberg\Documents\ETH\Master\Semesterarbeit\IEEE39-Data.zip'


    def __init__(self, root, gridtype = 'ieee24', datatype = 'Binary',  trainsize = 1, testsize = 0, valsize = 0, delete = None, transform=None, pre_transform=None, pre_filter=None, device = 'cpu'):
        self.gridtype = gridtype
        self.datatype = datatype
        self.trainsize = trainsize
        self.testsize = testsize
        self.valsize = valsize
        self.delete = delete
        self.device = device
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['of_bi*', 'of_reg*', 'of_mc*', 'Bf*', 'Ef*', 'edge_order*']


    @property
    def processed_file_names(self):
        self.len()
        length = self.length
        for idx in range (length):
            return f'dataset_{self.gridtype}_{idx}.pt'


    def download(self):
        #path = download_url(self.url, self.raw_dir)
        extract_zip(r'/cluster/home/alakshmanan/ra_work/raw_24.zip', self.raw_dir)

    def len(self):
        path = os.path.join(self.raw_dir, 'of_bi.mat')
        of_bi = mat73.loadmat(path)
        of_bi = of_bi['of_bi']
        length = len(of_bi)
        self.length = length
        return length

    def process(self):
        #function that deletes row
        def th_delete(tensor, indices):
            mask = torch.ones(tensor.size(), dtype=torch.bool)
            mask[indices] = False
            return tensor[mask]
        # load branch list also called edge order or edge index
        path = os.path.join(self.raw_dir, 'edge_order.csv')
        edge_order = pd.read_csv(path, index_col=False, header=None)
        edge_order = torch.tensor(edge_order.values)
        # load output binary classification labels
        path = os.path.join(self.raw_dir, 'of_bi.mat')
        of_bi = mat73.loadmat(path)
        # load output binary regression labels
        path = os.path.join(self.raw_dir, 'of_reg.mat')
        of_reg = mat73.loadmat(path)
        # load output multiclass classification labels
        path = os.path.join(self.raw_dir, 'of_mc.mat')
        of_mc = mat73.loadmat(path)
        # load output node feature matrix
        path = os.path.join(self.raw_dir, 'Bf.mat')
        node_f = mat73.loadmat(path)
        # load output edge feature matrix
        path = os.path.join(self.raw_dir, 'Ef.mat')
        edge_f = mat73.loadmat(path)

        type_model = ('bi', 'reg', 'mc')
        of_bi, of_reg, of_mc = of_bi['of_{}'.format(type_model[0])], of_reg['of_{}'.format(type_model[1])], of_mc['of_{}'.format(type_model[2])]
        node_f = node_f['Bf']
        edge_f = edge_f['Ef']

        # store data
        reg = [of_reg[i].item() for i in range(len(of_bi))]
        # for i in range(len(of_bi)):
        #     a = of_reg[i].item()
        #     reg.append(a)

        # flip edge order (Message passing purposes)
        edge_order_flip = torch.fliplr(edge_order)
        edge_i = torch.cat((edge_order, edge_order_flip), 0)
        edge_index = edge_i.t().contiguous()

        parent_dir = self.processed_dir
        directoryall = 'all'
        directorytrain = 'train'
        directorytest = 'test'
        directoryval = 'validation'
        pathall = os.path.join(parent_dir, directoryall)
        pathtrain = os.path.join(parent_dir, directorytrain)
        pathtest = os.path.join(parent_dir, directorytest)
        pathval = os.path.join(parent_dir, directoryval)
        os.mkdir(pathall)
        os.mkdir(pathtrain)
        os.mkdir(pathtest)
        os.mkdir(pathval)
        self.pathall = pathall

        data_list = []
        idx = 0
        # MAIN data processing loop
        for i in range(len(node_f)):
            # node feat
            x = torch.tensor(node_f[i][0], dtype=torch.float32).reshape([-1, 3]).to(self.device)
            # edge feat
            f = torch.tensor(edge_f[i][0], dtype=torch.float32)
            # contigency lists, finds where do we have contigencies from the .mat edge feature matrices
            # ( if a line is part of the contigency list all egde features are set 0)
            cont = [j for j in range(len(f)) if np.all(np.array(f[j])) == 0]
            # remove edge features of the associated line
            f_tot = th_delete(f, cont).reshape([-1, 4]).type(torch.float32)
            # concat the post-contigency edge feature matrix to take into account the reversed edges
            f_totw = torch.cat((f_tot, f_tot), 0).to(self.device)
            # remove failed lines from branch list
            edge_iw = th_delete(edge_order, cont).reshape(-1, 2).type(torch.int64)
            # flip branch list
            edge_iwr = torch.fliplr(edge_iw)
            #  and concat the non flipped and flipped branch list
            edge_iw = torch.cat((edge_iw, edge_iwr), 0)
            edge_iw = edge_iw.t().contiguous().to(self.device)

            # getting binary/ regression/ multiclass values
            data_type = self.datatype
            if data_type == 'Binary' or data_type == 'binary':
                ydata = int(of_bi[i][0])
            if data_type == 'Regression' or data_type == 'regression':
                ydata = of_reg[i]
            if data_type == 'Multiclass' or data_type == 'multiclass':
                ydata = of_mc[i][0]

            # Fill Data object, 1 Data object -> 1 graph
            data = Data(x=x, edge_index=edge_iw, edge_attr=f_totw, y=torch.tensor(ydata, dtype=torch.float32, device=self.device))
            # append Data object to datalist
            data_list.append(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(pathall, f'dataset_{self.gridtype}_{idx}.pt'))
            idx+=1

        graphs = [i for i in range(len(of_bi))]
        # for i in range(len(of_bi)):
        #     graphs.append(i)

        idxtrain = self.trainsize
        graphs_train, graphs_rem = train_test_split(graphs, train_size=idxtrain, shuffle=True)
        idxtest = self.testsize
        idxval = self.valsize
        rem = 100 / (idxtest + idxval) * idxtest
        test = rem / 100
        graphs_val, graphs_test = train_test_split(graphs_rem, test_size=test, shuffle=True)
        self.graphs_train = graphs_train
        self.graphs_test = graphs_test
        self.graphs_val = graphs_val

        delete = self.delete
        data_train = []
        data_test = []
        data_val = []
        for i in range (len(graphs_train)):
            n = graphs_train[i]
            data = torch.load(osp.join(pathall, f'dataset_{self.gridtype}_{n}.pt'))
            torch.save(data, osp.join(pathtrain, f'dataset_{self.gridtype}_train_{i}.pt'))
            data_train.append(data)
            if delete == 'Yes' or delete == 'yes':
                os.remove(osp.join(pathall, f'dataset_{self.gridtype}_{n}.pt'))
        for i in range (len(graphs_test)):
            n = graphs_test[i]
            data = torch.load(osp.join(pathall, f'dataset_{self.gridtype}_{n}.pt'))
            torch.save(data, osp.join(pathtest, f'dataset_{self.gridtype}_test_{i}.pt'))
            data_test.append(data)
            if delete == 'Yes' or delete == 'yes':
                os.remove(osp.join(pathall, f'dataset_{self.gridtype}_{n}.pt'))
        for i in range (len(graphs_val)):
            n = graphs_val[i]
            data = torch.load(osp.join(pathall, f'dataset_{self.gridtype}_{n}.pt'))
            torch.save(data, osp.join(pathval, f'dataset_{self.gridtype}_val_{i}.pt'))
            data_val.append(data)
            if delete == 'Yes' or delete == 'yes':
                os.remove(osp.join(pathall, f'dataset_{self.gridtype}_{n}.pt'))
        if delete == 'Yes' or delete == 'yes':
            os.rmdir(osp.join(self.processed_dir, 'all'))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    # dataset = IEEE24(root='datasetIEEE24mc',datatype='multiclass', trainsize = 0.8, testsize = 0.1, valsize = 0.1, delete = 'yes', device = device)
    dataset = Powergrid(root=args.grid_type, grid_type = args.grid_type, datatype=args.datatype,
                    trainsize = args.train_test_split[0], testsize = args.train_test_split[1], 
                    valsize = args.train_test_split[2], delete = 'yes', device = device)