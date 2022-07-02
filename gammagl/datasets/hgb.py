import json
import ssl
import urllib.request
import sys
import os
import os.path as osp
from collections import defaultdict
from typing import Callable, List, Optional
import numpy as np
import tensorlayerx as tlx
import shutil
import errno
from gammagl.data import (HeteroGraph, InMemoryDataset, extract_zip)


def makedirs(path: str):
    r"""Recursive directory creation function."""
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise

def download_url(url: str, folder: str, filename: str, log: bool = True):
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log:
        print(f'Downloading {url}', file=sys.stderr)

    makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path

class HGBDataset(InMemoryDataset):
    r"""A variety of heterogeneous graph benchmark datasets from the
    `"Are We Really Making Much Progress? Revisiting, Benchmarking, and
    Refining Heterogeneous Graph Neural Networks"
    <http://keg.cs.tsinghua.edu.cn/jietang/publications/
    KDD21-Lv-et-al-HeterGNN.pdf>`_ paper.

    .. note::
        Test labels are randomly given to prevent data leakage issues.
        If you want to obtain final test performance, you will need to submit
        your model predictions to the
        `HGB leaderboard <https://www.biendata.xyz/hgb/>`_.
        
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (one of :obj:`"ACM"`,
            :obj:`"DBLP"`, :obj:`"Freebase"`, :obj:`"IMDB"`)
        transform (callable, optional): A function/transform that takes in an
            :class:`gammmgl.transform` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :class:`gammmgl.transform` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://cloud.tsinghua.edu.cn/d/2d965d2fc2ee41d09def/files/'
           '?p=%2F{}.zip&dl=1')

    names = {
        'acm': 'ACM',
        'dblp': 'DBLP',
        'freebase': 'Freebase',
        'imdb': 'IMDB',
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in set(self.names.keys())
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        x = ['info.dat', 'node.dat', 'link.dat', 'label.dat', 'label.dat.test', 'label.dat.test_full', 'meta.dat']
        return x

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + '_data.pt'

    def download(self):
        url = self.url.format(self.names[self.name])
        path = download_url(url, self.raw_dir, self.names[self.name]+'.zip')
        extract_zip(path, self.raw_dir)
        shutil.rmtree(osp.join(self.raw_dir, "__MACOSX"))
        for filename in self.raw_file_names:
            filePath = osp.join(self.raw_dir,self.names[self.name],filename)
            shutil.move(filePath, self.raw_dir)
        shutil.rmtree(osp.join(self.raw_dir,self.names[self.name]))
        os.unlink(path)

    def process(self):
        data = HeteroGraph()

        if self.name in ['acm', 'dblp', 'imdb']:
            with open(self.raw_paths[0], 'r') as f:  # `info.dat`
                info = json.load(f)
            n_types = info['node.dat']['node type']
            n_types = {int(k): v for k, v in n_types.items()}
            e_types = info['link.dat']['link type']
            e_types = {int(k): tuple(v.values()) for k, v in e_types.items()}
            for key, (src, dst, rel) in e_types.items():
                src, dst = n_types[int(src)], n_types[int(dst)]
                rel = rel.split('-')[1]
                rel = rel if rel != dst and rel[1:] != dst else 'to'
                e_types[key] = (src, rel, dst)
            num_classes = len(info['label.dat']['node type']['0'])
        elif self.name in ['freebase']:
            with open(self.raw_paths[0], 'r') as f:  # `info.dat`
                info = f.read().split('\n')
            start = info.index('TYPE\tMEANING') + 1
            end = info[start:].index('')
            n_types = [v.split('\t\t') for v in info[start:start + end]]
            n_types = {int(k): v.lower() for k, v in n_types}

            e_types = {}
            start = info.index('LINK\tSTART\tEND\tMEANING') + 1
            end = info[start:].index('')
            for key, row in enumerate(info[start:start + end]):
                row = row.split('\t')[1:]
                src, dst, rel = [v for v in row if v != '']
                src, dst = n_types[int(src)], n_types[int(dst)]
                rel = rel.split('-')[1]
                e_types[key] = (src, rel, dst)
        else:  # Link prediction:
            raise NotImplementedError

        # Extract node information:
        mapping_dict = {}  # Maps global node indices to local ones.
        x_dict = defaultdict(list)
        num_nodes_dict = defaultdict(lambda: 0)
        with open(self.raw_paths[1], 'r') as f:  # `node.dat`
            xs = [v.split('\t') for v in f.read().split('\n')[:-1]]
        for x in xs:
            n_id, n_type = int(x[0]), n_types[int(x[2])]
            mapping_dict[n_id] = num_nodes_dict[n_type]
            num_nodes_dict[n_type] += 1
            if len(x) >= 4:  # Extract features (in case they are given).
                x_dict[n_type].append([float(v) for v in x[3].split(',')])
        for n_type in n_types.values():
            if len(x_dict[n_type]) == 0:
                data[n_type].x = tlx.ops.eye(num_nodes_dict[n_type])
                data[n_type].num_nodes = num_nodes_dict[n_type]
            else:
                data[n_type].x = tlx.ops.convert_to_tensor(x_dict[n_type])
                data[n_type].num_nodes = num_nodes_dict[n_type]

        edge_index_dict = defaultdict(list)
        edge_weight_dict = defaultdict(list)

        with open(self.raw_paths[2], 'r') as f:  # `link.dat`
            edges = [v.split('\t') for v in f.read().split('\n')[:-1]]
        for src, dst, rel, weight in edges:
            e_type = e_types[int(rel)]
            src, dst = mapping_dict[int(src)], mapping_dict[int(dst)]
            edge_index_dict[e_type].append([src, dst])
            edge_weight_dict[e_type].append(float(weight))

        for e_type in e_types.values():
            edge_index = tlx.ops.convert_to_tensor(np.array(edge_index_dict[e_type], dtype=np.int64).T)
            data[e_type].edge_index = edge_index

            edge_weight = np.array(edge_weight_dict[e_type])
            # Only add "weighted" edgel to the graph:
            if not np.allclose(edge_weight, np.ones_like(edge_weight)):
                edge_weight = tlx.ops.convert_to_tensor(edge_weight)
                data[e_type].edge_weight = edge_weight

        # Node classification:
        if self.name in ['acm', 'dblp', 'freebase', 'imdb']:
            with open(self.raw_paths[3], 'r') as f:  # `label.dat`
                train_ys = [v.split('\t') for v in f.read().split('\n')[:-1]]
            with open(self.raw_paths[4], 'r') as f:  # `label.dat.test`
                test_ys = [v.split('\t') for v in f.read().split('\n')[:-1]]
            for y in train_ys:
                n_id, n_type = mapping_dict[int(y[0])], n_types[int(y[2])]

                if not hasattr(data[n_type], 'y'):
                    num_nodes = data[n_type].num_nodes
                    if self.name in ['imdb']:  # multi-label
                        y_list = np.zeros((num_nodes, num_classes))
                    else:
                        y_list = np.full((num_nodes, ), -1, dtype='int64')
                    train_mask = np.full((num_nodes), False, dtype='bool')
                    test_mask = np.full((num_nodes), False, dtype='bool')
                    
                if(len(y_list.shape) > 1):
                # multi-label
                    for v in y[3].split(','):
                        y_list[n_id, int(v)] = 1
                else:
                    y_list[n_id] = int(y[3])
                train_mask[n_id] = True
            for y in test_ys:
                n_id, n_type = mapping_dict[int(y[0])], n_types[int(y[2])]
                test_mask[n_id] = True

            data[n_type].y = tlx.ops.convert_to_tensor(y_list)
            data[n_type].train_mask = tlx.ops.convert_to_tensor(train_mask)
            data[n_type].test_mask = tlx.ops.convert_to_tensor(test_mask)
        else:  # Link prediction:
            raise NotImplementedError
            
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        self.save_data(self.collate([data]), self.processed_paths[0])
        

    def __repr__(self) -> str:
        return f'{self.names[self.name]}()'


    def save_results(self, logits, test_idx, file_path):
        test_logits = logits[test_idx]
        if self.name == 'imdb':
            pred = (tlx.convert_to_numpy(test_logits) > 0).astype(int)
            multi_label = []
            for i in range(pred.shape[0]):
                label_list = [str(j) for j in range(pred[i].shape[0]) if pred[i][j] == 1]
                multi_label.append(','.join(label_list))
            pred = multi_label
        elif self.name in ['acm', 'dblp', 'freebase']:
            pred = tlx.convert_to_numpy(test_logits).argmax(axis=-1)
            pred = np.array(pred)
        else:
            return
        
        with open(file_path, "w") as f:
            for nid, l in zip(test_idx, pred):
                f.write(f"{nid}\t\t{0}\t{l}\n")
