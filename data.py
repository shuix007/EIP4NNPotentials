import os
import os.path as osp
import json
import numpy as np
import pandas as pd
from functools import reduce
from tqdm import tqdm

import dgl
from dgl.data.utils import load_graphs, save_graphs, Subset
import torch

# Collate function for ordinary graph classification

def collate_dgl(samples):
    samples = list(map(list, zip(*samples)))
    if len(samples) == 3:
        graphs, energies, has_dft = samples
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.stack(energies), torch.stack(has_dft)
    elif len(samples) == 4:
        graphs, ep_energies, ep_indicators, has_dft = samples
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.stack(ep_energies), torch.stack(ep_indicators), torch.stack(has_dft)
    else:
        raise RuntimeError('Wrong batch with len {}.'.format(len(samples)))

class PreTrainRegDataset(object):
    def __init__(self, cache_dir, potentials=['sw'], cut_per_atom_energy=200, graph_type='mg'):
        self.potentials = ['energy_'+p for p in potentials]
        self.cache_dir = cache_dir
        self.graph_type = graph_type
        self.cut_per_atom_energy = cut_per_atom_energy

        self._load_graphs()

    def get_config_index(self, configs):
        indices = list()
        for c in configs:
            if c in self.config2id:
                indices.append(self.config2id[c])
        indices = torch.LongTensor(indices)

        return indices

    def _load_graphs(self):
        print('Loading...')
        self.graphs, properties = load_graphs(
            osp.join(self.cache_dir, 'graphs')
        )
        label_df = pd.read_csv(
            osp.join(self.cache_dir, 'labels.csv'),
            dtype={'configs': str}
        )

        self.config_names = label_df['configs']
        self.config2id = {config:i for i, config in enumerate(self.config_names)}

        print('Number of graphs {}.'.format(len(self.graphs)))

        if self.graph_type == 'hmg':
            self.num_atoms = [g.nodes['atom'].data['atom_is_contributing'].sum() for g in self.graphs]
        elif self.graph_type == 'mg':
            self.num_atoms = [g.ndata['is_contributing'].sum() for g in self.graphs]
        else:
            raise ValueError('Graph type could only be hmg or mg.')

        # filter ep energy values
        for pname in self.potentials:
            for i in range(len(self.graphs)):
                per_atom_energy = label_df[pname].values[i] / self.num_atoms[i]
                if per_atom_energy > 200:
                    print('Dropping ep {} on config {} with per atom energy {:.4f} and total energy {:.4f}.'.format(
                        pname, self.config_names[i], per_atom_energy, label_df[pname].values[i]))
                    # if per atom energy is larger than 200, drop
                    label_df[pname].values[i] = 200001

        self.ep_energies = torch.FloatTensor(label_df[self.potentials].values)

    def __getitem__(self, idx):
        '''Get datapoint with index'''

        if isinstance(idx, int):
            return self.graphs[idx], self.ep_energies[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.ep_energies[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))
    
    def __len__(self):
        '''Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        '''
        return len(self.graphs)

class PreTrainClsDataset(object):
    def __init__(self, cache_dir, potentials=['sw'], cut_per_atom_energy=1, graph_type='mg'):
        self.potentials = ['energy_'+p for p in potentials]
        self.cache_dir = cache_dir
        self.graph_type = graph_type
        self.cut_per_atom_energy = cut_per_atom_energy

        self._load_graphs()

    def get_config_index(self, configs):
        indices = list()
        for c in configs:
            if c in self.config2id:
                indices.append(self.config2id[c])
        indices = torch.LongTensor(indices)

        return indices

    def get_configs(self, indices):
        configs = [self.config_names[idx] for idx in indices.tolist()]
        return configs

    def _load_graphs(self):
        print('Loading...')
        self.graphs, _ = load_graphs(
            osp.join(self.cache_dir, 'graphs')
        )
        label_df = pd.read_csv(
            osp.join(self.cache_dir, 'labels.csv'),
            dtype={'configs': str}
        )

        self.config_names = label_df['configs']
        self.config2id = {config:i for i, config in enumerate(self.config_names)}

        print('Number of graphs {}.'.format(len(self.graphs)))

        if self.graph_type == 'hmg':
            self.num_atoms = [g.nodes['atom'].data['atom_is_contributing'].sum() for g in self.graphs]
        elif self.graph_type == 'mg':
            self.num_atoms = [g.ndata['is_contributing'].sum() for g in self.graphs]
        else:
            raise ValueError('Graph type could only be hmg or mg.')

        self.dft = label_df['energy_reference'].values
        self.ep = label_df[self.potentials].values

        num_invalid_ep_configs = 0
        best_potentials = np.abs(self.ep - self.dft[:, None]).argmin(axis=1)
        for i, pid in enumerate(best_potentials):
            # test if the instance can be correctly predicted by an ep
            if np.isnan(self.dft[i]):
                best_potentials[i] = len(self.potentials)
            elif np.abs(self.ep[i, pid] - self.dft[i]) / self.num_atoms[i] > self.cut_per_atom_energy:
                best_potentials[i] = len(self.potentials)
                num_invalid_ep_configs += 1
        
        print('Found {}/{} invalid configs without good ep predictions.'.format(num_invalid_ep_configs, len(best_potentials)))
        self.ep_indices = torch.LongTensor(best_potentials)

    def get_best_ep_preds(self, config_indices, ep_logits):
        configs = self.get_configs(config_indices)
        ep_indices = ep_logits.argmax(axis=1)

        valid_configs = list()
        valid_ep_preds = list()
        valid_dfts = list()
        for c, pid in zip(configs, ep_indices):
            if pid != len(self.potentials):
                valid_configs.append(c)
                valid_ep_preds.append(self.ep[self.config2id[c], pid])
                valid_dfts.append(self.dft[self.config2id[c]])

        df = pd.DataFrame({
            'configs': valid_configs,
            'dft_preds': valid_ep_preds,
            'dft': valid_dfts 
        })
        print('Found {} valid configurations.'.format(len(valid_configs)))
        return df

    def __getitem__(self, idx):
        '''Get datapoint with index'''

        if isinstance(idx, int):
            return self.graphs[idx], self.ep_indices[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.ep_indices[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))
    
    def __len__(self):
        '''Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        '''
        return len(self.graphs)


class PreTrainDataset(object):
    def __init__(self, cache_dir, potentials=['sw'], cut_per_atom_energy=1, graph_type='mg'):
        self.potentials = ['energy_'+p for p in potentials]
        self.cache_dir = cache_dir
        self.graph_type = graph_type
        self.cut_per_atom_energy = cut_per_atom_energy

        self._load_graphs()

    def get_config_index(self, configs):
        indices = list()
        for c in configs:
            if c in self.config2id:
                indices.append(self.config2id[c])
        indices = torch.LongTensor(indices)

        return indices

    def get_configs(self, indices):
        configs = [self.config_names[idx] for idx in indices.tolist()]
        return configs

    def _load_graphs(self):
        print('Loading...')
        self.graphs, _ = load_graphs(
            osp.join(self.cache_dir, 'graphs')
        )
        label_df = pd.read_csv(
            osp.join(self.cache_dir, 'labels.csv'),
            dtype={'configs': str}
        )

        self.config_names = label_df['configs']
        self.config2id = {config:i for i, config in enumerate(self.config_names)}

        print('Number of graphs {}.'.format(len(self.graphs)))

        if self.graph_type == 'hmg':
            self.num_atoms = [g.nodes['atom'].data['atom_is_contributing'].sum() for g in self.graphs]
        elif self.graph_type == 'mg':
            self.num_atoms = [g.ndata['is_contributing'].sum() for g in self.graphs]
        else:
            raise ValueError('Graph type could only be hmg or mg.')

        # filter ep energy values
        for pname in self.potentials:
            for i in range(len(self.graphs)):
                per_atom_energy = label_df[pname].values[i] / self.num_atoms[i]
                if per_atom_energy > 200:
                    print('Dropping ep {} on config {} with per atom energy {:.4f} and total energy {:.4f}.'.format(
                        pname, self.config_names[i], per_atom_energy, label_df[pname].values[i]))
                    # if per atom energy is larger than 200, drop
                    label_df[pname].values[i] = 200001

        self.dft = label_df['energy_reference'].values
        self.ep = label_df[self.potentials].values

        num_invalid_ep_configs = 0
        best_potentials = np.abs(self.ep - self.dft[:, None]).argmin(axis=1)
        for i, pid in enumerate(best_potentials):
            # test if the instance can be correctly predicted by an ep
            if np.isnan(self.dft[i]):
                best_potentials[i] = len(self.potentials)
            elif np.abs(self.ep[i, pid] - self.dft[i]) / self.num_atoms[i] > self.cut_per_atom_energy:
                best_potentials[i] = len(self.potentials)
                num_invalid_ep_configs += 1

        print('Found {}/{} invalid configs without good ep predictions.'.format(num_invalid_ep_configs, len(best_potentials)))
        self.ep_indices = torch.LongTensor(best_potentials)
        self.ep_energies = torch.FloatTensor(label_df[self.potentials].values)
        self.has_dft = torch.ones_like(self.ep_indices, dtype=torch.bool)

    def update_pretrain_configs(self, pretrain_configs):
        indices = self.get_config_index(pretrain_configs)
        self.has_dft[indices] = False

    def get_best_ep_preds(self, config_indices, ep_logits):
        configs = self.get_configs(config_indices)
        ep_indices = ep_logits.argmax(axis=1)

        valid_configs = list()
        valid_ep = list()
        valid_ep_preds = list()
        valid_dfts = list()
        for c, pid in zip(configs, ep_indices):
            if pid != len(self.potentials):
                valid_configs.append(c)
                valid_ep_preds.append(self.ep[self.config2id[c], pid])
                valid_dfts.append(self.dft[self.config2id[c]])
                valid_ep.append(self.potentials[pid])

        df = pd.DataFrame({
            'configs': valid_configs,
            'dft_preds': valid_ep_preds,
            'dft': valid_dfts,
            'ep': valid_ep
        })
        print('Found {} valid configurations.'.format(len(valid_configs)))
        return df

    def __getitem__(self, idx):
        '''Get datapoint with index'''

        if isinstance(idx, int):
            return self.graphs[idx], self.ep_energies[idx], self.ep_indices[idx], self.has_dft[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.ep_energies[idx], self.ep_indices[idx], self.has_dft[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))
    
    def __len__(self):
        '''Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        '''
        return len(self.graphs)


class FineTuneDataset(object):
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self._load_graphs()

    def get_config_index(self, configs):
        indices = list()
        for c in configs:
            if c in self.config2id:
                indices.append(self.config2id[c])
        indices = torch.LongTensor(indices)

        return indices

    def get_configs(self, indices):
        configs = [self.config_names[idx] for idx in indices.tolist()]
        return configs

    def get_statistics(self, config_ids):
        mean = self.target[config_ids].mean()
        std = self.target[config_ids].std()

        return mean, std

    def purterb_dft(self, configs, purterbed_labels, mean, std, mode='gaussian'):
        """Purterb the dft energies with gaussian noise

        Args:
            configs (_type_): _description_
            mean (_type_): _description_
            std (_type_): _description_
        """

        purterb_indices = self.get_config_index(configs)
        self.has_dft[purterb_indices] = False

        if mode == 'ep_pred':
            print('Purterbing {} configurations with predicted best performing potentials.'.format(len(configs)))
            self.target[purterb_indices] = torch.FloatTensor(purterbed_labels)
        elif mode == 'ep':
            print('Purterbing {} configurations with best performing potentials.'.format(len(configs)))
            label_df = pd.read_csv(
                osp.join(self.cache_dir, 'labels.csv'),
                dtype={'configs':str}
            )

            potentials = [p for p in list(label_df.columns) if 'energy' in p and p != 'energy_reference']
            print('Available potentials: {}'.format(','.join(potentials)))

            best_potentials = np.abs(label_df[potentials].values - label_df['energy_reference'].values[:, None]).argmin(axis=1)
            best_potential_preds = label_df[potentials].values[np.arange(best_potentials.shape[0]), best_potentials]
            print('Best potential prediction mae: {:.4f}'.format((best_potential_preds - label_df['energy_reference'].values).mean()))

            best_potential_preds = torch.FloatTensor(best_potential_preds)
            self.target[purterb_indices] = best_potential_preds[purterb_indices]
            print('Purterbation mean: {:.4f}, std: {:.4f}'.format(best_potential_preds[purterb_indices].mean().item(), best_potential_preds[purterb_indices].std().item()))
        elif mode == 'gaussian':
            print('Purterbing {} configurations with mean {:.4f}, std {:.4f}.'.format(len(configs), mean, std))
            self.target[purterb_indices] = self.target[purterb_indices] + \
                torch.normal(mean, std, size=self.target[purterb_indices].size())
        else:
            print('Unknown purterb mode, not purterbing.')

    def _load_graphs(self):
        print('Loading...')
        self.graphs, _ = load_graphs(
            osp.join(self.cache_dir, 'graphs')
        )
        label_df = pd.read_csv(
            osp.join(self.cache_dir, 'labels.csv'),
            dtype={'configs':str}
        )

        self.config_names = label_df['configs'].values
        self.config2id = {config:i for i, config in enumerate(self.config_names)}

        print('Setting DFT as the target.')
        self.target = torch.FloatTensor(label_df['energy_reference'].values)
        self.has_dft = torch.ones_like(self.target, dtype=torch.bool)

    def __getitem__(self, idx):
        '''Get datapoint with index'''

        if isinstance(idx, int):
            return self.graphs[idx], self.target[idx], self.has_dft[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.target[idx], self.has_dft[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))

    def __len__(self):
        '''Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        '''
        return len(self.graphs)

if __name__ == '__main__':
    dataset = FineTuneDataset(
        cache_dir='CachedData/soapnet/Al'
    )
    print(collate_dgl([dataset[0], dataset[1], dataset[2]]))
