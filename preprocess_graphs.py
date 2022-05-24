import argparse
import os
import os.path as osp
import json
import numpy as np
import pandas as pd
from functools import reduce
from dscribe.descriptors import SOAP
from tqdm import tqdm

import dgl
from dgl.data.utils import load_graphs, save_graphs
import torch
from ase.io import read

from mol import (
    neighbor_list_to_molecular_graph, 
    remove_duplicate_edges, 
    AL_POTENTIALS,
    SI_POTENTIALS
)
from utils import get_pbc_graphs

def _compute_soap_descriptors(data_dir, configs, species):
    if species == 'Si':
        species = ["Si"]
        rcut = 5.0
    elif species == 'Al':
        species = ["Al"]
        rcut = 7.0
    nmax = 8
    lmax = 6

    # Setting up the SOAP descriptor
    soap = SOAP(
        species=species,
        periodic=True,
        rcut=rcut,
        nmax=nmax,
        lmax=lmax,
    )

    ase_configs = []
    for c in configs:
        config_filename = os.path.join(data_dir, c, c+'.xyz')
        atoms = read(config_filename)
        ase_configs.append(atoms)
    
    print('Computing SOAP descriptors for {} {} configurations with cut {:.1f}.'.format(len(ase_configs), species, rcut))
    soap_all = soap.create(ase_configs)
    soap_all_labeled = [[configs[i], s] for i, s in enumerate(soap_all)]
    print('Done')
    return soap_all_labeled

def _prepare_soap_graphs(data_dir, prpty_names, species):
    graphs = list()
    properties = {pname: [] for pname in ['configs']+prpty_names}

    configs = os.listdir(data_dir)
    soaps = _compute_soap_descriptors(data_dir, configs, species)

    n = len(soaps)
    for i in range(n):
        config = soaps[i][0]
        soap = soaps[i][1]

        num_atoms = soap.shape[0]
        g = dgl.graph((torch.LongTensor([]), torch.LongTensor([])), num_nodes=num_atoms)
        g.ndata['is_contributing'] = torch.ones(num_atoms, dtype=torch.float32).unsqueeze(1)
        g.ndata['node_feats'] = torch.FloatTensor(soap)

        filename = osp.join(
            data_dir,
            "{}/{}.xyz".format(config, config)
        )
        prpty_dict = _load_properties(filename)

        graphs.append(g)

        if 'energy_reference' not in prpty_dict:
            prpty_dict['energy_reference'] = np.nan # if the configuration does not have dft

        for key, value in prpty_dict.items():
            if key in properties:
                properties[key].append(value)
        
        properties['configs'].append(config)

    return (
        graphs,
        properties
    )

def _load_config(filename, cut_r):
    atoms = read(filename, format="extxyz")

    pos = atoms.get_positions()
    species = atoms.get_chemical_symbols()
    pbc = atoms.get_pbc()

    r_max = cut_r

    # Cell vectors as a 3x3 ndarray
    cell = atoms.get_cell()

    edge_index, pos_all, species_all, atomic_numbers_all, image_of_all, is_contributing_all = get_pbc_graphs(
        pos=pos,
        species=species,
        r_cut=r_max,
        cell=cell,
        pbc=pbc,
    )

    return edge_index, pos_all, atomic_numbers_all, image_of_all, is_contributing_all

def _load_properties(filename):
    prpty_dict = dict()
    with open(filename, 'r') as f:
        lines = [_.strip() for _ in f.readlines()]
        property_string = lines[1]
        property_string = property_string[property_string.find('energy'):]

        properties = property_string.split(' ')

        for prp in properties:
            prp = prp.split('=')
            prpty_dict[prp[0]] = float(prp[1])

        return prpty_dict

def _prepare_molecular_graphs(data_dir, prpty_names, cut_r):
    graphs = list()
    properties = {pname: [] for pname in ['configs']+prpty_names}

    config_names = os.listdir(data_dir)
    for name in config_names:
        filename = osp.join(
            data_dir,
            "{}/{}.xyz".format(name, name)
        )
        try:
            edge_index, pos_all, atomic_numbers_all, image_of_all, is_contributing_all = _load_config(
                filename, cut_r)
            prpty_dict = _load_properties(filename)
        except:
            continue

        # convert edge indices to heterogeneous molecular graphs
        edge_index = remove_duplicate_edges(edge_index)

        g = neighbor_list_to_molecular_graph(
            (torch.from_numpy(edge_index[0]),
                torch.from_numpy(edge_index[1])),
            torch.from_numpy(pos_all),
            torch.from_numpy(atomic_numbers_all),
            torch.from_numpy(image_of_all),
            torch.from_numpy(is_contributing_all).float()
        )

        graphs.append(g)

        if 'energy_reference' not in prpty_dict:
            prpty_dict['energy_reference'] = np.nan # if the configuration does not have dft

        for key, value in prpty_dict.items():
            if key in properties:
                properties[key].append(value)
        
        properties['configs'].append(name)

    return (
        graphs,
        properties
    )


def preprocess(data_dir, cut_r, prpty_names, cache_dir, graph_type, species):
    assert graph_type in [
        'mg', 'soap'], 'Graph must be in hmg or mg. Provided {}'.format(graph_type)
    
    if graph_type == 'mg':
        (
            graphs,
            properties
        ) = _prepare_molecular_graphs(data_dir, prpty_names, cut_r)
    else:
        (
            graphs,
            properties
        ) = _prepare_soap_graphs(data_dir, prpty_names, species)

    print('Number of valid configurations: {}.'.format(len(graphs)))

    print('Saving...')
    save_graphs(
        osp.join(cache_dir, 'graphs'),
        graphs
    )

    df = pd.DataFrame(properties)
    df.to_csv(
        osp.join(cache_dir, 'labels.csv'),
        index=False,
        header=True
    )


def test_load(cache_dir):
    print('Loading...')
    graphs, _ = load_graphs(
        osp.join(cache_dir, 'graphs')
    )
    labels = pd.read_csv(
        osp.join(cache_dir, 'labels.csv')
    )

    print('Number of graphs {}.'.format(len(graphs)))
    labels.head()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess KimRec')
    parser.add_argument('--data_dir', type=str, default='Data',
                        help='directory to the dataset.')
    parser.add_argument('--cache_dir', type=str, default='CachedData',
                        help='directory to the dataset.')
    parser.add_argument('--species', type=str, default='Si',
                        help='atom species.')
    parser.add_argument('--graph_type', type=str, default='soap',
                        help='directory to the dataset.')
    parser.add_argument('--cut_r', type=float, default=5.,
                        help='Cutoff distance (default: 5.)')
    args = parser.parse_args()

    assert args.species in ['Si', 'Al'], "Only support Si and Al for now."
    
    if args.species == 'Si':
        data_dir = os.path.join(args.data_dir, 'KIM-Si')
        potentials = ['reference'] + SI_POTENTIALS
    elif args.species == 'Al':
        data_dir = os.path.join(args.data_dir, 'ANI-Al')
        potentials = ['reference'] + AL_POTENTIALS
    prpty_names = ['energy_' + p for p in potentials]

    gnn = 'soapnet' if args.graph_type == 'soap' else 'schnet'
    cache_dir = os.path.join(args.cache_dir, gnn, args.species)
    if not osp.exists(cache_dir):
        os.makedirs(cache_dir)

    preprocess(
        data_dir=data_dir,
        prpty_names=prpty_names,
        cut_r=args.cut_r,
        cache_dir=cache_dir,
        graph_type=args.graph_type,
        species=args.species
    )

    test_load(cache_dir)