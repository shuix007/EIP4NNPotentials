import os
import h5py
import argparse
import numpy as np
import pandas as pd

import dgl
import torch

from tqdm.notebook import tqdm
from ase.calculators.kim import KIM
from ase.io import read, write
from ase import Atoms

potential_abbrev = {
    'EAM_Dynamo_MendelevKramerBecker_2008_Al__MO_106969701023_005': 'MKB',
    'EAM_Dynamo_WineyKubotaGupta_2010_Al__MO_149316865608_005': 'WKG',
    'EAM_Dynamo_Zhakhovsky_2009_Al__MO_519613893196_000': 'Zha',
    'EAM_Dynamo_SturgeonLaird_2000_Al__MO_120808805541_005': 'SL',
    'EAM_Dynamo_ZhouJohnsonWadley_2004NISTretabulation_Al__MO_060567868558_000': 'ZJW_NIST',
    'EAM_Dynamo_ZopeMishin_2003_Al__MO_664470114311_005': "ZM",
    'EAM_ErcolessiAdams_1994_Al__MO_324507536345_003': 'EA',
    'EMT_Asap_Standard_JacobsenStoltzeNorskov_1996_Al__MO_623376124862_001': 'JSN',
    'Morse_Shifted_GirifalcoWeizer_1959HighCutoff_Al__MO_140175748626_004': 'GW_High',
    'Morse_Shifted_GirifalcoWeizer_1959LowCutoff_Al__MO_411898953661_004': 'GW_Low',
    'Morse_Shifted_GirifalcoWeizer_1959MedCutoff_Al__MO_279544746097_004': 'GW_Med'
}

potentials = list(potential_abbrev.keys())

def convert_h5py_to_extxyz(input_dir, output_dir, filenames):
    num_configs = 0
    for filename in filenames:
        filename = os.path.join(input_dir, filename)
        with h5py.File(filename, "r") as f:
            keys = list(f.keys())

            for k in keys:
                energies = list(f[k]['energy'])
                cells  = list(f[k]['cell'])
                fermis = list(f[k]['fermi'])
                species = list(f[k]['species'])
                forces = list(f[k]['force'])
                coordinates = list(f[k]['coordinates'])

                for i in range(len(energies)):
                    data_obj = Atoms(
                        ['Al']*coordinates[i].shape[0],
                        positions=coordinates[i],
                        cell=cells[i],
                        pbc=(True, True, True)
                    )
                    data_obj.arrays['forces'] = forces[i]
                    data_obj.info['energy_reference'] = energies[i]
                    data_obj.info['fermi'] = fermis[i]

                    output_filename = os.path.join(output_dir, str(num_configs), '{}.xyz'.format(num_configs))
                    if not os.path.isdir(os.path.join(output_dir, str(num_configs))):
                        os.mkdir(os.path.join(output_dir, str(num_configs)))
                    write(output_filename, data_obj)
                    num_configs += 1

def compute_EIP_energies(configs, output_dir):
    config_atoms = list()
    for config in configs:
        config_filename = os.path.join(output_dir, config, config+'.xyz')
        atoms = read(config_filename)
        config_atoms.append(atoms)

    potential_dict = dict()
    for pname in potentials:
        try:
            potential = KIM(pname)
            potential_energies = list()
            for atoms in config_atoms:
                atoms.calc = potential
                energy = atoms.get_potential_energy()
                potential_energies.append(energy)
            potential_dict[pname] = potential_energies
        except:
            pass
    print('List of EIPs used: {}'.format(','.join(list(potential_dict.keys()))))

    dft_energies = list()
    for atoms in config_atoms:
        energy = atoms.info['energy_reference']
        dft_energies.append(energy)
    potential_dict['dft'] = dft_energies

    potential_dict['configs'] = configs
    df = pd.DataFrame(data=potential_dict)
    return df

def write_EIP_energies(df, output_dir):
    for index, line in df.iterrows():
        config = line['configs']
        config_filename = os.path.join(output_dir, config, config+'.xyz')

        ep_string = ''
        for p in potentials:
            ep_string += 'energy_{}={:.6f} '.format(potential_abbrev[p], line[p])
        
        with open(config_filename, 'r') as f:
            config_file_contents = f.read().splitlines()

            # reorder property line
            property_line = config_file_contents[1]

            eidx = property_line.find('energy')
            fidx = property_line.find('fermi')
            pidx = property_line.find('pbc')

            property_line_keep = property_line[:eidx]
            energy_str = property_line[eidx:fidx-1]
            fermi_str = property_line[fidx:pidx-1]
            pbc_str = property_line[pidx:]

            config_file_contents[1] = \
                property_line_keep + \
                pbc_str + ' ' + \
                fermi_str + ' ' + \
                energy_str + ' ' + \
                ep_string
        
        with open(config_filename, 'w') as f:
            f.write('\n'.join(config_file_contents))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess KimRec')
    parser.add_argument('--input_dir', type=str, default='RawData/ANI-Al',
                        help='directory to the dataset.')
    parser.add_argument('--output_dir', type=str, default='Data/ANI-Al',
                        help='directory to the dataset.')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filenames = [f for f in os.listdir(input_dir) if f.split('.')[-1] == 'h5']

    convert_h5py_to_extxyz(input_dir, output_dir, filenames)
    configs = os.listdir(output_dir)
    eip_energies_df = compute_EIP_energies(configs, output_dir)
    write_EIP_energies(eip_energies_df, output_dir)