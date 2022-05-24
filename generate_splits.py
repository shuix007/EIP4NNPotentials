import argparse
import os
import os.path as osp
import json
from posixpath import split
import numpy as np
import pandas as pd

def _generate_split(split, split_dir, val_portion, split_name, fold_no):
    pretrain_configs = split['pretrain']
    train_configs = split['training']
    test_configs = split['test']

    np.random.shuffle(train_configs)

    n = len(train_configs)
    n_val = int(n * val_portion)

    val_configs = train_configs[:n_val]
    train_configs = train_configs[n_val:]

    df = pd.DataFrame({
        'configs': np.concatenate([pretrain_configs, train_configs, val_configs, test_configs]),
        'mark': ['pretrain'] * len(pretrain_configs) + ['train'] * len(train_configs) + ['val'] * len(val_configs) + ['test'] * len(test_configs)
    })

    output_filename = osp.join(
        split_dir, '{}_{}.csv'.format(split_name, fold_no))
    df.to_csv(output_filename, index=False, header=True)


def load_si_config_splits(data_dir, split_dir, cache_dir, val_portion=0.2):
    label_df = pd.read_csv(
        osp.join(cache_dir, 'labels.csv'))
    all_configs = label_df['configs'].values

    random_split_filename = osp.join(
        data_dir, 'cross_validation/random_five_fold/folds/all_folds.json')

    random_split = json.load(open(random_split_filename))

    # The old random split of 2k configs
    for i in range(1, 3):
        split = random_split['fold_{}'.format(i)]
        split['pretrain'] = np.setdiff1d(all_configs, split['training'] + split['test'])

        _generate_split(
            split,
            split_dir=split_dir,
            val_portion=val_portion,
            split_name='random',
            fold_no=i
        )

def load_al_config_splits(split_dir, cache_dir, num_splits, split_portion):
    label_df = pd.read_csv(
        osp.join(cache_dir, 'labels.csv'))
    all_configs = label_df['configs'].values

    num_pretrains, num_trains, num_vals = (len(all_configs) * split_portion[:3]).astype(int)
    num_tests = len(all_configs) - num_pretrains - num_trains - num_vals

    np.random.shuffle(all_configs)
    pretrain_configs = all_configs[:num_pretrains]
    finetune_configs = all_configs[num_pretrains:]

    for fold_no in range(1, num_splits+1):
        np.random.shuffle(finetune_configs)
        df = pd.DataFrame({
            'configs': np.concatenate([pretrain_configs, finetune_configs]),
            'mark': ['pretrain'] * num_pretrains + \
                    ['train'] * num_trains + \
                    ['val'] * num_vals + \
                    ['test'] * num_tests
        })

        output_filename = osp.join(
            split_dir, 'random_{}.csv'.format(fold_no))
        df.to_csv(output_filename, index=False, header=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess KimRec')
    parser.add_argument('--data_dir', type=str, default='Data/KIM-Si',
                        help='directory to the dataset.')
    parser.add_argument('--split_dir', type=str, default='Splits/',
                        help='directory to save the splits.')
    parser.add_argument('--cache_dir', type=str, default='CachedData/',
                        help='directory to save the splits.')
    parser.add_argument('--species', type=str, default='Si',
                        help='atom species.')
    parser.add_argument('--split', type=str, default='0.8,0.12,0.04,0.04',
                        help='validation portion (default: .2)')
    args = parser.parse_args()

    np.random.seed(42)

    split_portion = np.array(list(map(float, args.split.split(','))))
    
    split_dir = osp.join(
        args.split_dir, args.species
    )
    cache_dir = osp.join(
        args.cache_dir, args.species
    )
    
    if args.species == 'Si':
        load_si_config_splits(
            data_dir=args.data_dir,
            split_dir=split_dir,
            cache_dir=cache_dir,
            val_portion=split_portion[2]
        )
    elif args.species == 'Al':
        load_al_config_splits(
            split_dir=split_dir,
            cache_dir=cache_dir,
            num_splits=3,
            split_portion=split_portion
        )
