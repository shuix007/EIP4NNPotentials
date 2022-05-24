import argparse
import dgl
import numpy as np
import pandas as pd
import os
import os.path as osp
import math
import random
import time

import torch
import torch.optim as optim

from data import FineTuneDataset
from mol import LARGEST_ATOMIC_NUMBER
from Models import SchNet, SOAPNet, shifted_softplus
from trainer import MGNNFineTuner, SOAPFineTuner
from utils import save_args

def load_splits(split_dir, split_name, purterb=0.):
    filename = os.path.join(split_dir, split_name+'.csv')
    split = pd.read_csv(filename, dtype={'configs': str, 'mark': str})

    pretrain_configs = split[split['mark'] == 'pretrain']['configs'].tolist()
    train_configs = split[split['mark'] == 'train']['configs'].tolist()
    val_configs = split[split['mark'] == 'val']['configs'].tolist()
    test_configs = split[split['mark'] == 'test']['configs'].tolist()

    if purterb > 0:
        num_additional_configs = int(len(pretrain_configs) * purterb)
        purterb_configs = pretrain_configs[-num_additional_configs:]
    else:
        purterb_configs = []

    return {
        'purterb': purterb_configs,
        'train': train_configs,
        'val': val_configs,
        'test': test_configs
    }

def load_purterbed_labels(purterbed_label_dir):
    filename = os.path.join(purterbed_label_dir, 'cls_labels.csv')
    label_df = pd.read_csv(filename, dtype={'configs': str})

    configs = label_df['configs'].tolist()
    purterbed_labels = label_df['dft_preds'].tolist()

    return configs, purterbed_labels

def main():
    # Data configuration
    parser = argparse.ArgumentParser(description='GNN4KimRec')
    parser.add_argument('--cache_dir', type=str, required=True,
                        help='directory to the processed dataset.')
    parser.add_argument('--workspace', type=str, required=True,
                        help='directory to save checkpoint')
    parser.add_argument('--pretrained_model_dir', type=str, default='',
                        help='dir to pre-trained weights.')
    parser.add_argument('--split_dir', type=str, default='Splits',
                        help='dir to splits.')
    parser.add_argument('--split_name', type=str, default='random_1',
                        help='name of the split.')
    parser.add_argument('--purterbed_label_dir', type=str, default='',
                        help='name of the split.')

    # Experiment configuration
    parser.add_argument('--gnn', type=str, default='schnet',
                        help='GNN to use.')
    parser.add_argument('--inference_only', action='store_true')

    # Training configuration
    parser.add_argument("--max_grad_norm", type=float, default=5.,
                        help="DFT threshold for stabelize training.")
    parser.add_argument('--target', type=str, default='dft',
                        help='Target to finetune the potential.')
    parser.add_argument('--scheduler', type=str, default='linear',
                        help='Scheduler to use.')
    parser.add_argument("--max_dft", type=float, default=200000,
                        help="DFT threshold for stabelize training.")
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed to use (default: 42)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help='input batch size for evaluation (default: 256)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='dropout ratio (default: 0)')
    parser.add_argument("--lr_decrease_rate", type=float, default=0.1,
                        help="learning rate decreasing rate.")
    parser.add_argument("--weight_decay", type=float, default=0.,
                        help="l2 regularization.")
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--decay_epochs', type=int, default=100,
                        help='number of epochs to evaluate (default: 70)')
    parser.add_argument('--tol', type=int, default=200,
                        help='number of tolerence epochs (default: 100)')

    # model configuration
    parser.add_argument('--num_convs', type=int, default=5,
                        help='number of graph convolutions (default: 0)')
    parser.add_argument('--num_output_heads', type=int, default=1,
                        help='number of graph convolutions (default: 0)')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--alpha', type=float, default=0.,
                        help='ratio to adjust the contribution of noise label (default: 0)')
    parser.add_argument('--cut_r', type=float, default=3.,
                        help='Cutoff distance (default: 3.)')
    parser.add_argument('--hidden_dims', type=int, default=128,
                        help='dimensionality of hidden units in GNNs (default: 128)')

    # purterb
    parser.add_argument('--purterb_ratio', type=float, default=0.,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--purterb_mean', type=float, default=-1.,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--purterb_std', type=float, default=-1.,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--purterb_mode', type=str, default='ep',
                        help='dropout ratio (default: 0)')

    args = parser.parse_args()

    print(args)

    # fix all random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True

    # save the arguments
    if not os.path.exists(args.workspace):
        os.mkdir(args.workspace)
    save_args(args, args.workspace)

    Dataset = FineTuneDataset(
        cache_dir=args.cache_dir
    )
    
    split_dict = load_splits(args.split_dir, args.split_name, purterb=args.purterb_ratio)
    if args.purterbed_label_dir != '':
        purterbed_configs, purterbed_labels = load_purterbed_labels(args.purterbed_label_dir)
        split_dict['purterb'] = purterbed_configs

    if len(split_dict['purterb']) > 0:
        Dataset.purterb_dft(
            configs=split_dict['purterb'],
            purterbed_labels=purterbed_labels,
            mean=args.purterb_mean,
            std=args.purterb_std,
            mode=args.purterb_mode
        ) # purterb additional configurations for training

    args.pgratio = len(split_dict['train']) / len(split_dict['purterb']) if len(split_dict['purterb']) > 0 else 0.
    split_dict['train'] = split_dict['train'] + split_dict['purterb']
    split_id_dict = {key: Dataset.get_config_index(
        split_dict[key]) for key in ['train', 'val', 'test']}

    train_dataset = Dataset[split_id_dict['train']]
    val_dataset = Dataset[split_id_dict['val']]
    test_dataset = Dataset[split_id_dict['test']]

    print('Number of train: {}, val: {}, test: {}.'.format(
        len(train_dataset), len(val_dataset), len(test_dataset)))

    assert args.gnn in ['schnet', 'soapnet'], 'GNN must be schnet.'
    if args.gnn == 'schnet':
        model = SchNet(
            n_convs=args.num_convs,
            n_atoms=LARGEST_ATOMIC_NUMBER,
            n_output_heads=args.num_output_heads,
            hidden_dims=args.hidden_dims,
            K=args.hidden_dims,
            activation=shifted_softplus,
            cut_r=args.cut_r,
            dropout=args.dropout,
            problem='reg'
        ).to(args.device)

        if args.pretrained_model_dir != '':
            print('Loading pretrained model weights from {}'.format(args.pretrained_model_dir))
            model.load_pretrained(args.pretrained_model_dir)

        trainer = MGNNFineTuner(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            args=args
        )
    elif args.gnn == 'soapnet':
        model = SOAPNet(
            num_layers=args.num_convs,
            input_dims=252,
            hidden_dims=args.hidden_dims,
            n_output_heads=args.num_output_heads,
            activation=shifted_softplus,
            problem='reg'
        ).to(args.device)

        if args.pretrained_model_dir != '':
            print('Loading pretrained model weights from {}'.format(args.pretrained_model_dir))
            model.load_pretrained(args.pretrained_model_dir)

        trainer = SOAPFineTuner(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            args=args
        )

    trainer.train()
    trainer.load_model()
    preds = trainer.test()

    data = {
        'configs': Dataset.get_configs(split_id_dict['test']),
        'dft_preds': preds
    }
    pd.DataFrame(
        data=data
    ).to_csv(osp.join(args.workspace, 'fine-tune-preds.csv'), index=False, header=True)


if __name__ == '__main__':
    main()
