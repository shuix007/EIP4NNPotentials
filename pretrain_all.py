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

from data import PreTrainDataset
from mol import SI_POTENTIALS, AL_POTENTIALS, LARGEST_ATOMIC_NUMBER
from Models import SchNetAll, SOAPNetAll, shifted_softplus
from trainer import MGNNPreTrainerAll, SOAPPreTrainerAll
from utils import save_args

def load_splits(split_dir, split_name):
    filename = os.path.join(split_dir, split_name+'.csv')
    split = pd.read_csv(filename, dtype={'configs': str, 'mark': str})

    pretrain_configs = split[split['mark'] == 'pretrain']['configs'].tolist()
    train_configs = split[split['mark'] == 'train']['configs'].tolist()
    val_configs = split[split['mark'] == 'val']['configs'].tolist()
    test_configs = split[split['mark'] == 'test']['configs'].tolist()

    return {
        'pretrain': pretrain_configs,
        'train': train_configs,
        'val': val_configs,
        'test': test_configs
    }

def main():
    # Data configuration
    parser = argparse.ArgumentParser(description='GNN4KimRec')
    parser.add_argument('--species', type=str, default='Si',
                        help='Species.')
    parser.add_argument('--cache_dir', type=str, required=True,
                        help='directory to the processed dataset.')
    parser.add_argument('--workspace', type=str, required=True,
                        help='directory to save checkpoint')
    parser.add_argument('--split_dir', type=str, default='Splits',
                        help='dir to splits.')
    parser.add_argument('--split_name', type=str, default='random_1',
                        help='name of the split.')
    parser.add_argument('--pretrained_model_dir', type=str, default='',
                        help='dir to pre-trained weights.')

    # Experiment configuration
    parser.add_argument('--gnn', type=str, default='schnet',
                        help='GNN to use.')
    parser.add_argument('--inference_only', action='store_true')

    # Training configuration
    parser.add_argument("--max_grad_norm", type=float, default=5.,
                        help="DFT threshold for stabelize training.")
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed to use (default: 42)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--eval_batch_size', type=int, default=128,
                        help='input batch size for evaluation (default: 256)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='dropout ratio (default: 0)')
    parser.add_argument("--lr_decrease_rate", type=float, default=0.1,
                        help="learning rate decreasing rate.")
    parser.add_argument("--weight_decay", type=float, default=0.,
                        help="l2 regularization.")
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--decay_epochs', type=int, default=70,
                        help='number of epochs to evaluate (default: 70)')
    parser.add_argument('--tol', type=int, default=200,
                        help='number of tolerence epochs (default: 100)')

    # model configuration
    parser.add_argument('--num_convs', type=int, default=5,
                        help='number of graph convolutions (default: 0)')
    parser.add_argument('--num_eps', type=int, default=8,
                        help='number of graph convolutions (default: 0)')
    parser.add_argument('--ep', type=str, default='',
                        help='number of graph convolutions (default: 0)')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--cut_r', type=float, default=3.,
                        help='cut off distance (default: 3.)')
    parser.add_argument('--cut_per_atom_energy', type=float, default=1.,
                        help='cut off distance (default: 3.)')
    parser.add_argument('--hidden_dims', type=int, default=128,
                        help='dimensionality of hidden units in GNNs (default: 128)')

    parser.add_argument('--cls_lambda', type=float, default=1.,
                        help='cut off distance (default: 3.)')
    parser.add_argument('--reg_lambda', type=float, default=1.,
                        help='cut off distance (default: 3.)')
    parser.add_argument('--entropy_lambda', type=float, default=1.,
                        help='cut off distance (default: 3.)')

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
    
    assert args.species in ['Si', 'Al'], 'The species must be Si or Al.'

    if args.species == 'Si':
        potentials = SI_POTENTIALS
    else:
        potentials = AL_POTENTIALS

    Dataset = PreTrainDataset(
        cache_dir=args.cache_dir,
        potentials=potentials,
        cut_per_atom_energy=args.cut_per_atom_energy
    )

    split_dict = load_splits(args.split_dir, args.split_name)
    Dataset.update_pretrain_configs(split_dict['pretrain']) # let the dataset know the pre-training configs do not have dft

    labels, counts = torch.unique(Dataset.ep_indices[Dataset.get_config_index(split_dict['train'])], return_counts=True)
    print(labels, counts)
    labels, counts = torch.unique(Dataset.ep_indices[Dataset.get_config_index(split_dict['val'])], return_counts=True)
    print(labels, counts)

    split_dict['train'] = split_dict['train'] + split_dict['pretrain'] if args.entropy_lambda > 0 else split_dict['train']
    split_id_dict = {key: Dataset.get_config_index(
        configs) for key, configs in split_dict.items()}

    pretrain_dataset = Dataset[split_id_dict['pretrain']]
    train_dataset = Dataset[split_id_dict['train']]
    val_dataset = Dataset[split_id_dict['val']]

    assert args.gnn in ['schnet', 'soapnet'], 'GNN must be in schnet or soapnet.'

    if args.gnn == 'schnet':
        model = SchNetAll(
            n_convs=args.num_convs,
            n_atoms=LARGEST_ATOMIC_NUMBER,
            n_output_heads=len(potentials),
            hidden_dims=args.hidden_dims,
            K=args.hidden_dims,
            dropout=args.dropout,
            activation=shifted_softplus,
            cut_r=args.cut_r,
        ).to(args.device)
        
        if args.pretrained_model_dir != '':
            print('Loading pretrained model weights from {}'.format(args.pretrained_model_dir))
            model.load_pretrained(args.pretrained_model_dir)

        lambdas = {
            'cls': args.cls_lambda,
            'reg': args.reg_lambda,
            'entropy': args.entropy_lambda
        }

        trainer = MGNNPreTrainerAll(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            problem='all',
            lambdas=lambdas,
            args=args
        )
    elif args.gnn == 'soapnet':
        model = SOAPNetAll(
            num_layers=args.num_convs,
            input_dims=252,
            hidden_dims=args.hidden_dims,
            n_output_heads=len(potentials),
            activation=shifted_softplus
        ).to(args.device)

        if args.pretrained_model_dir != '':
            print('Loading pretrained model weights from {}'.format(args.pretrained_model_dir))
            model.load_pretrained(args.pretrained_model_dir)

        lambdas = {
            'cls': args.cls_lambda,
            'reg': args.reg_lambda,
            'entropy': args.entropy_lambda
        }

        trainer = SOAPPreTrainerAll(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            problem='all',
            lambdas=lambdas,
            args=args
        )

    trainer.train()
    trainer.load_model()
    metrics, labels, preds = trainer.eval_one_epoch(pretrain_dataset)

    pred_df = Dataset.get_best_ep_preds(
        config_indices=split_id_dict['pretrain'],
        ep_logits=preds
    )
    pred_df.to_csv(osp.join(args.workspace, 'cls_labels.csv'), index=False, header=True)

if __name__ == '__main__':
    main()
