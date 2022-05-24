import os
import time
import dgl
from dgl.batch import batch
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import collate_dgl
from scheduler import get_slanted_triangular_scheduler, get_linear_scheduler


class Trainer(object):
    def __init__(self, model, train_dataset, val_dataset, test_dataset, args):
        self.device = args.device
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.tol = args.tol

        self.workspace = args.workspace
        self.model = model

        self.loss_fn = nn.MSELoss()
        self.best_epoch = 0
        self.best_metric = 100.
        self.comp_func = np.less

        self.optimizer = Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            amsgrad=False
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=args.decay_epochs,
            factor=args.lr_decrease_rate,
            verbose=True
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_dgl
        )

        self.eval_train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_dgl
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_dgl
        )

        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_dgl
        )

        self.logger = SummaryWriter(
            os.path.join(self.workspace, 'log'))

    def compute_loss(self, preds, labels):
        if preds.size(0) > 0:
            return self.loss_fn(preds, labels)
        else:
            return torch.tensor([0.], device=self.device)

    def compute_metrics(self, preds, labels):
        MAE = (preds-labels).abs().mean()
        return MAE
        # return self.loss_fn(preds, labels)

    def early_stop(self, metric, epoch):
        if self.comp_func(metric, self.best_metric):
            self.best_metric = metric
            self.best_epoch = epoch
            print('Saving the current best model at metric: {:.4f}, epoch: {}'.format(metric, epoch))
            self.save_model()

        return (epoch - self.best_epoch >= self.tol)

    def save_model(self):
        model_filename = os.path.join(self.workspace, 'best_model.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'metrics_dict': self.best_metric
        }, model_filename)

    def load_model(self):
        model_filename = os.path.join(self.workspace, 'best_model.pt')
        state_dict = torch.load(model_filename)

        self.model.load_state_dict(state_dict['model_state_dict'])
        self.best_metric = state_dict['metrics_dict']

    def checkpoint(self, epoch):
        checkpoint_filename = os.path.join(self.workspace, 'checkpoint.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, checkpoint_filename)

    def load_checkpoint(self):
        checkpoint_filename = os.path.join(self.workspace, 'checkpoint.pt')
        state_dict = torch.load(checkpoint_filename)

        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])

        return state_dict['epoch']

    def log_tensorboard(self, metrics, losses, epoch):
        ''' Write experiment log to tensorboad. '''

        self.logger.add_scalars('Training loss', losses, epoch)
        self.logger.add_scalars('MAE', metrics, epoch)

    def log_print(self, metrics, losses, epoch, train_time, eval_time):
        ''' Stdout of experiment log. '''

        print('===========================')
        print('Epoch: {}, train time: {:.4f}, eval time: {:.4f}'.format(
            epoch, train_time, eval_time))

        loss_str = ' '.join([key + ': {:.4f}'.format(value)
                             for key, value in losses.items()])
        metric_str = ' '.join(
            [key + ': {:.4f}'.format(value) for key, value in metrics.items()])
        print('Training loss: ' + loss_str)
        print('Val MAE: ' + metric_str)


class FineTuner(Trainer):
    def __init__(self, model, train_dataset, val_dataset, test_dataset, args):
        self.device = args.device
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.scheduler_type = args.scheduler
        self.max_grad_norm = args.max_grad_norm
        self.pgratio = args.pgratio
        self.alpha = args.alpha
        self.tol = args.tol

        assert self.scheduler_type in ['linear', 'slanted', 'reduce_on_pleatue'], \
            'Scheduler can only be selected from linear, slanted, reduce_on_pleatue'

        self.workspace = args.workspace
        self.model = model

        self.best_epoch = 0
        self.loss_fn = nn.MSELoss()
        self.best_metric = 1e8
        self.comp_func = np.less

        self.optimizer = Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            amsgrad=False
        )

        if self.scheduler_type == 'reduce_on_pleatue':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=args.decay_epochs,
                factor=args.lr_decrease_rate,
                verbose=True
            )
        elif self.scheduler_type == 'linear':
            self.scheduler = get_linear_scheduler(
                optimizer=self.optimizer,
                num_training_epochs=self.num_epochs,
                initial_lr=args.learning_rate
            )
        elif self.scheduler_type == 'slanted':
            self.scheduler = get_slanted_triangular_scheduler(
                optimizer=self.optimizer,
                num_epochs=self.num_epochs
            )

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_dgl,
            drop_last=True
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_dgl
        )

        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_dgl
        )

        self.logger = SummaryWriter(
            os.path.join(self.workspace, 'log'))

    def compute_per_atom_error(self, preds, labels, num_atoms):
        return ((preds-labels).abs() / num_atoms).mean()

    def compute_tukey_loss(self, preds, labels, has_dft):
        c = 4.6851
        # if there are surrogate labels
        if has_dft.sum() < has_dft.size(0):
            MAD = (labels - preds.detach()).abs().median()

            r = (labels[~has_dft] - preds[~has_dft]) / (1.4826 * MAD)
            
            inlier = (r.abs() <= c)
            inlier_loss = (c**2 * (1 - (1 - (r[inlier] / c)**2)**3) / 6).sum()
            outlier_loss = (c**2 / 6) * (~inlier).sum()

            return (inlier_loss + outlier_loss) / r.size(0)
        else:
            return torch.tensor([0.], device=self.device)

    def train(self):
        for epoch in range(1, self.num_epochs+1):
            start = time.time()
            losses = self.train_one_epoch()
            end_train = time.time()
            metrics, _, = self.eval_one_epoch(test=False)
            end = time.time()

            self.log_tensorboard(metrics, losses, epoch)
            self.log_print(metrics, losses, epoch,
                           end_train-start, end-end_train)

            if self.scheduler_type == 'reduce_on_pleatue':
                self.scheduler.step(metrics['combined'])
            else:
                self.scheduler.step()

            if self.early_stop(metric=metrics['combined'], epoch=epoch):
                break

    def test(self):
        metrics, preds = self.eval_one_epoch(test=True)
        print('Best validation results:')
        print(self.best_metric)
        print('Test results:')
        print(' '.join([key + ': {:.4f}'.format(value)
                        for key, value in metrics.items()]))
        return preds

    def train_one_epoch(self):
        pass

    def eval_one_epoch(self):
        pass


class PreTrainer(Trainer):
    def __init__(self, model, train_dataset, val_dataset, problem, args):
        self.device = args.device
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.max_grad_norm = args.max_grad_norm
        self.problem = problem
        self.tol = args.tol

        assert self.problem in ['cls', 'reg', 'all'], "Can only be cls or reg problem."

        self.workspace = args.workspace
        self.model = model

        self.best_epoch = 0
        if self.problem == 'reg':
            self.reg_loss_fn = nn.MSELoss()
            self.best_metric = 1e8
            self.comp_func = np.less
        elif self.problem == 'cls':
            self.cls_loss_fn = nn.CrossEntropyLoss()
            self.best_metric = -np.inf
            self.comp_func = np.greater
        else:
            self.cls_loss_fn = nn.CrossEntropyLoss()
            self.reg_loss_fn = nn.MSELoss()
            self.best_metric = -np.inf
            self.comp_func = np.greater

        self.optimizer = Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            amsgrad=False
        )

        self.scheduler = get_slanted_triangular_scheduler(
            self.optimizer,
            num_epochs=self.num_epochs
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_dgl,
            drop_last=True
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_dgl
        )

        self.logger = SummaryWriter(
            os.path.join(self.workspace, 'log'))

    def compute_loss(self, preds, labels):
        if self.problem == 'reg':
            return self.compute_reg_loss(preds, labels)
        elif self.problem == 'cls':
            return self.compute_cls_loss(preds, labels)

    def compute_cls_loss(self, preds, labels):
        if preds.size(0) > 0:
            return self.cls_loss_fn(preds, labels)
        else:
            return torch.tensor([0.], device=self.device)

    def compute_reg_loss(self, preds, labels):
        masks = labels < 200000
        return self.reg_loss_fn(preds[masks], labels[masks])

    def compute_entropy_loss(self, preds):
        if preds.size(0) > 0:
            probs = F.softmax(preds, -1) + 1e-8
            entropy = - probs * torch.log(probs)
            entropy = torch.sum(entropy, -1)
            return entropy.mean()
        else:
            return torch.tensor([0.], device=self.device)

    def compute_metrics(self, preds, labels):
        if self.problem == 'reg':
            return self.compute_reg_metrics(preds, labels)
        else:
            return self.compute_cls_metrics(preds, labels)

    def compute_cls_metrics(self, preds, labels):
        pred_labels = preds.argmax(dim=1)
        acc = (pred_labels == labels).sum() / labels.size(0)
        return acc
    
    def compute_reg_metrics(self, preds, labels):
        # remove outliers
        masks = labels < 200000
        MAE = (preds[masks] - labels[masks]).abs().mean()
        return MAE

    def train(self):
        for epoch in range(1, self.num_epochs+1):
            start = time.time()
            losses = self.train_one_epoch()
            end_train = time.time()
            metrics, _, _ = self.eval_one_epoch()
            end = time.time()

            self.log_tensorboard(
                metrics,
                losses,
                epoch
            )
            self.log_print(metrics, losses, epoch,
                           end_train-start, end-end_train)
            self.scheduler.step()

            early_stop_metric = metrics['combined'] if self.cls_lambda > self.reg_lambda else -metrics['reg']
            if self.early_stop(metric=early_stop_metric, epoch=epoch):
                break

    def save_model(self):
        model_filename = os.path.join(self.workspace, 'best_model.pt')
        self.model.save_pretrained(model_filename)

        torch.save(self.model, os.path.join(self.workspace, 'best_model_full.pt'))

    def load_model(self):
        self.model = torch.load(os.path.join(self.workspace, 'best_model_full.pt')).to(self.device)

    def train_one_epoch(self):
        pass

    def eval_one_epoch(self):
        pass


class MGNNFineTuner(FineTuner):
    def train_one_epoch(self):
        total_loss = 0.

        self.model.train()

        for batched_graphs, labels, has_dft in self.train_dataloader:
            batched_graphs = batched_graphs.to(self.device)
            node_feats = batched_graphs.ndata.pop('node_feats')
            edge_feats = batched_graphs.edata.pop('edge_feats')
            image = batched_graphs.ndata.pop('image')
            is_contributing = batched_graphs.ndata.pop('is_contributing')

            labels = labels.to(self.device)
            has_dft = has_dft.to(self.device)

            preds = self.model(
                batched_graphs,
                node_feats,
                edge_feats,
                is_contributing
            )

            self.optimizer.zero_grad()

            dft_loss = self.compute_loss(preds[has_dft], labels[has_dft])
            purterbed_dft_loss = self.compute_tukey_loss(preds, labels, has_dft)

            loss = dft_loss + self.alpha * self.pgratio * purterbed_dft_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if torch.isnan(loss).any().item() or torch.isinf(loss).any().item():
                raise ValueError('Nan/Inf Loss.')

            total_loss += loss.item()

        return {
            'combined': total_loss / len(self.train_dataloader)
        }

    def eval_one_epoch(self, test):
        self.model.eval()

        if test:
            dataloader = self.test_dataloader  # for inference
        else:
            dataloader = self.val_dataloader  # for validation

        Preds, Labels, NumAtoms = list(), list(), list()
        with torch.no_grad():
            for batched_graphs, labels, has_dft in dataloader:
                NumAtoms.append(dgl.sum_nodes(
                    batched_graphs, feat='is_contributing').squeeze())

                batched_graphs = batched_graphs.to(self.device)
                node_feats = batched_graphs.ndata.pop('node_feats')
                edge_feats = batched_graphs.edata.pop('edge_feats')
                image = batched_graphs.ndata.pop('image')
                is_contributing = batched_graphs.ndata.pop('is_contributing')

                preds = self.model(
                    batched_graphs,
                    node_feats,
                    edge_feats,
                    is_contributing
                )

                Labels.append(labels)
                Preds.append(preds.detach().cpu())

            Labels = torch.cat(Labels)
            Preds = torch.cat(Preds)
            NumAtoms = torch.cat(NumAtoms)

            combined_metrics = self.compute_metrics(Preds, Labels).item()
            per_atom_metrics = self.compute_per_atom_error(
                Preds, Labels, NumAtoms).item()

            metrics = {
                'combined': combined_metrics,
                'per_atom': per_atom_metrics
            }

            return (
                metrics,
                Preds.numpy(),
            )

class SOAPFineTuner(FineTuner):
    def train_one_epoch(self):
        total_loss = 0.

        self.model.train()

        for batched_graphs, labels, has_dft in self.train_dataloader:
            batched_graphs = batched_graphs.to(self.device)
            node_feats = batched_graphs.ndata.pop('node_feats')
            is_contributing = batched_graphs.ndata.pop('is_contributing')

            labels = labels.to(self.device)
            has_dft = has_dft.to(self.device)

            preds = self.model(
                batched_graphs,
                node_feats,
                is_contributing
            )

            self.optimizer.zero_grad()

            dft_loss = self.compute_loss(preds[has_dft], labels[has_dft])
            purterbed_dft_loss = self.compute_tukey_loss(preds, labels, has_dft)

            loss = dft_loss + self.alpha * self.pgratio * purterbed_dft_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if torch.isnan(loss).any().item() or torch.isinf(loss).any().item():
                raise ValueError('Nan/Inf Loss.')

            total_loss += loss.item()

        return {
            'combined': total_loss / len(self.train_dataloader)
        }

    def eval_one_epoch(self, test):
        self.model.eval()

        if test:
            dataloader = self.test_dataloader  # for inference
        else:
            dataloader = self.val_dataloader  # for validation

        Preds, Labels, NumAtoms = list(), list(), list()
        with torch.no_grad():
            for batched_graphs, labels, has_dft in dataloader:
                NumAtoms.append(dgl.sum_nodes(
                    batched_graphs, feat='is_contributing').squeeze())

                batched_graphs = batched_graphs.to(self.device)
                node_feats = batched_graphs.ndata.pop('node_feats')
                is_contributing = batched_graphs.ndata.pop('is_contributing')

                preds = self.model(
                    batched_graphs,
                    node_feats,
                    is_contributing
                )

                Labels.append(labels)
                Preds.append(preds.detach().cpu())

            Labels = torch.cat(Labels)
            Preds = torch.cat(Preds)
            NumAtoms = torch.cat(NumAtoms)

            combined_metrics = self.compute_metrics(Preds, Labels).item()
            per_atom_metrics = self.compute_per_atom_error(
                Preds, Labels, NumAtoms).item()

            metrics = {
                'combined': combined_metrics,
                'per_atom': per_atom_metrics
            }

            return (
                metrics,
                Preds.numpy(),
            )


class MGNNPreTrainerAll(PreTrainer):
    def __init__(self, model, train_dataset, val_dataset, problem, lambdas, args):
        self.device = args.device
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.max_grad_norm = args.max_grad_norm
        self.problem = problem
        self.tol = args.tol

        self.reg_lambda = lambdas['reg']
        self.cls_lambda = lambdas['cls']
        self.entropy_lambda = lambdas['entropy']

        assert self.problem in ['cls', 'reg', 'all'], "Can only be cls or reg problem."

        self.workspace = args.workspace
        self.model = model

        self.best_epoch = 0
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.MSELoss()
        self.best_metric = -np.inf
        self.comp_func = np.greater

        self.optimizer = Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            amsgrad=False
        )

        self.scheduler = get_slanted_triangular_scheduler(
            self.optimizer,
            num_epochs=self.num_epochs
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_dgl,
            drop_last=True
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_dgl
        )

        self.logger = SummaryWriter(
            os.path.join(self.workspace, 'log'))

    def train_one_epoch(self):
        total_loss = 0.
        total_cls_loss = 0.
        total_reg_loss = 0.
        total_entropy_loss = 0.

        self.model.train()
        for batched_graphs, reg_labels, cls_labels, has_cls_labels in self.train_dataloader:
            batched_graphs = batched_graphs.to(self.device)
            node_feats = batched_graphs.ndata.pop('node_feats')
            edge_feats = batched_graphs.edata.pop('edge_feats')
            image = batched_graphs.ndata.pop('image')
            is_contributing = batched_graphs.ndata.pop('is_contributing')

            reg_labels = reg_labels.to(self.device)
            cls_labels = cls_labels.to(self.device)
            has_cls_labels = has_cls_labels.to(self.device)

            cls_preds, reg_preds = self.model(
                batched_graphs,
                node_feats,
                edge_feats,
                is_contributing
            )

            self.optimizer.zero_grad()
            cls_loss = self.compute_cls_loss(cls_preds[has_cls_labels], cls_labels[has_cls_labels])
            reg_loss = self.compute_reg_loss(reg_preds, reg_labels)
            entropy_loss = self.compute_entropy_loss(cls_preds[~has_cls_labels])

            loss = self.cls_lambda * cls_loss \
                 + self.reg_lambda * reg_loss \
                 + self.entropy_lambda * entropy_loss
                 
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if torch.isnan(loss).any().item() or torch.isinf(loss).any().item():
                raise ValueError('Nan/Inf Loss.')

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            total_entropy_loss += entropy_loss.item()

        return {
            'total': total_loss / len(self.train_dataloader),
            'cls': total_cls_loss / len(self.train_dataloader),
            'reg': total_reg_loss / len(self.train_dataloader),
            'entropy': total_entropy_loss / len(self.train_dataloader)
        }

    def eval_one_epoch(self, dataset=None):
        self.model.eval()

        CLSPreds, CLSLabels = list(), list()
        REGPreds, REGLabels = list(), list()
        if dataset is None:
            dataloader = self.val_dataloader
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=128,
                shuffle=False,
                collate_fn=collate_dgl
            )
            print('Using outside loader.')
            
        with torch.no_grad():
            for batched_graphs, reg_labels, cls_labels, has_cls_labels in dataloader:
                batched_graphs = batched_graphs.to(self.device)
                node_feats = batched_graphs.ndata.pop('node_feats')
                edge_feats = batched_graphs.edata.pop('edge_feats')
                image = batched_graphs.ndata.pop('image')
                is_contributing = batched_graphs.ndata.pop('is_contributing')

                cls_preds, reg_preds = self.model(
                    batched_graphs,
                    node_feats,
                    edge_feats,
                    is_contributing
                )

                CLSLabels.append(cls_labels)
                REGLabels.append(reg_labels)
                CLSPreds.append(cls_preds.detach().cpu())
                REGPreds.append(reg_preds.detach().cpu())

            CLSLabels = torch.cat(CLSLabels)
            REGLabels = torch.cat(REGLabels)
            CLSPreds = torch.cat(CLSPreds)
            REGPreds = torch.cat(REGPreds)

            cls_metrics = self.compute_cls_metrics(CLSPreds, CLSLabels).item()
            reg_metrics = self.compute_reg_metrics(REGPreds, REGLabels).item()

            metrics = {
                'combined': cls_metrics,
                'reg': reg_metrics
            }

            return metrics, CLSLabels.numpy(), CLSPreds.numpy()


class SOAPPreTrainerAll(PreTrainer):
    def __init__(self, model, train_dataset, val_dataset, problem, lambdas, args):
        self.device = args.device
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.max_grad_norm = args.max_grad_norm
        self.problem = problem
        self.tol = args.tol

        self.reg_lambda = lambdas['reg']
        self.cls_lambda = lambdas['cls']
        self.entropy_lambda = lambdas['entropy']

        assert self.problem in ['cls', 'reg', 'all'], "Can only be cls or reg problem."

        self.workspace = args.workspace
        self.model = model

        self.best_epoch = 0
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.MSELoss()
        self.best_metric = -np.inf
        self.comp_func = np.greater

        self.optimizer = Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            amsgrad=False
        )

        self.scheduler = get_slanted_triangular_scheduler(
            self.optimizer,
            num_epochs=self.num_epochs
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_dgl,
            drop_last=True
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_dgl
        )

        self.logger = SummaryWriter(
            os.path.join(self.workspace, 'log'))

    def train_one_epoch(self):
        total_loss = 0.
        total_cls_loss = 0.
        total_reg_loss = 0.
        total_entropy_loss = 0.

        self.model.train()
        for batched_graphs, reg_labels, cls_labels, has_cls_labels in self.train_dataloader:
            batched_graphs = batched_graphs.to(self.device)
            node_feats = batched_graphs.ndata.pop('node_feats')
            is_contributing = batched_graphs.ndata.pop('is_contributing')

            reg_labels = reg_labels.to(self.device)
            cls_labels = cls_labels.to(self.device)
            has_cls_labels = has_cls_labels.to(self.device)

            cls_preds, reg_preds = self.model(
                batched_graphs,
                node_feats,
                is_contributing
            )

            self.optimizer.zero_grad()
            cls_loss = self.compute_cls_loss(cls_preds[has_cls_labels], cls_labels[has_cls_labels])
            reg_loss = self.compute_reg_loss(reg_preds, reg_labels)
            entropy_loss = self.compute_entropy_loss(cls_preds[~has_cls_labels])

            loss = self.cls_lambda * cls_loss \
                 + self.reg_lambda * reg_loss \
                 + self.entropy_lambda * entropy_loss
                 
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if torch.isnan(loss).any().item() or torch.isinf(loss).any().item():
                raise ValueError('Nan/Inf Loss.')

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            total_entropy_loss += entropy_loss.item()

        return {
            'total': total_loss / len(self.train_dataloader),
            'cls': total_cls_loss / len(self.train_dataloader),
            'reg': total_reg_loss / len(self.train_dataloader),
            'entropy': total_entropy_loss / len(self.train_dataloader)
        }

    def eval_one_epoch(self, dataset=None):
        self.model.eval()

        CLSPreds, CLSLabels = list(), list()
        REGPreds, REGLabels = list(), list()
        if dataset is None:
            dataloader = self.val_dataloader
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=128,
                shuffle=False,
                collate_fn=collate_dgl
            )
            print('Using outside loader.')
            
        with torch.no_grad():
            for batched_graphs, reg_labels, cls_labels, has_cls_labels in dataloader:
                batched_graphs = batched_graphs.to(self.device)
                node_feats = batched_graphs.ndata.pop('node_feats')
                is_contributing = batched_graphs.ndata.pop('is_contributing')

                cls_preds, reg_preds = self.model(
                    batched_graphs,
                    node_feats,
                    is_contributing
                )

                CLSLabels.append(cls_labels)
                REGLabels.append(reg_labels)
                CLSPreds.append(cls_preds.detach().cpu())
                REGPreds.append(reg_preds.detach().cpu())

            CLSLabels = torch.cat(CLSLabels)
            REGLabels = torch.cat(REGLabels)
            CLSPreds = torch.cat(CLSPreds)
            REGPreds = torch.cat(REGPreds)

            cls_metrics = self.compute_cls_metrics(CLSPreds, CLSLabels).item()
            reg_metrics = self.compute_reg_metrics(REGPreds, REGLabels).item()

            metrics = {
                'combined': cls_metrics,
                'reg': reg_metrics
            }

            return metrics, CLSLabels.numpy(), CLSPreds.numpy()