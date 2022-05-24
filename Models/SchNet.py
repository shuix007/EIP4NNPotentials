import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP, RegOutputModule, ClsOutputModule, SchNetGNN

class SchNet(nn.Module):
    def __init__(
        self,
        n_atoms,
        n_convs,
        hidden_dims,
        n_output_heads,
        activation,
        dropout,
        cut_r,
        K,
        problem
    ):
        super(SchNet, self).__init__()
        assert problem in ['cls', 'reg'], "Can only be cls or reg."
        self.problem = problem
        self.n_output_heads = n_output_heads

        self.gnn = SchNetGNN(
            node_feats=hidden_dims,
            hidden_feats=[hidden_dims]*n_convs,
            num_node_types=n_atoms+1,
            cutoff=cut_r,
            gap=cut_r/K,
            dropout=dropout
        )
        
        if problem == 'cls':
            self.output_layer = ClsOutputModule(
                hidden_dims=hidden_dims,
                n_output_heads=n_output_heads, 
                activation=activation,
                readout_type='sum'
            )
        else:
            self.output_layer = RegOutputModule(
                n_output_heads=n_output_heads,
                hidden_dims=hidden_dims,
                activation=activation
            )

    def save_pretrained(self, filename):
        torch.save({
            'gnn': self.gnn.state_dict(),
            'output_layer': self.output_layer.state_dict()
        }, filename)

    def load_pretrained(self, filename):
        pretrained_state_dict = torch.load(filename)
        self.gnn.load_state_dict(pretrained_state_dict['gnn'])

    def forward(self, batch_g, ndata, edata, is_contributing):
        ''' batch_g: a batch of DGLGraphs
            ndata: Atomic numbers
            edata: Edge distance
            is_contributing: if the item is contributing to the prediction of energy
        '''
        ndata = ndata.squeeze(1)

        h = self.gnn(batch_g, ndata, edata)
        preds = self.output_layer(batch_g, h, is_contributing)

        if self.n_output_heads == 1:
            preds = preds.squeeze(1)

        return preds


class SchNetAll(nn.Module):
    def __init__(
        self,
        n_atoms,
        n_convs,
        hidden_dims,
        n_output_heads,
        activation,
        dropout,
        cut_r,
        K
    ):
        super(SchNetAll, self).__init__()
        self.n_output_heads = n_output_heads

        self.gnn = SchNetGNN(
            node_feats=hidden_dims,
            hidden_feats=[hidden_dims]*n_convs,
            num_node_types=n_atoms+1,
            cutoff=cut_r,
            gap=cut_r/K,
            dropout=dropout
        )
        
        self.cls_output_layer = ClsOutputModule(
            hidden_dims=hidden_dims,
            n_output_heads=n_output_heads+1, 
            activation=activation,
            readout_type='sum'
        )
        self.reg_output_layer = RegOutputModule(
            n_output_heads=n_output_heads,
            hidden_dims=hidden_dims,
            activation=activation
        )

    def save_pretrained(self, filename):
        torch.save({
            'gnn': self.gnn.state_dict(),
            'cls_output_layer': self.cls_output_layer.state_dict(),
            'reg_output_layer': self.reg_output_layer.state_dict()
        }, filename)

    def load_pretrained(self, filename):
        pretrained_state_dict = torch.load(filename)
        self.gnn.load_state_dict(pretrained_state_dict['gnn'])

    def forward(self, batch_g, ndata, edata, is_contributing):
        ''' batch_g: a batch of DGLGraphs
            ndata: Atomic numbers
            edata: Edge distance
            is_contributing: if the item is contributing to the prediction of energy
        '''
        ndata = ndata.squeeze(1)

        h = self.gnn(batch_g, ndata, edata)
        cls_preds = self.cls_output_layer(batch_g, h, is_contributing)
        reg_preds = self.reg_output_layer(batch_g, h, is_contributing)

        if self.n_output_heads == 1:
            reg_preds = reg_preds.squeeze(1)

        return cls_preds, reg_preds