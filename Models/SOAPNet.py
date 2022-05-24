import dgl
import math
from regex import R
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MMLP, MLP, ClsOutputModule, RegOutputModule

class SOAPNet(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dims,
        hidden_dims,
        n_output_heads,
        activation,
        problem
    ):
        super(SOAPNet, self).__init__()
        assert problem in ['cls', 'reg'], "Can only be cls or reg."
        self.n_output_heads = n_output_heads

        self.gnn = MMLP(
            num_layers=num_layers,
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            output_dims=hidden_dims,
            activation=activation
        )
        
        if problem == 'cls':
            self.output_layer = ClsOutputModule(
                input_dims=hidden_dims,
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

    def forward(self, batch_g, ndata, is_contributing):
        ''' batch_g: a batch of DGLGraphs
            ndata: Atom descriptors
        '''
        h = self.gnn(batch_g, ndata)
        preds = self.output_layer(batch_g, h, is_contributing)

        if self.n_output_heads == 1:
            preds = preds.squeeze(1)

        return preds


class SOAPNetAll(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dims,
        hidden_dims,
        n_output_heads,
        activation
    ):
        super(SOAPNetAll, self).__init__()
        self.n_output_heads = n_output_heads

        self.gnn = MMLP(
            num_layers=num_layers,
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            output_dims=hidden_dims,
            activation=activation
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

    def forward(self, batch_g, ndata, is_contributing):
        ''' batch_g: a batch of DGLGraphs
            ndata: Atom descriptors
        '''
        h = self.gnn(batch_g, ndata)
        cls_preds = self.cls_output_layer(batch_g, h, is_contributing)
        reg_preds = self.reg_output_layer(batch_g, h, is_contributing)

        if self.n_output_heads == 1:
            reg_preds = reg_preds.squeeze(1)

        return cls_preds, reg_preds