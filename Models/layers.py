import dgl
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import CFConv

def shifted_softplus(data):
    return F.softplus(data) - math.log(2)

class RBFExpansion(nn.Module):
    r"""Expand distances between nodes by radial basis functions.
    .. math::
        \exp(- \gamma * ||d - \mu||^2)
    where :math:`d` is the distance between two nodes and :math:`\mu` helps centralizes
    the distances. We use multiple centers evenly distributed in the range of
    :math:`[\text{low}, \text{high}]` with the difference between two adjacent centers
    being :math:`gap`.
    The number of centers is decided by :math:`(\text{high} - \text{low}) / \text{gap}`.
    Choosing fewer centers corresponds to reducing the resolution of the filter.
    Parameters
    ----------
    low : float
        Smallest center. Default to 0.
    high : float
        Largest center. Default to 30.
    gap : float
        Difference between two adjacent centers. :math:`\gamma` will be computed as the
        reciprocal of gap. Default to 0.1.
    """
    def __init__(self, low=0., high=30., gap=0.1):
        super(RBFExpansion, self).__init__()

        num_centers = int(np.ceil((high - low) / gap))
        self.centers = np.linspace(low, high, num_centers)
        self.centers = nn.Parameter(torch.tensor(self.centers).float(), requires_grad=False)
        self.gamma = 1 / gap

    def reset_parameters(self):
        """Reinitialize model parameters."""
        device = self.centers.device
        self.centers = nn.Parameter(
            self.centers.clone().detach().float(), requires_grad=False).to(device)

    def forward(self, edge_dists):
        """Expand distances.
        Parameters
        ----------
        edge_dists : float32 tensor of shape (E, 1)
            Distances between end nodes of edges, E for the number of edges.
        Returns
        -------
        float32 tensor of shape (E, len(self.centers))
            Expanded distances.
        """
        radial = edge_dists - self.centers
        coef = - self.gamma
        return torch.exp(coef * (radial ** 2))

class Interaction(nn.Module):
    """Building block for SchNet.
    SchNet is introduced in `SchNet: A continuous-filter convolutional neural network for
    modeling quantum interactions <https://arxiv.org/abs/1706.08566>`__.
    This layer combines node and edge features in message passing and updates node
    representations.
    Parameters
    ----------
    node_feats : int
        Size for the input and output node features.
    edge_in_feats : int
        Size for the input edge features.
    hidden_feats : int
        Size for hidden representations.
    """
    def __init__(self, node_feats, edge_in_feats, hidden_feats):
        super(Interaction, self).__init__()

        self.conv = CFConv(node_feats, edge_in_feats, hidden_feats, node_feats)
        self.project_out = nn.Linear(node_feats, node_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for layer in self.conv.project_edge:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.conv.project_node.reset_parameters()
        self.conv.project_out[0].reset_parameters()
        self.project_out.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Performs message passing and updates node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feats)
            Input node features, V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features, E for the number of edges.
        Returns
        -------
        float32 tensor of shape (V, node_feats)
            Updated node representations.
        """
        node_feats = self.conv(g, node_feats, edge_feats)
        return self.project_out(node_feats)

class SchNetGNN(nn.Module):
    """SchNet.
    SchNet is introduced in `SchNet: A continuous-filter convolutional neural network for
    modeling quantum interactions <https://arxiv.org/abs/1706.08566>`__.
    This class performs message passing in SchNet and returns the updated node representations.
    Parameters
    ----------
    node_feats : int
        Size for node representations to learn. Default to 64.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of hidden representations for the i-th interaction
        layer. ``len(hidden_feats)`` equals the number of interaction layers.
        Default to ``[64, 64, 64]``.
    num_node_types : int
        Number of node types to embed. Default to 100.
    cutoff : float
        Largest center in RBF expansion. Default to 30.
    gap : float
        Difference between two adjacent centers in RBF expansion. Default to 0.1.
    """
    def __init__(self, node_feats=64, hidden_feats=None, num_node_types=100, cutoff=30., gap=0.1, dropout=0.):
        super(SchNetGNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        if hidden_feats is None:
            hidden_feats = [64, 64, 64]

        self.embed = nn.Embedding(num_node_types, node_feats)
        self.rbf = RBFExpansion(high=cutoff, gap=gap)

        n_layers = len(hidden_feats)
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(
                Interaction(node_feats, len(self.rbf.centers), hidden_feats[i]))

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.embed.reset_parameters()
        self.rbf.reset_parameters()
        for layer in self.gnn_layers:
            layer.reset_parameters()

    def forward(self, g, node_types, edge_dists):
        """Performs message passing and updates node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_types : int64 tensor of shape (V)
            Node types to embed, V for the number of nodes.
        edge_dists : float32 tensor of shape (E, 1)
            Distances between end nodes of edges, E for the number of edges.
        Returns
        -------
        node_feats : float32 tensor of shape (V, node_feats)
            Updated node representations.
        """
        node_feats = self.embed(node_types)
        expanded_dists = self.rbf(edge_dists)
        for gnn in self.gnn_layers:
            node_feats = self.dropout(node_feats)
            node_feats = gnn(g, node_feats, expanded_dists)
        return node_feats

class MLP(nn.Module):
    """MLP-based Readout.
    This layer updates node representations with a MLP and computes graph representations
    out of node representations with max, mean or sum.
    Parameters
    ----------
    node_feats : int
        Size for the input node features.
    hidden_feats : int
        Size for the hidden representations.
    graph_feats : int
        Size for the output graph representations.
    activation : callable
        Activation function. Default to None.
    mode : 'max' or 'mean' or 'sum'
        Whether to compute elementwise maximum, mean or sum of the node representations.
    """

    def __init__(self, node_feats, hidden_feats, graph_feats, activation=None):
        super(MLP, self).__init__()

        self.in_project = nn.Linear(node_feats, hidden_feats)
        self.activation = activation
        self.out_project = nn.Linear(hidden_feats, graph_feats)

    def forward(self, g, node_feats):
        """Computes graph representations out of node features.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feats)
            Input node features, V for the number of nodes.
        Returns
        -------
        graph_feats : float32 tensor of shape (G, graph_feats)
            Graph representations computed. G for the number of graphs.
        """
        node_feats = self.in_project(node_feats)
        if self.activation is not None:
            node_feats = self.activation(node_feats)
        node_feats = self.out_project(node_feats)

        return node_feats

class MMLP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dims,
        hidden_dims,
        output_dims,
        activation
    ):
        super(MMLP, self).__init__()

        self.num_layers = num_layers
        self.activation = activation
        self.layers = nn.ModuleList()

        # the initial layer
        self.layers.append(nn.Linear(input_dims, hidden_dims, bias=True))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dims, hidden_dims, bias=True))
        # the last layer
        self.layers.append(nn.Linear(hidden_dims, output_dims, bias=True))

    def forward(self, g, input_feats):
        h = input_feats

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != self.num_layers - 1:
                h = self.activation(h)
        return h

class RegOutputModule(nn.Module):
    def __init__(
        self,
        hidden_dims,
        n_output_heads,
        activation
    ):
        super(RegOutputModule, self).__init__()

        self.mlp = MLP(
            node_feats=hidden_dims,
            hidden_feats=hidden_dims,
            graph_feats=n_output_heads,
            activation=activation
        )

    def forward(self, batch_g, node_feats, is_contributing=None):
        score = self.mlp(batch_g, node_feats)  # n * num_output_heads

        if is_contributing is not None:
            score = score * is_contributing

        batch_g.ndata.update({'score': score})
        preds = dgl.sum_nodes(
            batch_g, 'score')  # n * num_outputs

        return preds


class ClsOutputModule(nn.Module):
    def __init__(self, 
        hidden_dims, 
        n_output_heads, 
        activation,
        readout_type='sum'
    ):
        super(ClsOutputModule, self).__init__()

        if readout_type == 'sum':
            self.readout_fn = dgl.sum_nodes
        elif readout_type == 'mean':
            self.readout_fn = dgl.mean_nodes
        elif readout_type == 'max':
            self.readout_fn = dgl.max_nodes
        else:
            raise ValueError(
                'Readout function {} not supported.'.format(readout_type))

        self.batch_norm = nn.BatchNorm1d(hidden_dims)
        self.fc = MLP(
            node_feats=hidden_dims,
            hidden_feats=hidden_dims,
            graph_feats=n_output_heads,
            activation=activation
        )

    def forward(self, batch_g, node_feats, is_contributing=None):
        batch_g.ndata.update({
            'node_feats': node_feats
        })
        graph_feats = self.readout_fn(batch_g, 'node_feats')
        graph_feats = self.fc(batch_g, self.batch_norm(graph_feats))

        return graph_feats