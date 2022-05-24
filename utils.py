import os
import warnings
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch

import ase.geometry
import ase.neighborlist
from ase.data import atomic_numbers

def save_args(args, workspace):
    args_path = os.path.join(workspace, 'args.txt')
    with open(args_path, 'w') as f:
        f.write(str(args).replace(', ', ',\n'))

def get_pbc_graphs(
    pos: np.ndarray,
    species: List[str],
    r_cut: float,
    cell: np.ndarray,
    pbc: Tuple[bool, bool, bool] = (True, True, True),
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[bool]]:
    """
    Create distance-based graph for crystals, with the ability to handle periodic
    boundary conditions (PBCs).
    Args:
        pos: shape (N, 3), positions of atoms, where N is the number of atoms before
            considering the PBCs, i.e. the number of contributing atoms.
        species: atomic species of the N atoms.
        r_cut: cutoff distance to determine neighbors.
        cell: shape (3, 3), supercell of the crystal, the 1st, 2nd, and 3rd rows
            represent the three cell vectors of the crystal.
        pbc: whether to apply PBCs along the 1st, 2nd, and 3rd cell vectors.
    Returns:
        new_edge_index (2, num_edges): all edges between the atoms
        pos_all (num_atoms, 3): positions of all atoms (both contributing and padding
            atoms). contributing atoms: original atoms that exists in the cell;
            padding atoms: newly created atoms to satisfy PBCs.
        species_all (num_atoms): atomic species of all atoms.
        image_of_all (num_atoms): an atom is an image of which atom? each contributing
            atoms is an image of itself; a padding atom is an image of a contributing
            atom.
        is_contributing_all (num_atoms): whether an atom is a contributing atom.
    """

    pos = np.asarray(pos)
    assert pos.shape[1] == 3, "pos should be N by 3 array"

    species = np.asarray(species)
    assert pos.shape[0] == species.shape[0], "pos size and species size should be equal"

    cell = np.asarray(cell)
    assert cell.shape == (3, 3), "cell should be of shape (3, 3)"

    edge_index, shifts, cell, _ = neighbor_list_and_relative_vec(
        pos, r_cut, cell=cell, pbc=pbc
    )

    i, j = edge_index

    # identify padding atoms in two steps:
    # - find all padding edges across cell, i.e. shift != (0,0,0) ones
    # - find unique padding atoms, i.e. unique (j, shift) pairs
    # then, unique paddings are the intersection of the two
    # step 1
    zero_shift = np.array([0, 0, 0], dtype=shifts.dtype)
    is_padding_edge = (shifts != zero_shift).any(axis=1)
    j_padding = j[is_padding_edge]
    shifts_padding = shifts[is_padding_edge]
    # step 2
    js_padding = np.hstack((j_padding.reshape(-1, 1), shifts_padding))
    unique, index, inverse = np.unique(
        js_padding, axis=0, return_index=True, return_inverse=True
    )
    j_padding_unique = j_padding[index]
    shifts_padding_unique = shifts_padding[index]

    # re-index padding atoms, starting from the number of contributing atoms (n_contrib)
    n_contrib = len(pos)
    n_padding = len(unique)
    padding_atom_indices = np.arange(n_contrib, n_contrib + n_padding)

    # change padding j to re-indexed indices
    # reverse step 2
    j_padding_new_index = padding_atom_indices[inverse]
    # reverse step 1
    new_j = j.copy()
    new_j[is_padding_edge] = j_padding_new_index
    new_edge_index = np.vstack((i, new_j))

    # generate pos and species for padding atoms
    pos_padding = pos[j_padding_unique] + np.dot(shifts_padding_unique, cell)
    species_padding = species[j_padding_unique]

    pos_all = np.vstack((pos, pos_padding))
    species_all = np.concatenate((species, species_padding))
    atomic_numbers_all = np.array([atomic_numbers[atomic_symbol] for atomic_symbol in species_all], dtype=np.int64)

    # original atoms are contributing, padding atoms are not
    is_contributing_all = np.concatenate(
        (np.ones(n_contrib, dtype=bool), np.zeros(n_padding, dtype=bool))
    )

    # image of (an original atom is an image of itself, a padding atom is an image of
    # some original atom)
    original_image_of = np.arange(n_contrib)
    padding_image_of = j_padding_unique
    image_of_all = np.concatenate((original_image_of, padding_image_of))

    return new_edge_index, pos_all, species_all, atomic_numbers_all, image_of_all, is_contributing_all

def neighbor_list_and_relative_vec(
    pos: np.ndarray,
    r_max: float,
    self_interaction: bool = False,
    strict_self_interaction: bool = True,
    cell: np.ndarray = None,
    pbc: Union[bool, List[bool]] = False,
):
    """
    Create neighbor list (``edge_index``) and relative vectors (``edge_attr``) based on
    radial cutoff.
    Edges are given by the following convention:
    - ``edge_index[0]`` is the *source* (convolution center).
    - ``edge_index[1]`` is the *target* (neighbor).
    Thus, ``edge_index`` has the same convention as the relative vectors:
    :math:`\\vec{r}_{source, target}`
    Args:
        pos (shape [N, 3]): Positional coordinate; Tensor or numpy array. If Tensor,
            must be on CPU.
        r_max (float): Radial cutoff distance for neighbor finding.
        self_interaction (bool): Whether to include same periodic image self-edges in
            the neighbor list. Should be False for most applications.
        strict_self_interaction (bool): Whether to include *any* self interaction edges
            in the graph, even if the two instances of the atom are in different
            periodic images. Should be True for most applications.
        pbc (bool or 3-tuple of bool): Whether the system is periodic in each of the
            three cell dimensions.
        cell (shape [3, 3]): Cell for periodic boundary conditions.
            Ignored if ``pbc == False``.
    Returns:
        edge_index (shape [2, num_edges]): List of edges.
        edge_cell_shift (shape [num_edges, 3]): Relative cell shift
            vectors. Returned only if cell is not None.
        cell (shape [3, 3]): the cell. Returned only if cell is not None.
        num_neigh ([N]) number of neighbors for each atom.
    """
    if isinstance(pbc, bool):
        pbc = (pbc,) * 3

    temp_pos = np.asarray(pos)

    # Get a cell on the CPU no matter what
    if cell is not None:
        cell = np.array(cell)
    else:
        # ASE will "complete" this correctly.
        cell = np.zeros((3, 3), dtype=temp_pos.dtype)

    # ASE dependent part
    cell = ase.geometry.complete_cell(cell)

    first_idex, second_idex, shifts = ase.neighborlist.primitive_neighbor_list(
        "ijS",
        pbc,
        cell,
        temp_pos,
        cutoff=float(r_max),
        # we want edges from atom to itself in different periodic images!
        self_interaction=strict_self_interaction,
        use_scaled_positions=False,
    )

    # Eliminate true self-edges that don't cross periodic boundaries
    if not self_interaction:
        bad_edge = first_idex == second_idex
        bad_edge &= np.all(shifts == 0, axis=1)
        keep_edge = ~bad_edge
        if not np.any(keep_edge):
            raise ValueError(
                "After eliminating self edges, no edges remain in this system."
            )
        first_idex = first_idex[keep_edge]
        second_idex = second_idex[keep_edge]
        shifts = shifts[keep_edge]

    # Build output:
    edge_index = np.vstack((first_idex, second_idex))

    # Number of neighbors for each atoms
    num_neigh = np.bincount(first_idex)

    # Some atoms with large atom index may not have neighbors due to the use of bincount
    # As a concrete example, suppose we have 5 atoms and first_idex is [0,1,1,3,3,3,3],
    # then bincount will be [1, 2, 0, 4], which means atoms 0,1,2,3 have 1,2,0,4
    # neighbors respectively. Although atom 2 is handled by bincount, atom 4 cannot.
    # The below part is to make this work.
    if len(num_neigh) != len(pos):
        tmp_num_neigh = np.zeros(len(pos), dtype=num_neigh.dtype)
        tmp_num_neigh[list(range(len(num_neigh)))] = num_neigh
        num_neigh = tmp_num_neigh

    return edge_index, shifts, cell, num_neigh