import os

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, TorsionFingerprints

import torch
import yaml, time
from collections import defaultdict
from diffusion.score_model import TensorProductScoreModel


def get_model(args):
    return TensorProductScoreModel(in_node_features=args.in_node_features, in_edge_features=args.in_edge_features,
                                   ns=args.ns, nv=args.nv, sigma_embed_dim=args.sigma_embed_dim,
                                   sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                                   num_conv_layers=args.num_conv_layers,
                                   max_radius=args.max_radius, radius_embed_dim=args.radius_embed_dim,
                                   scale_by_sigma=args.scale_by_sigma,
                                   use_second_order_repr=args.use_second_order_repr,
                                   residual=not args.no_residual, batch_norm=not args.no_batch_norm)


def get_optimizer_and_scheduler(args, model):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        raise NotImplementedError("Optimizer not implemented.")

    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7,
                                                               patience=args.scheduler_patience, min_lr=args.lr / 100)
    else:
        print('No scheduler')
        scheduler = None

    return optimizer, scheduler


def save_yaml_file(path, content):
    assert isinstance(path, str), f'path must be a string, got {path} which is a {type(path)}'
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)


class TimeProfiler:
    def __init__(self):
        self.times = defaultdict(float)
        self.starts = {}
        self.curr = None

    def start(self, tag):
        self.starts[tag] = time.time()

    def end(self, tag):
        self.times[tag] += time.time() - self.starts[tag]
        del self.starts[tag]

def get_conformer_energies(mol: Chem.Mol):
    """Returns a list of energies for each conformer in `mol`.
    """
    energies = []
    AllChem.MMFFSanitizeMolecule(mol)
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
    for conf in mol.GetConformers():
        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf.GetId())
        energy = ff.CalcEnergy()
        energies.append(energy)
    
    return np.asarray(energies, dtype=float)

def prune_conformers(mol, tfd_thresh):
    """Prunes all the conformers in the molecule.
    Removes conformers that have a TFD (torsional fingerprint deviation) lower than
    `tfd_thresh` with other conformers. Lowest energy conformers are kept.
    Parameters
    ----------
    mol : RDKit Mol
        The molecule to be pruned.
    tfd_thresh : float
        The minimum threshold for TFD between conformers.
    Returns
    -------
    mol : RDKit Mol
        The updated molecule after pruning.
    """
    if tfd_thresh < 0 or mol.GetNumConformers() <= 1:
        return mol

    energies = get_conformer_energies(mol)
    tfd = tfd_matrix(mol)
    sort = np.argsort(energies)  # sort by increasing energy
    keep = []  # always keep lowest-energy conformer
    discard = []

    for i in sort:
        this_tfd = tfd[i][np.asarray(keep, dtype=int)]
        # discard conformers within the tfd threshold
        if np.all(this_tfd >= tfd_thresh):
            keep.append(i)
        else:
            discard.append(i)

    # create a new molecule to hold the chosen conformers
    # this ensures proper conformer IDs and energy-based ordering
    new = Chem.Mol(mol)
    new.RemoveAllConformers()
    for i in keep:
        conf = mol.GetConformer(int(i))
        new.AddConformer(conf, assignId=True)

    return new

def tfd_matrix(mol: Chem.Mol) -> np.array:
    """Calculates the TFD matrix for all conformers in a molecule.
    """
    tfd = TorsionFingerprints.GetTFDMatrix(mol, useWeights=False)
    n = int(np.sqrt(len(tfd)*2))+1
    idx = np.tril_indices(n, k=-1, m=n)
    matrix = np.zeros((n,n))
    matrix[idx] = tfd
    matrix += np.transpose(matrix)
    return matrix