"""utilities"""
from typing import Sequence, Callable, Dict, Tuple, Optional, NamedTuple, Union, Any, Mapping, Iterable
import jax
# import flax.linen as nn
import jax.numpy as jnp
from functools import partial
from jax import lax, ops, vmap, jit, grad, random
from jax.scipy.special import logsumexp
import numpy as np
from jax_md.partition import NeighborList

from jax.config import config
config.update("jax_enable_x64", True)

# Typing
Array = jnp.array # get arraytree
ArrayTree = Union[jnp.ndarray, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]
EnergyFn = Callable[[Array, ...], float]

# E(n)-Equivariant Graph Fns
class Graph(NamedTuple): #a NamedTuple object for the graph node (it carries latent features hs, positions xs, and velocities vs)
    """
    given `N` particles with dimension `dim`, `h_features`, `edge_features`
    """
    hs : ArrayTree # shape=(N, h_features)
    xs : Array # shape=(N, dim)
    vs : Array # shape=(N, dim)
    edges : Array # shape=(N,N,edge_features)

def get_vacuum_neighbor_list(num_particles : int) -> NeighborList:
    from jax_md.partition import NeighborListFormat
    vacuum_neighbor_list_idx = jnp.transpose(jnp.repeat(jnp.arange(num_particles, dtype=jnp.int64)[..., jnp.newaxis], repeats = num_particles, axis=-1))
    back_diag = jnp.diag(num_particles - jnp.diag(vacuum_neighbor_list_idx))
    vacuum_neighbor_list = NeighborList(idx = vacuum_neighbor_list_idx + back_diag,
                                        reference_position=None,
                                        did_buffer_overflow=False,
                                        max_occupancy=0.,
                                        format = NeighborListFormat.Dense,
                                        update_fn = None,
                                        cell_list_capacity=None) #why we need this arg?
    return vacuum_neighbor_list


def kinetic_energy(vs, mass):
    def ke(_v, _mass):
        return 0.5 * _v.dot(_v) * _mass
    return vmap(ke, in_axes=(0,0))(vs, mass).sum()

def normal_kinetic_energy(vs):
    def ke(vs): #this should be a split
        return 0.5 * vs.dot(vs)
    vke = vmap(ke)
    return vke(vs).sum()

def logZ_from_works(works : Array # reduced works
                   ) -> float: # reduced free energy
    """compute the free energy from a work array in nats"""
    N = len(works)
    w_min = jnp.min(works)
    return -(w_min - logsumexp(-works + w_min) + jnp.log(N))

def Bennet_implicit_fn(dF, fwd_works, bkwd_works):
    forward_term = fwd_works - dF
    backward_term = bkwd_works + dF
    return (forward_term - backward_term).sum()

def Bennet_solution(fwd_reduced_works, bkwd_reduced_works):
    """use Bennet's Acceptance Ratio to compute the solution of two works"""
    from scipy.optimize import fsolve
    from aquaregia.utils import logZ_from_works

    p_fn = partial(Bennet_implicit_fn, fwd_works = fwd_reduced_works, bkwd_works = bkwd_reduced_works)
    init_guess = -logZ_from_works(fwd_reduced_works)

    a = fsolve(p_fn, init_guess)
    return a[0]

def ESS(works : Array # reduced works
       ) -> float: # ESS quantity
    """compute effective sample size"""
    log_weights = -works
    Ws = jnp.exp(log_weights - logsumexp(log_weights))
    ESS = 1. / jnp.sum(Ws**2) / len(works)
    return ESS

def weights_from_works(works):
    min_works = jnp.min(works)
    log_denom = min_works + logsumexp(-works  - min_works)
    return jnp.exp(-works - log_denom)

def make_test_graph(seed=random.PRNGKey(13), num_nodes=10, h_features=5, dimensions=3, edge_features=4, hs = None):
    """deprecated"""
    xseed, vseed, nodes_seed, edges_seed = random.split(seed, num=4) #split the seed for xseed and vseed
    n_graphs=1 #we only batch a single graph...batching dimension is nodes!
    if hs is None:
        hs = random.normal(nodes_seed, shape=(num_nodes, h_features))
    xs = random.normal(xseed, shape=(num_nodes, dimensions))
    vs = random.normal(xseed, shape=(num_nodes, dimensions))
    edges = random.normal(edges_seed, shape=((num_nodes * (num_nodes - 1)) * n_graphs, edge_features))

    g = get_augmented_fully_connected_graph(n_node_per_graph=num_nodes,
                                    node_features=ECGNode(hs = hs,
                                                          xs = xs,
                                                          vs = vs),
                                    edges = edges,
                                    global_features=None)
    return g

def get_normal_test_graph(seed = random.PRNGKey(13),
                          num_nodes = 10,
                          dimension = 3,
                          hs_features = 5,
                          edges_features = 2):
    hs_seed, xs_seed, vs_seed, edges_seed = random.split(seed, num = 4)
    g = Graph(hs=random.normal(hs_seed, shape=(num_nodes, hs_features)),
              xs = random.normal(xs_seed, shape=(num_nodes, dimension)),
              vs = random.normal(vs_seed, shape=(num_nodes, dimension)),
              edges = random.normal(edges_seed, shape=(num_nodes, num_nodes, edges_features)))
    return g

def radial_basis(d_ij : float,
                 mu_ks : Array,
                 gamma : float) -> Array:
    """
    turn a radial distance `d_ij` into a radial basis function with `mu_ks` radial bases
    and a spread of `gamma`

    Arguments:
        d_ij : float
            radial distance
        mu_ks : Array
            radial bases of shape (Q,) where Q is the number of radial bases
        gamma : float
            spread of radial basis interaction
    """
    return jnp.exp(-gamma * (d_ij - mu_ks)**2)

def polynomial_switching_fn(r : Array, r_cutoff : float, r_switch : float) -> Array:
    x = (r - r_switch) / (r_cutoff - r_switch) # compute argument to polynomial
    mults = jnp.where(jnp.logical_and(r > r_switch, r <= r_cutoff), 1. + (x**3) * (-10. + x * (15. - (6. * x))), 1.)
    final_mults = jnp.where(r > r_cutoff, 0., mults)
    return final_mults

def get_periodic_distance_calculator(metric,
                                     r_cutoff):
    d = vmap(vmap(metric, (None, 0)))
    def distances(xs, neighbor_list):
        mask = neighbor_list.idx != xs.shape[0]
        xs_neigh = xs[neighbor_list.idx]
        dr = d(xs, xs_neigh)
        return jnp.where(mask, dr, r_cutoff)
    return distances

def get_mask(neighbor_list):
    num_particles, max_neighbors = neighbor_list.idx.shape
    mask = neighbor_list.idx != num_particles
    return mask

def rotation_matrix(axis, theta):
    import scipy
    return scipy.linalg.expm(np.cross(np.eye(3), axis * theta))

def random_rotation_matrix(numpy_random_state, epsilon=1e-6):
    """
    Generates a random 3D rotation matrix from axis and angle.
    Args:
        numpy_random_state: numpy random state object
    Returns:
        Random rotation matrix.
    """
    rng = numpy_random_state
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis) + epsilon
    theta = 2 * np.pi * rng.uniform(0.0, 1.0)
    return rotation_matrix(axis, theta)

def compute_atom_centered_fingerprints(mol,
                                       generator,
                                       fpSize,
                                       normalize = True):
    """
    compute an atom-centric fingerprint of a molecule. You need `rdkit`

    Arguments:
    mol : rdkit.Chem.Mol
    generator : return of `rdFingerprintGenerator.GetCountFingerPrint
    fpSize : size of fingerprint
    normalize : reduce so that all output vals are <= 1.

    Return:
    fingerprints : np.array(mol.GetNumAtoms(), fpSize, dtype=np.float64)

    TODO : fix thee typing (do we need to import `rdkit` here?)

    Example:
    >>> import rdkit
    >>> import numpy as np
    >>> #print(rdkit.__version__)
    >>> from rdkit import Chem
    >>> from rdkit.Chem import RDKFingerprint
    >>> from rdkit.Chem import rdFingerprintGenerator
    >>> mol = Chem.SDMolSupplier('mol.sdf', removeHs=False)[0] # this assumes you want the 0th mol from an sdf called `mol.sdf`
    >>> fpSize = 32
    >>> generator = rdFingerprintGenerator.GetRDKitFPGenerator(minPath=5, maxPath=5, fpSize=fpSize)
    >>> X = compute_atom_centered_fingerprints(mol, generator, fpSize, normalize=True)
    """
    n_atoms = mol.GetNumAtoms()
    fingerprints = np.zeros((n_atoms, fpSize), dtype=int)

    for i in range(mol.GetNumAtoms()):
        fingerprint = generator.GetCountFingerprint(mol, fromAtoms=[i])
        for (key, val) in fingerprint.GetNonzeroElements().items():
            fingerprints[i, key] = val

    fp = np.array(fingerprints, dtype=np.float64)
    if normalize:
        _max = np.max(fp)
        fp = fp / _max

    return fp

def generate_edges_from_mol(mol, normalize=True):
    """
    generate an array of edge features (of 2 features);
    the features we use are
    `bond_type` (float) and `is_conjugated` (bool as binary float);
    nonbonded atoms are zeros.

    Arguments:
    mol : rdkit.Chem.Mol
    normalize : reduce so that all output vals are <= 1.

    Return:
    out_array : np.array of shape (mol.GetNumAtoms, mol.GetNumAtoms, 2)

    NOTE : this has not been stress tested on big molecules/protein sequences
    """
    num_atoms = mol.GetNumAtoms()
    edge_array = np.zeros((num_atoms, num_atoms, 2))
    for bond in mol.GetBonds():
        p1, p2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        conj_bool = 1. if bond.GetIsConjugated() else 0.
        lof = np.array([bond_type, conj_bool])
        edge_array[p1, p2] = lof
        edge_array[p2, p1] = lof

    if normalize:
        max_val = np.max(edge_array)
        out_array = edge_array / max_val
    else:
        out_array = edge_array

    return out_array
