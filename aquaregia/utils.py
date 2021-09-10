"""utilities"""
from typing import Sequence, Callable, Dict, Tuple, Optional, NamedTuple
import jax
import flax.linen as nn
import jax.numpy as jnp
from functools import partial
from jax import lax, ops, vmap, jit, grad, random
from jax.scipy.special import logsumexp

from jax.config import config
config.update("jax_enable_x64", True)

# Typing
Array = jnp.array
from jraph._src.utils import ArrayTree
EnergyFn = Callable[[Array, ArrayTree, ...], float]

# E(n)-Equivariant Graph Fns
class Graph(NamedTuple): #a NamedTuple object for the graph node (it carries latent features hs, positions xs, and velocities vs)
    """
    given `N` particles with dimension `dim`, `h_features`, `edge_features`
    """
    hs : ArrayTree # shape=(N, h_features)
    xs : Array # shape=(N, dim)
    vs : Array # shape=(N, dim)
    edges : Array # shape=(N,N,edge_features)

def kinetic_energy(V: Array, # velocities in nm ps**{-1}
                   masses: Array # in daltons)
                   )-> float: # in kJ mol**(-1)
    return 0.5 * V.dot(V) / masses

def logZ_from_works(works : Array # reduced works
                   ) -> float: # reduced free energy
    """compute the free energy from a work array in nats"""
    N = len(works)
    w_min = jnp.min(works)
    return -(w_min - logsumexp(-works + w_min) + jnp.log(N))

def ESS(works : Array # reduced works
       ) -> float: # ESS quantity
    """compute effective sample size"""
    log_weights = -works
    Ws = jnp.exp(log_weights - logsumexp(log_weights))
    ESS = 1. / jnp.sum(Ws**2) / len(works)
    return ESS

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
