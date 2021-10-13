"""constructors of rotational-, translational- and permutational-equivariant RealNVP-based deterministic leapfrog integrators with support for both free and periodic boundary conditions"""
from typing import Sequence, Callable, Dict, Tuple, Optional, Any
import jax
import flax.linen as nn
import jax.numpy as jnp
from functools import partial
from jax import lax, ops, vmap, jit, grad, random

from jax.config import config

config.update("jax_enable_x64", True)

import jraph
from jax.ops import segment_sum
import jax.tree_util as tree
from jax_md import space
from flax.linen import Module
from aquaregia.utils import Graph
from aquaregia.nets import make_mlp, make_scalar_mlp, scalar_mlp_kernel_init

from aquaregia.utils import Array, ArrayTree
MLPFn = Callable[[ArrayTree], ArrayTree] # default MLP returned from a fn

"""
modules for 2D toys
"""
def get_2D_velocity_update(base_mlp, forward=True):

    def log_s_fn(x):
        base = base_mlp(x)
        return base, jnp.sum(base)

    def t_fn(x):
        base = base_mlp(x)
        return base

    if forward:
        update = lambda v, log_s, t : v * jnp.exp(log_s) + t
    else:
        update = lambda v, log_s, t : (v - t) * jnp.exp(-log_s)

    class VelocityUpdate(nn.Module):
        """
        velocity update goes like:

        log_s = mlp1(x)
        t = mlp2(x)

        v' = v * e^(log_s) + t # forward

        v = (v' - t) * e^(-log_s) # backward

        logdetJ = sum(log_s)
        """

        @nn.compact
        def __call__(self, x, v):
            log_s, logdetJ = log_s_fn(x)
            t = t_fn(x)
            out_v = update(v, log_s, t)
            return x, update(v, log_s, t), logdetJ

    return VelocityUpdate

def get_2D_position_update(base_mlp, forward=True):

    if forward:
        update = lambda x, t : x + t
    else:
        update = lambda x, t : x - t


    class PositionUpdate(nn.Module):
        """
        position update goes like:

        t = mlp(v)

        x' = x + t

        logdetJ = 0
        """

        @nn.compact
        def __call__(self, x, v):
            t = base_mlp(v)
            return update(x, t), v, 0.

    return PositionUpdate

"""
base RNVP module builder
"""
def get_RNVP_module(R_fwd,
                   V_fwd,
                   R_bkwd,
                   V_bkwd,
                   num_repeats = 1):

    class VRVForward(nn.Module):
        @nn.compact
        def __call__(self, x, v):
            x1, v1, logdetJ1 = V_fwd(name=f"V1")(x, v)
            x2, v2, logdetJ2 = R_fwd(name=f"R1")(x1, v1)
            x3, v3, logdetJ3 = V_fwd(name=f"V2")(x2, v2)

            return x3, v3, logdetJ1 + logdetJ2 + logdetJ3

    class VRVBackward(nn.Module):
        @nn.compact
        def __call__(self, x, v):
            x1, v1, logdetJ1 = V_bkwd(name=f"V2")(x, v)
            x2, v2, logdetJ2 = R_bkwd(name=f"R1")(x1, v1)
            x3, v3, logdetJ3 = V_bkwd(name=f"V1")(x2, v2)

            return x3, v3, logdetJ1 + logdetJ2 + logdetJ3

    class RNVPForward(nn.Module):
        @nn.compact
        def __call__(self, x, v):

            logdetJ = 0.
            for i in range(num_repeats):
                x, v, _logdetJ = VRVForward(name=f"VRV_{i}")(x, v)
                logdetJ = _logdetJ + logdetJ

            return x, v, logdetJ

    class RNVPBackward(nn.Module):
        @nn.compact
        def __call__(self, x, v):
            logdetJ = 0.
            for i in range(num_repeats)[::-1]:
                x, v, _logdetJ = VRVBackward(name=f"VRV_{i}")(x, v)
                logdetJ = _logdetJ + logdetJ

            return x, v, logdetJ

    return RNVPForward(), RNVPBackward()


"""
distance utilities
"""
def get_periodic_distance_calculator(metric,
                                     r_cutoff):
    d = vmap(vmap(metric, (None, 0)))
    def distances(xs, neighbor_list):
        mask = neighbor_list.idx != xs.shape[0]
        xs_neigh = xs[neighbor_list.idx]
        dr = d(xs, xs_neigh)
        return jnp.where(mask, dr, r_cutoff)
    return distances

def get_vacuum_distance_calculator(displacment_or_metric_fn):
    metric = space.canonicalize_displacement_or_metric(displacement_or_metric)
    return vmap(vmap(metric, in_axes = (None,0)), (0, None))


def polynomial_switching_fn(r : Array,
                            r_cutoff : float,
                            r_switch : float) -> float:
    x = (r - r_switch) / (r_cutoff - r_switch)
    switched_value = lax.cond(r < r_switch,
                              lambda _x: 1.,
                              lambda _x: 1. + (_x**3) * (-10. + _x * (15. - (6. * _x))),
                              x)
    return lax.cond(r > r_cutoff, lambda _x : 0., lambda _x: _x, switched_value)

def get_smooth_decay_rbfs(rbf_kwargs, r_cutoff, r_switch):
    from aquaregia.utils import radial_basis
    rbf = partial(radial_basis, **rbf_kwargs)
    polynomial_switch = partial(polynomial_switching_fn, r_cutoff = r_cutoff, r_switch = r_switch)
    def smooth_rbf(r):
        r = r[0]
        mod_r = polynomial_switch(r)
        return mod_r * rbf(r)
    return smooth_rbf

def get_smooth_decay_distance_fn(r_cutoff, r_switch):
    polynomial_switch = partial(polynomial_switching_fn, r_cutoff = r_cutoff, r_switch = r_switch)
    return lambda r : polynomial_switch(r)

def get_rbfs(rbf_kwargs, r_cutoff=None, r_switch=None):
    from aquaregia.utils import radial_basis
    rbf = partial(radial_basis, **rbf_kwargs)
    if (r_cutoff is None) or (r_switch is None):
        out_rbf = rbf
    else:
        polynomial_switch = partial(polynomial_switching_fn, r_cutoff = r_cutoff, r_switch = r_switch)
        def out_rbf(r):
            r = r[0]
            mod_r = polynomial_switch(r)
            return mod_r * rbf(r)
    return out_rbf

"""graph functionality"""
def get_En_graph_position_update(shift_fn,
                                 scalar_multiplier = 1.,
                                 forward=True):
    """return an `nn.Module` subclass that will return updated x, v, logdetJ=0"""
    vshift_fn = vmap(shift_fn, in_axes=(0,0))

    if forward:
        update = lambda x, t: vshift_fn(x, t)
    else:
        update = lambda x, t: vshift_fn(x, -t)


    class PositionUpdate(nn.Module):
        """
        """

        @nn.compact
        def __call__(self, x, v):
            t = make_scalar_mlp(num = x.shape[0], kernel_init = scalar_mlp_kernel_init(val=scalar_multiplier))(v)
            return update(x, t), v, 0.

    return PositionUpdate

def get_En_graph_velocity_update(message_fn, # input is xs, hs (edges are partialed)
                                 h_fn, # input is messages, hs
                                 log_s_fn, # input is new hs
                                 t_fn, # input is x, messages
                                 forward=True):

    if forward:
        update = lambda v, log_s, t : v * jnp.exp(log_s) + t
    else:
        update = lambda v, log_s, t : (v - t) * jnp.exp(-log_s)

    def log_s_and_t_fn(xs):
        masked_messages, masked_displacements, updated_neighbor_list = message_fn(xs)
        new_hs = h_fn(masked_messages)

        log_s, logdetJ = log_s_fn(new_hs)
        ts = t_fn(xs, masked_messages, masked_displacements)
        return log_s, ts, logdetJ


    class VelocityUpdate(nn.Module):
        """
        velocity update goes like:

        log_s = mlp1(x)
        t = mlp2(x)

        v' = v * e^(log_s) + t # forward

        v = (v' - t) * e^(-log_s) # backward

        logdetJ = sum(log_s)
        """

        @nn.compact
        def __call__(self, x, v):
            messages, distances, new_nbr_list = log_s_and_t_fn(x)
            log_s, t, logdetJ = log_s_and_t_fn(x)
            out_v = update(v, log_s, t)
            return x, out_v, logdetJ

    return VelocityUpdate


def get_stacked_mat_from_vector(neighbor_list, hs)-> Array:
    """
    utility to make a stacked vector (e.g. for hs stacking)

    Arguments:
    vec : Array(shape=N,M)

    Returns:
    Array : Array(shape=N,N,M)
    """

    num_attrs, attr_features = hs.shape
    tiled_vec = jnp.tile(hs, (num_attrs,1,1))
    swapped_tiled_vec = jnp.swapaxes(tiled_vec, 1,0)
    return jnp.concatenate([swapped_tiled_vec,tiled_vec], axis=-1)

def get_periodic_stacked_mat_from_vector(neighbor_list, hs):
    _nbrs = neighbor_list.idx
    N, max_nbrs = _nbrs.shape

    h_is = lax.map(lambda x: jnp.tile(x,(max_nbrs,1)), hs)
    h_js = hs[_nbrs]

    return jnp.concatenate([h_is, h_js], axis=-1)

def retrieve_periodic_edge_array(neighbor_list, edges):
    num_hs = edges.shape[0]
    masked_indices = jnp.where(neighbor_list.idx != num_hs, neighbor_list.idx, 0)
    return jnp.take_along_axis(edges, masked_indices, axis=1)



def make_message_fn(mlp_e : MLPFn, # mlp_e : R^{2*nf_h + 1 + 1} -> R^{nf_m}
                             neighbor_fn,
                             v_displacement_fn,
                             smooth_distance_featurizer,
                             hs,
                             edges) -> Callable[[Graph], ArrayTree]:
    """
    """
    hs_len, hs_feature_size = hs.shape
    periodic = False if neighbor_fn is None else True
    vacuum_mask = Array(jnp.ones((hs_len, hs_len)) - jnp.eye(hs_len), dtype=jnp.bool_)

    if edges is not None:
        assert edges.shape == (hs_len, hs_len)


    if periodic:
        pass_neighbor_fn = lambda x, y: neighbor_fn(x,y)
        stack_hs_to_nbr_matrix = partial(get_periodic_stacked_mat_from_vector, hs = hs) #stack hs
        if edges is None:
            get_message_edges = lambda neighbor_list : jnp.zeros_like(neighbor_list.idx)
        else:
            get_message_edges = partial(retrieve_periodic_edge_array, edges = edges)

        def get_mask(neighbor_list):
            num_particles, max_neighbors = neighbor_list.idx.shape
            mask = neighbor_list.idx != num_particles
            return mask

    else:
        pass_neighbor_fn = lambda x, y: None
        stack_hs_to_nbr_matrix = partial(get_stacked_mat_from_vector, hs = hs) #stack hs
        if edges is None:
            get_message_edges = lambda neighbor_list : jnp.zeros((hs_len, hs_len))
        else:
            get_message_edges = lambda neighbor_list : edges

        def get_mask(neighbor_list): return vacuum_mask



    def gn_message_fn(xs, neighbor_list):
        N, dim = xs.shape

        #WARNING : we might have to forego this fn and just throw out the result if there is a buffer overflow
        _nbrs = pass_neighbor_fn(x = xs, y = neighbor_list)
        mask = get_mask(_nbrs)

        displacements = v_displacement_fn(xs = xs, neighbor_list = _nbrs)
        masked_displacements = jnp.where(jnp.repeat(mask[..., jnp.newaxis], repeats = dim, axis=-1), displacements, 0.)
        distances_squared = (displacements**2).sum(axis=2)[..., jnp.newaxis]


        # compute radial basis features
#         rbs = jnp.apply_along_axis(smooth_distance_featurizer, axis=2, arr=distances)

        #compute message_edges
        message_edges = get_message_edges(neighbor_list = _nbrs)[..., jnp.newaxis] #add a new axis so we can pass to mpl_e


        # compute the unmasked messages (no edge information)
        in_hs = stack_hs_to_nbr_matrix(neighbor_list=_nbrs)
        unmasked_messages = mlp_e(in_hs, distances_squared, message_edges)
        out_masked_messages = jnp.where(jnp.repeat(mask[..., jnp.newaxis], repeats=unmasked_messages.shape[-1], axis=-1), unmasked_messages, 0.)

        return out_masked_messages, masked_displacements, _nbrs

    return gn_message_fn

def make_periodic_update_h_fn(mlp_h : MLPFn, # mlp_h : R^{nf_h + nf_m} -> R^{nf_h}
                            ) -> Callable[[ArrayTree, Array, Array], ArrayTree]:
    """function that returns Eq.6"""
    def aggregate_message_fn(messages): return jnp.mean(messages, axis=1)

    def gn_update_h_fn(messages, hs):
        aggregated_message_attributes = aggregate_message_fn(messages) #sum over the senders aggregated on receivers
        updated_hs = mlp_h(hs, aggregated_message_attributes) # pass hs and aggregated message attrs to the h-updating mlp
        return updated_hs

    return gn_update_h_fn

def make_periodic_log_s_fn(dimension, mlp_v, scalar_multiplier):
    """
    """

    def log_s_fn(hs):
        log_multipliers = mlp_v(hs) #compute log multipliers
        log_multipliers = make_scalar_mlp(num = hs.shape[0], kernel_init = scalar_mlp_kernel_init(val=scalar_multiplier))(log_multipliers) # multiply this by a scalar if we need
        tiled_multipliers = dimension * log_multipliers
        logdetJ = tiled_multipliers.sum() #this is a single summation
        return log_multipliers, logdetJ
    return log_s_fn

def get_periodic_v_displacement_fn(displacement_fn):
    return vmap(vmap(displacement_fn, in_axes=(None, 0)))

def get_vacuum_v_displacement_fn(displacement_fn):
    return vmap(vmap(displacement_fn, in_axes = (None,0)), (0, None))

def get_v_displacement_fn(displacement_fn, periodic):
    if periodic:
        vdisp = get_periodic_v_displacement_fn(displacement_fn)
        out_vdisp = lambda xs, neighbor_list : vdisp(xs, xs[neighbor_list.idx])
    else:
        vdisp = get_vacuum_v_displacement_fn(displacement_fn)
        out_vdisp = lambda xs, neighbor_list : vdisp(xs, xs)
    return out_vdisp

def make_periodic_t_fn(mlp_x,
                       C_offset,
                       scalar_multiplier):
    """we should be sure that the masked_messages are zero-masked"""

    def t_fn(xs,
             masked_messages,
             masked_displacements):
        num_positions, dimension = xs.shape

        masked_distances = jnp.sqrt((masked_displacements**2).sum(axis=2))[..., jnp.newaxis]
        offset_metrics = masked_distances + C_offset
        aug_normalized_x_displacements = masked_displacements / offset_metrics


        mlp_messages = mlp_x(masked_messages) #offset this?

        summands = aug_normalized_x_displacements * mlp_messages
        #summands = ops.index_update(summands, ops.index[jnp.diag_indices(max_neighbors)], jnp.zeros(dimension)) # we shouldn't need this now, right?
        updated_vectors = jnp.sum(summands, axis=1)
        return make_scalar_mlp(num = xs.shape[0], kernel_init = scalar_mlp_kernel_init(val=scalar_multiplier))(updated_vectors)

    return t_fn

"""
base graph class
"""
class GraphRNVP(object):
    def __init__(self,
                 hs,
                 edges,
                 mlp_e = make_mlp(features=[8,4,4], activation=nn.swish), # mlp for m_ij = m_ij(h_i, h_j, r_ij, edge_ij)
                 mlp_h = make_mlp(features = [8,4,4], activation=nn.swish), # mlp for h_i = h_i(h_i, m_i)
                 mlp_v = make_mlp(features = [8,4,1], activation=nn.swish), # mlp for log_s = log_s(h_i)
                 mlp_x = make_mlp(features = [8,4,1], activation=nn.swish), # mlp for t = t(m_ij)
                 C_offset = 1., # offset for t_fn
                 log_s_scalar = 1e-3,
                 t_scalar = 1e-3,
                 dt_scalar = 1e-3,
                 r_cutoff = None,
                 r_switch = None,
                 neighbor_fn = None,
                 box_vectors = None,
                 dimension = 3,
                 **kwargs
                 ):
        """"""
        # graph features
        self.hs = hs
        self.edges = edges
        self._check_hs_and_edges()

        self._dimension = dimension

        # mlps
        self._mlp_e = mlp_e
        self._mlp_h = mlp_h
        self._mlp_v = mlp_v
        self._mlp_x = mlp_x

        # scalars and offsets
        self._C_offset = C_offset
        self._log_s_scalar = log_s_scalar
        self._t_scalar = t_scalar
        self._dt_scalar = dt_scalar
        self._r_cutoff = r_cutoff
        self._r_switch = r_switch
        self._neighbor_fn = neighbor_fn
        self._box_vectors = box_vectors
        self._check_periodicity()

        #space attrs
        displacement_fn, shift_fn = self._get_space_attributes()
        self._displacement_fn = displacement_fn
        self._metric = space.canonicalize_displacement_or_metric(self._displacement_fn)
        self._shift_fn = shift_fn



    def _check_hs_and_edges(self):
        num_nodes, hs_feature_size = self.hs.shape
        assert len(self.edges.shape) == 2, f"the edges attr must be a 2D array"
        assert self.edges.shape == (num_nodes, num_nodes), f"the edge dimension is inconsistent with hs"

    def _check_periodicity(self):
        """simple function to handle periodic conditions"""
        if self._neighbor_fn is not None:
            self._periodic = True
        else:
            self._periodic = False

        if self._periodic:
            assert self._box_vectors is not None, f"passing a `neighbor_fn` implies periodicity, but `box_vectors` were supplied as `None`"
            assert self._r_cutoff is not None
            assert self._dimension == len(self._box_vectors)
            assert jnp.all(0.5 * self._box_vectors > self._r_cutoff), f"cutoff distance cannot be greater than half the periodic box size"
        else:
            assert self._r_cutoff is None, f"vacuum simulations necessitate no cutoffs"

    def _get_space_attributes(self):
        if self._periodic:
            return space.periodic(self._box_vectors)
        else:
            return space.free()

    def _get_message_fn(self, base_neighbor_list):
        self._set_v_displacement_fn(neighbor_list=base_neighbor_list)
        message_fn = make_message_fn(mlp_e = self._mlp_e, # mlp_e : R^{2*nf_h + 1 + 1} -> R^{nf_m}
                                     neighbor_fn = self._neighbor_fn,
                                     v_displacement_fn = self._v_displacement_fn,
                                     smooth_distance_featurizer = None, # NOTE : we might want to smooth this in the future
                                     hs = self.hs,
                                     edges = self.edges)
        return partial(message_fn, neighbor_list = base_neighbor_list)

    def _get_hs_fn(self):
        return partial(make_periodic_update_h_fn(self._mlp_h), hs = self.hs)

    def _get_log_s_fn(self):
        return make_periodic_log_s_fn(dimension = self._dimension,
                                  mlp_v = self._mlp_v,
                                  scalar_multiplier = self._log_s_scalar)

    def _set_v_displacement_fn(self, neighbor_list):
        self._v_displacement_fn = get_v_displacement_fn(displacement_fn=self._displacement_fn, periodic=self._periodic)


    def _get_t_fn(self, base_neighbor_list):
        return make_periodic_t_fn(mlp_x = self._mlp_x,
                                  C_offset = self._C_offset,
                                  scalar_multiplier = self._t_scalar)

    def _get_R_module(self,
                      forward=True):
        return get_En_graph_position_update(shift_fn = self._shift_fn,
                             scalar_multiplier = self._dt_scalar,
                             forward=forward)

    def _get_V_module(self,
                      base_neighbor_list,
                      forward=True):
        return get_En_graph_velocity_update(message_fn = self._get_message_fn(base_neighbor_list = base_neighbor_list), # input is xs, hs (edges are partialed)
                             h_fn = self._get_hs_fn(), # input is messages, hs
                             log_s_fn = self._get_log_s_fn(), # input is new hs
                             t_fn = self._get_t_fn(base_neighbor_list=base_neighbor_list), # input is x, messages
                             forward=forward)
    def rnvp_modules(self, xs, num_repeats = 1):
        forward_x_update_module = self._get_R_module(forward=True)
        backward_x_update_module = self._get_R_module(forward=False)

        if self._periodic:
            base_neighbor_list = self._neighbor_fn(xs)
        else:
            base_neighbor_list = None

        forward_v_update_module = self._get_V_module(base_neighbor_list = base_neighbor_list, forward=True)
        backward_v_update_module = self._get_V_module(base_neighbor_list = base_neighbor_list, forward=False)

        forward_rnvp, backward_rnvp = get_RNVP_module(R_fwd = forward_x_update_module,
                                                  V_fwd = forward_v_update_module,
                                                  R_bkwd = backward_x_update_module,
                                                  V_bkwd = backward_v_update_module,
                                                  num_repeats = num_repeats)

        return forward_rnvp, backward_rnvp, base_neighbor_list
