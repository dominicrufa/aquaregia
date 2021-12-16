"""constructors of rotational-, translational- and permutational-equivariant RealNVP-based deterministic leapfrog integrators with support for both free and periodic boundary conditions"""
from jax.config import config
config.update("jax_enable_x64", True)
from typing import Sequence, Callable, Dict, Tuple, Optional, Any
import jax
import flax.linen as nn
import jax.numpy as jnp
from functools import partial
from jax import lax, ops, vmap, jit, grad, random, value_and_grad
from jax.experimental import optimizers

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

    class RVForward(nn.Module):
        @nn.compact
        def __call__(self, x, v):
            x1, v1, logdetJ1 = R_fwd(name=f"R1")(x, v)
            x2, v2, logdetJ2 = V_fwd(name=f"V1")(x1, v1)
            return x2, v2, logdetJ1 + logdetJ2

    class RVBackward(nn.Module):
        @nn.compact
        def __call__(self, x, v):
            x1, v1, logdetJ1 = R_bkwd(name=f"R1")(x, v)
            x2, v2, logdetJ2 = V_bkwd(name=f"V1")(x1, v1)
            return x2, v2, logdetJ1 + logdetJ2

    def rnvp_scanner_fn(carrier, t):
        x, v, stack_module = carrier
        out_x, out_v, _out_logdetJ = stack_module(name=f"RV_{t}")(x, v)
        return (out_x, out_v, stack_module), (out_x, out_v, _out_logdetJ)

    class _RNVPForwardScan(nn.Module):
        @nn.compact
        def __call__(self, x, v):

            #initialize with V module so that it is symplectic
            x0, v0, logdetJ0 = V_fwd(name=f"V0")(x, v)
            scan_init_carrier = (x0, v0, RVForward)
            (out_x, out_v, _), (stack_xs, stack_vs, stack_out_logdetJs) = nn.scan(f = rnvp_scanner_fn,
                                                                                   init = scan_init_carrier,
                                                                                   xs = jnp.arange(num_repeats, dtype = jnp.int64)
                                                                                   )
            # for the purpose of returning trajectories
            #out_x_stack = jnp.vstack((x[jnp.newaxis, ...], x0[jnp.newaxis, ...], stack_xs))
            #out_v_stack = jnp.vstack((v[jnp.newaxis, ...], v0[jnp.newaxis, ...], stack_vs))
            out_logdetJ_stack = jnp.cumsum(jnp.concatenate((Array([0., logdetJ0]), stack_out_logdetJs)))
            #return out_x_stack, out_v_stack, out_logdetJ_stack
            return out_x, out_v, out_logdetJ_stack[-1]

    class _RNVPBackwardScan(nn.Module):
        @nn.compact
        def __call__(self, x, v):

            #initialize with V module so that it is symplectic
            x0, v0, logdetJ0 = V_fwd(name=f"V0")(x, v)
            scan_init_carrier = (x0, v0, RVBackward)
            (out_x, out_v, _), (stack_xs, stack_vs, stack_out_logdetJs) = nn.scan(f = rnvp_scanner_fn,
                                                                                   init = scan_init_carrier,
                                                                                   xs = jnp.arange(num_repeats, dtype = jnp.int64)
                                                                                   )
            # for the purpose of returning trajectories
            #out_x_stack = jnp.vstack((x[jnp.newaxis, ...], x0[jnp.newaxis, ...], stack_xs))
            #out_v_stack = jnp.vstack((v[jnp.newaxis, ...], v0[jnp.newaxis, ...], stack_vs))
            out_logdetJ_stack = jnp.cumsum(jnp.concatenate((Array([0., logdetJ0]), stack_out_logdetJs)))
            #return out_x_stack, out_v_stack, out_logdetJ_stack
            return out_x, out_v, out_logdetJ_stack[-1]

    class RNVPForwardScan(nn.Module):
        @nn.compact
        def __call__(self, x, v):
            x, v, logdetJ = V_fwd(name=f"V_0")(x, v)
            for i in range(1,num_repeats*2,2):
                x, v, _logdetJ_R = R_fwd(name=f"R_{i}")(x, v)
                x, v, _logdetJ_V = V_fwd(name=f"V_{i+1}")(x, v)
                logdetJ = logdetJ + _logdetJ_R + _logdetJ_V
            return x, v, logdetJ

    class RNVPBackwardScan(nn.Module):
        @nn.compact
        def __call__(self, x, v):
            x, v, logdetJ = V_bkwd(name=f"V_{num_repeats*2}")(x, v)
            for i in range(1,num_repeats*2, 2)[::-1]:
                x, v, _logdetJ_R = R_bkwd(name=f"R_{i}")(x, v)
                x, v, _logdetJ_V = V_bkwd(name=f"V_{i-1}")(x, v)
                logdetJ = logdetJ + _logdetJ_R + _logdetJ_V
            return x, v, logdetJ

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

    #return RNVPForward, RNVPBackward
    return RNVPForwardScan, RNVPBackwardScan

def get_generic_EnGNNModule(hs,
                            num_repeats,
                            message_fn,
                            hs_update_fn,
                            h_retrieval_mlp,
                            return_aggregator):

    start_hs = hs

    class GenericEnGNNModule(nn.Module):
        @nn.compact
        def __call__(self, x):
            hs = start_hs
            for i in range(num_repeats): #for through the sequential edge/h update/aggregation (respectively)
                masked_messages, _, _ = message_fn(xs = x, hs = hs)
                hs = hs_update_fn(messages = masked_messages, hs = hs)
            out_features = h_retrieval_mlp(hs)
            return return_aggregator(out_features)

    return GenericEnGNNModule


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

def get_mask(neighbor_list):
    num_particles, max_neighbors = neighbor_list.idx.shape
    mask = neighbor_list.idx != num_particles
    return mask

def make_message_fn(mlp_e : MLPFn, # mlp_e : R^{2*nf_h + 1 + 1} -> R^{nf_m}
                             neighbor_fn,
                             v_displacement_fn,
                             smooth_distance_featurizer) -> Callable[[Graph], ArrayTree]:
    """
    """
    # hs_len, hs_feature_size = hs.shape
    get_message_edges = retrieve_periodic_edge_array
    # if edges is not None:
    #     # assert edges.shape == (hs_len, hs_len)
    #     # get_message_edges = partial(retrieve_periodic_edge_array, edges = edges)
    #     #get_message_edges = retrieve_periodic_edge_array
    # else:
    #     get_message_edges = lambda neighbor_list, edges : jnp.zeros_like(neighbor_list.idx)

    # stack_hs_to_nbr_matrix = partial(get_periodic_stacked_mat_from_vector, hs = hs) #stack hs
    stack_hs_to_nbr_matrix = get_periodic_stacked_mat_from_vector

    def gn_message_fn(xs, neighbor_list, hs, edges):
        N, dim = xs.shape

        #WARNING : we might have to forego this fn and just throw out the result if there is a buffer overflow
        _nbrs = neighbor_fn(xs, neighbor_list)
        mask = get_mask(_nbrs)
        #print(f"gm_message_fn: xs: {xs}; neighbor_list: {_nbrs}")
        displacements = v_displacement_fn(xs = xs, neighbor_list = _nbrs)
        masked_displacements = jnp.where(jnp.repeat(mask[..., jnp.newaxis], repeats = dim, axis=-1), displacements, 0.)
        distances_squared = (displacements**2).sum(axis=2)[..., jnp.newaxis]

        # compute radial basis features
#         rbs = jnp.apply_along_axis(smooth_distance_featurizer, axis=2, arr=distances)

        #compute message_edges
        message_edges = get_message_edges(neighbor_list = _nbrs, edges = edges)[..., jnp.newaxis] #add a new axis so we can pass to mpl_e

        # compute the unmasked messages (no edge information)
        in_hs = stack_hs_to_nbr_matrix(neighbor_list=_nbrs, hs = hs)
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

# scalar_mlp_delimiter
def scalar_mlp_delimiter(scalar_multiplier, num_scalars, arg_inputs):
    # this doesnt work because you can't mix jax transforms with linen (can't put a lax.cond statement in a flax.linen module, see https://flax.readthedocs.io/en/latest/flax.errors.html#flax.errors.JaxTransformError)
    return lax.cond(scalar_multiplier is None,
                    lambda _x : _x,
                    lambda _x : make_scalar_mlp(num = num_scalars,
                                                kernel_init = scalar_mlp_kernel_init(val=scalar_multiplier))(arg_inputs),
                    arg_inputs)

def make_periodic_log_s_fn(dimension,
                           mlp_v,
                           scalar_multiplier):
    """
    """
    if scalar_multiplier is None:
        def scalar_delimiter(scalar_multiplier, num_scalars, arg_inputs) : return arg_inputs
    else:
        def scalar_delimiter(scalar_multiplier, num_scalars, arg_inputs) : return make_scalar_mlp(num = num_scalars, kernel_init = scalar_mlp_kernel_init(val=scalar_multiplier))(arg_inputs)

    def log_s_fn(hs):
        log_multipliers = mlp_v(hs) #compute log multipliers
        log_multipliers = scalar_delimiter(scalar_multiplier = scalar_multiplier,
                                           num_scalars = hs.shape[0],
                                           arg_inputs = log_multipliers)
        #log_multipliers = make_scalar_mlp(num = hs.shape[0], kernel_init = scalar_mlp_kernel_init(val=scalar_multiplier))(log_multipliers) # multiply this by a scalar if we need
        tiled_multipliers = dimension * log_multipliers
        logdetJ = tiled_multipliers.sum() #this is a single summation
        return log_multipliers, logdetJ
    return log_s_fn

def get_periodic_v_displacement_fn(displacement_fn):
    return vmap(vmap(displacement_fn, in_axes=(None, 0)))

def get_v_displacement_fn(displacement_fn):
    vdisp = get_periodic_v_displacement_fn(displacement_fn)
    out_vdisp = lambda xs, neighbor_list : vdisp(xs, xs[neighbor_list.idx])
    return out_vdisp

def make_periodic_t_fn(mlp_x,
                       C_offset,
                       scalar_multiplier):
    """we should be sure that the masked_messages are zero-masked"""

    if scalar_multiplier is None:
        def scalar_delimiter(scalar_multiplier, num_scalars, arg_inputs) : return arg_inputs
    else:
        def scalar_delimiter(scalar_multiplier, num_scalars, arg_inputs) : return make_scalar_mlp(num = num_scalars, kernel_init = scalar_mlp_kernel_init(val=scalar_multiplier))(arg_inputs)

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
        out_val = updated_vectors
        # out_val = scalar_delimiter(scalar_multiplier = scalar_multiplier,
        #                            num_scalars = xs.shape[0],
        #                            arg_inputs = updated_vectors)
        #return make_scalar_mlp(num = xs.shape[0], kernel_init = scalar_mlp_kernel_init(val=scalar_multiplier))(updated_vectors)
        return out_val

    return t_fn

"""
base graph class
"""
class EnGNN(object):
    """
    Generic EnGNN returnbale
    """
    def __init__(self,
                 hs,
                 edges,
                 mlp_e = make_mlp(features=[8,4,4], activation=nn.swish), # mlp for m_ij = m_ij(h_i, h_j, r_ij, edge_ij),
                 mlp_h = make_mlp(features = [8,4,4], activation=nn.swish), # mlp for h_i = h_i(h_i, m_i)
                 r_cutoff = None,
                 r_switch = None,
                 allocate_neighbor_fn = None,
                 neighbor_fn = None, # update
                 box_vectors = None,
                 dimension = 3,
                 **kwargs):
        # graph features
        self.hs = hs
        self.edges = edges
        self._check_hs_and_edges()

        self._dimension = dimension

        # mlps
        self._mlp_e = mlp_e
        self._mlp_h = mlp_h

        self._r_cutoff = r_cutoff
        self._r_switch = r_switch
        self._allocate_neighbor_fn = allocate_neighbor_fn
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
            from aquaregia.utils import get_vacuum_neighbor_list
            num_particles = self.hs.shape[0]
            vacuum_neighbor_list = get_vacuum_neighbor_list(num_particles)
            self._neighbor_fn = lambda x, y: vacuum_neighbor_list

    def _get_space_attributes(self):
        if self._periodic:
            return space.periodic(self._box_vectors)
        else:
            return space.free()

    def _set_v_displacement_fn(self, neighbor_list):
        self._v_displacement_fn = get_v_displacement_fn(displacement_fn=self._displacement_fn)

    def _get_message_fn(self, base_neighbor_list):
        self._set_v_displacement_fn(neighbor_list=base_neighbor_list)
        message_fn = make_message_fn(mlp_e = self._mlp_e, # mlp_e : R^{2*nf_h + 1 + 1} -> R^{nf_m}
                                     neighbor_fn = self._neighbor_fn,
                                     v_displacement_fn = self._v_displacement_fn,
                                     smooth_distance_featurizer = None) # NOTE : we might want to smooth this in the future
        return partial(message_fn, neighbor_list = base_neighbor_list, edges = self.edges)

    def _get_hs_fn(self):
        return make_periodic_update_h_fn(self._mlp_h)

    def EnGNN_module(self,
                     xs,
                     num_repeats,
                     h_retrieval_mlp = lambda _x : jnp.mean(_x, axis=0),
                     return_aggregator = make_mlp(features=[4,1], activation=nn.swish)
                     ):

        if self._periodic:
            base_neighbor_list = self._allocate_neighbor_fn(xs) #this will generate the first neighbor list.
        else:
            base_neighbor_list = None

        # retrieve message and hs-update fn
        message_fn = self._get_message_fn(base_neighbor_list)
        hs_update_fn = self._get_hs_fn()
        module = get_generic_EnGNNModule(hs = self.hs,
                                         num_repeats = num_repeats,
                                         message_fn = message_fn,
                                         hs_update_fn = hs_update_fn,
                                         h_retrieval_mlp = h_retrieval_mlp,
                                         return_aggregator = return_aggregator)
        return module


class GraphRNVP(EnGNN):
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
        super().__init__(hs,
                         edges = edges,
                         mlp_e = mlp_e,
                         mlp_h = mlp_h,
                         r_cutoff = r_cutoff,
                         r_switch = r_switch,
                         neighbor_fn = neighbor_fn,
                         box_vectors = box_vectors,
                         dimension = dimension,
                         **kwargs)

        # mlps
        self._mlp_v = mlp_v
        self._mlp_x = mlp_x

        # scalars and offsets
        self._C_offset = C_offset
        self._log_s_scalar = log_s_scalar
        self._t_scalar = t_scalar
        self._dt_scalar = dt_scalar

    def _get_message_fn(self, base_neighbor_list):
        self._set_v_displacement_fn(neighbor_list=base_neighbor_list)
        message_fn = make_message_fn(mlp_e = self._mlp_e, # mlp_e : R^{2*nf_h + 1 + 1} -> R^{nf_m}
                                     neighbor_fn = self._neighbor_fn,
                                     v_displacement_fn = self._v_displacement_fn,
                                     smooth_distance_featurizer = None) # NOTE : we might want to smooth this in the future
        return partial(message_fn, neighbor_list = base_neighbor_list, hs = self.hs, edges = self.edges)

    def _get_hs_fn(self):
        return partial(make_periodic_update_h_fn(self._mlp_h), hs = self.hs)

    def _get_log_s_fn(self):
        return make_periodic_log_s_fn(dimension = self._dimension,
                                  mlp_v = self._mlp_v,
                                  scalar_multiplier = self._log_s_scalar)


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
    def rnvp_modules(self,
                     xs,
                     num_repeats = 1,
                     bind=True # whether to wrap the rnvps with a "()"; False if the rnvp modules will pass to another compact module.
                     ):
        forward_x_update_module = self._get_R_module(forward=True)
        backward_x_update_module = self._get_R_module(forward=False)

        if self._periodic:
            base_neighbor_list = self._allocate_neighbor_fn(xs)
        else:
            base_neighbor_list = None

        forward_v_update_module = self._get_V_module(base_neighbor_list = base_neighbor_list, forward=True)
        backward_v_update_module = self._get_V_module(base_neighbor_list = base_neighbor_list, forward=False)

        forward_rnvp, backward_rnvp = get_RNVP_module(R_fwd = forward_x_update_module,
                                                  V_fwd = forward_v_update_module,
                                                  R_bkwd = backward_x_update_module,
                                                  V_bkwd = backward_v_update_module,
                                                  num_repeats = num_repeats)
        if bind:
            out_forward_rnvp = forward_rnvp()
            out_backward_rnvp = backward_rnvp()
        else:
            out_forward_rnvp = forward_rnvp
            out_backward_rnvp = backward_rnvp
        return out_forward_rnvp, out_backward_rnvp, base_neighbor_list


# training utils
def forward_work_fn(params,
                    in_xs,
                    in_vs,
                    u_i_fn,
                    u_f_fn,
                    ke_fn,
                    initd_forward_rnvp,
                    kT):
    out_xs, out_vs, logdetJ = initd_forward_rnvp.apply(params, in_xs, in_vs)
    delta_ke = ke_fn(out_vs) - ke_fn(in_vs)
    delta_u = (u_f_fn(out_xs) - u_i_fn(in_xs)) / kT
    return delta_u + delta_ke - logdetJ


def backward_work_fn(params, in_xs, in_vs, u_i_fn, u_f_fn, ke_fn, initd_backward_rnvp, kT):
    out_xs, out_vs, logdetJ = initd_backward_rnvp.apply(params, in_xs, in_vs)
    delta_ke = ke_fn(out_vs) - ke_fn(in_vs)
    delta_u = (u_i_fn(out_xs) - u_f_fn(in_xs)) / kT
    return delta_u + delta_ke + logdetJ

# compute the energy of the forward/backward (A', B') distributions (for i.e. computing forces)
def u_forward_rnvp(params, xs, vs, initd_backward_rnvp, u_i_fn, ke_fn, kT):
    """compute the energy of the log-push-forward probability A'"""
    back_xs, back_vs, logdetJ = initd_backward_rnvp.apply(params, xs, vs) #first push the xs, vs, backward
    u_back, ke_back = u_i_fn(back_xs) / kT, ke_fn(back_vs) #compute the energy at the push_back start potential
    return u_back + ke_back + logdetJ # compute the energy then (e^{-u_i(x) - logdetJ_M(x)} = e^{-(u_i(x) + logdetJ_M(x))})

def u_backward_rnvp(params, xs, vs, initd_forward_rnvp, u_f_fn, ke_fn, kT):
    """compute the energy of the log-push-backward probability B'"""
    forward_xs, forward_vs, logdetJ = initd_forward_rnvp.apply(params, xs, vs) #first push the xs, vs forward
    u_forward, ke_forward = u_f_fn(forward_xs) / kT, ke_fn(forward_vs)
    return u_forward - logdetJ


def get_pull_samples_fn(p_i_samples,
                        p_f_samples,
                        minibatch_size, #if this is None, will train on the full batch (might get expensive depending on size of batch)
                        ke_sampler
                        ):
    full_batch = True if minibatch_size is None else False
    bidirectional = False if p_f_samples is None else True

    num_itrain_samples = p_i_samples.shape[0]
    if bidirectional:
        num_ftrain_samples = p_f_samples.shape[0]
    else:
        num_ftrain_samples = num_itrain_samples

    if full_batch:
        def pull_samples(seed):
            viseed, vfseed = random.split(seed)
            vis = vmap(ke_sampler)(random.split(viseed, num=len(p_i_samples)))
            vfs = vmap(ke_sampler)(random.split(vfseed, num=len(p_f_samples)))
            return p_i_samples, vis, p_f_samples, vfs
    else:
        def pull_samples(seed):
            iseed, fseed, viseed, vfseed = random.split(seed, num=4)
            irandints = random.randint(iseed, shape=(minibatch_size,), minval=0, maxval=num_itrain_samples)
            frandints = random.randint(fseed, shape=(minibatch_size,), minval=0, maxval=num_ftrain_samples)
            vis = vmap(ke_sampler)(random.split(viseed, num=minibatch_size))
            vfs = vmap(ke_sampler)(random.split(vfseed, num=minibatch_size))
            out_p_f_samples = lax.cond(bidirectional,
                                       lambda _x : p_f_samples[frandints],
                                       lambda _x : jnp.zeros_like(vfs), None)
            return p_i_samples[irandints], vis, out_p_f_samples, vfs
    return pull_samples



def get_step_fn(loss_fn, get_params_fn, update_fn, pull_samples_fn):
    if update_fn is not None:
        touch_fn = value_and_grad(loss_fn)
    else:
        def touch_fn(*args, **kwargs): return loss_fn(*args, **kwargs), None
        def update_fn(*args, **kwargs): return None

    def step(_iter, opt_state, seed):
        _params = get_params_fn(opt_state)
        in_pi_samples, in_vis, in_pf_samples, in_vfs = pull_samples_fn(seed)
        val, g = touch_fn(_params, in_pi_samples, in_vis, in_pf_samples, in_vfs)
        return val, update_fn(_iter, g, opt_state)
    return step

def get_optimizer_fn(train_step_fn, validation_step_fn):
    import tqdm
    import numpy as np
    validate = False if validation_step_fn is None else True

    def optimize(seed, num_iters, opt_state): #not a pure pythonic function.
        train_losses, validation_losses = np.zeros(num_iters), np.zeros(num_iters)
        _trange = tqdm.trange(num_iters, desc='Bar desc', leave=True)

        for i in _trange:
            seed, train_seed, validation_seed = random.split(seed, num=3)

            #validate
            if validate:
                validation_loss, _ = validation_step_fn(i, opt_state, validation_seed)
                validation_losses[i] = validation_loss
            else:
                validation_loss = f"N/A"

            #train
            train_loss, opt_state = train_step_fn(i, opt_state, train_seed)
            train_losses[i] = train_loss

            _trange.set_description(f"test / validation loss:    {train_loss} / {validation_loss}")
            _trange.refresh() # to show immediately the update

        return opt_state, train_losses, validation_losses

    return optimize

def get_RNVP_trainer(u_i,
                     u_f,
                     ke_fn,
                     ke_sampler,
                     forward_rnvp,
                     backward_rnvp,
                     p_i_train_samples,
                     p_f_train_samples,
                     seed,
                     kT,
                     optimizer = optimizers.adam,
                     optimizer_kwargs_dict = {'step_size': 1e-3},
                     minibatch_size = None, #if `None`, will train on full batch; see `get_pull_samples_fn`
                     p_i_validation_samples = None,
                     p_f_validation_samples = None
                     ):
    """
    let's build a rnvp
    """
    #query which optimization method to do
    assert p_i_train_samples is not None, f"p_i_train_samples must be provided"
    num_p_i_train_samples = p_i_train_samples.shape[0]

    #should we validate
    validate = False if p_i_validation_samples is None else True
    print(f"validate is {validate}")

    # split some seeds so we can initialize the forward
    init_vs_seed, init_fwd_seed = random.split(seed)
    init_xs = p_i_train_samples[0]
    init_vs = ke_sampler(init_fwd_seed)

    # forward initializer
    initd_forward_rnvp = forward_rnvp() #initialize the forward rnvp
    nn_params = initd_forward_rnvp.init(init_fwd_seed, init_xs, init_vs) # initialize the neural network parameters
    in_forward_work_fn = partial(forward_work_fn, u_i_fn = u_i, u_f_fn = u_f, ke_fn = ke_fn, initd_forward_rnvp = initd_forward_rnvp, kT=kT) # partial the work function

    if backward_rnvp is not None and p_f_train_samples is not None: # if we are provided backward rnvp and training samples at the target..
        bidirectional = True # we can use bidirectional estimators
        if validate : assert p_i_validation_samples is not None and p_f_validation_samples is not None, f"if the train is bidirectional, so must validation"
        num_p_f_train_samples = p_f_train_samples.shape[0] # get the number of samples at the posterior
        assert p_i_train_samples.shape[1:] == p_f_train_samples.shape[1:], f"the training data must have the same dimension"
        initd_backward_rnvp = backward_rnvp() #init the backward rnvp
        in_backward_work_fn = partial(backward_work_fn, u_i_fn = u_i, u_f_fn = u_f, ke_fn = ke_fn, initd_backward_rnvp = initd_backward_rnvp, kT=kT)

        #original loss
        def loss(params, pi_xs, pi_vs, pf_xs, pf_vs):
            forward_wks = vmap(in_forward_work_fn, in_axes=(None, 0, 0))(params, pi_xs, pi_vs)
            backward_wks = vmap(in_backward_work_fn, in_axes=(None, 0,0))(params, pf_xs, pf_vs)
            return jnp.mean(forward_wks) + jnp.mean(backward_wks)

        #force loss
        # if we have access to bothe forward and backward rnvps, we can use force matching, yay!
        u_forward_flow = partial(u_forward_rnvp, initd_backward_rnvp = initd_backward_rnvp, u_i_fn = u_i, ke_fn = ke_fn, kT = kT)
        u_backward_flow = partial(u_backward_rnvp, initd_forward_rnvp = initd_forward_rnvp, u_f_fn = u_f, ke_fn = ke_fn, kT = kT)

        def get_target_position_forces(x, u): # get the reduced forces of `u_i_fn`, `u_f_fn`
            return -1 * grad(lambda _x: u(_x) / kT)(x)
        def get_target_velocity_forces(v, ke_fn): # get the reduced forces of the velocity potential
            return -1 * grad(ke_fn)(v)
        def get_flow_forces(x, v, params, u_flow): # get the reduced forces of the flow potential
            x_forces, v_forces = -1 * grad(u_flow, argnums=(1,2))(params, x, v)
            return x_forces, v_forces

        forward_target_x_forces = partial(get_target_position_forces, u = u_f) # B target position forces
        backward_target_x_forces = partial(get_target_position_forces, u = u_i) # A target position forces
        target_v_forces = partial(get_target_velocity_forces, ke_fn = ke_fn) # A,B target velocity forces

        forward_flow_forces = partial(get_flow_forces, u_flow = u_forward_flow) # A' forces
        backward_flow_forces = partial(get_flow_forces, u_flow = u_backward_flow) # B' forces

        # wrap the partial forces fn to compute the l2 norm of forces.
        def get_force_l2_norm(x, v, params, target_x_forces_fn, target_v_forces_fn, flow_forces_fn): #init this, which we will partial and vmap
            target_x_forces = target_x_forces_fn(x)
            target_v_forces = target_v_forces_fn(v)
            flow_x_forces, flow_v_forces = flow_forces_fn(x, v, params)
            return jnp.sum((target_x_forces - flow_x_forces)**2) + jnp.sum((target_v_forces - flow_v_forces)**2)

        forward_force_l2_norm = partial(get_force_l2_norm,
                                        target_x_forces_fn = forward_target_x_forces,
                                        target_v_forces_fn = target_v_forces,
                                        flow_forces_fn = forward_flow_forces)
        backward_force_l2_norm = partial(get_force_l2_norm,
                                         target_x_forces_fn = backward_target_x_forces,
                                         target_v_forces_fn = target_v_forces,
                                         flow_forces_fn = backward_flow_forces)

        def force_loss(params, pi_xs, pi_vs, pf_xs, pf_vs):
            # forward
            forward_l2_norms = vmap(forward_force_l2_norm, in_axes = (0,0,None))(pf_xs, pf_vs, params)
            backward_l2_norms = vmap(backward_force_l2_norm, in_axes = (0,0,None))(pi_xs, pi_vs, params)
            return jnp.mean(forward_l2_norms) + jnp.mean(backward_l2_norms)

    else: #we only go unidirectionally
        bidirectional = False
        if validate: assert p_i_validation_samples is not None, f"validation samples not provided"
        num_p_f_train_samples = None
        initd_backward_rnvp = None
        in_backward_work_fn = None

        def loss(params, pi_xs, pi_vs, pf_xs, pf_vs):
            forward_wks = vmap(in_forward_work_fn, in_axes=(None, 0, 0))(params, pi_xs, pi_vs)
            return jnp.mean(forward_wks)

    print(f"bidirectional is {bidirectional}")

    #now make the initializers
    opt_init, opt_update, get_params = optimizer(**optimizer_kwargs_dict)
    opt_state = opt_init(nn_params)

    #training sampler and stepper
    train_pull_samples_fn = get_pull_samples_fn(p_i_samples = p_i_train_samples,
                                                p_f_samples = p_f_train_samples,
                                                minibatch_size = minibatch_size,
                                                ke_sampler = ke_sampler)

    train_step_fn = get_step_fn(loss_fn = loss,
                                get_params_fn = get_params,
                                update_fn = opt_update,
                                pull_samples_fn = train_pull_samples_fn)

    #validation sampler and stepper
    if validate:
        validate_pull_samples_fn = get_pull_samples_fn(p_i_samples = p_i_validation_samples,
                                                       p_f_samples = p_f_validation_samples,
                                                       minibatch_size=None,
                                                       ke_sampler = ke_sampler)


        validate_step_fn = get_step_fn(loss_fn = loss,
                                get_params_fn = get_params,
                                update_fn = None,
                                pull_samples_fn = validate_pull_samples_fn)
    else:
        validate_pull_samples_fn = None
        validate_step_fn = None

    #make the optimizer fn
    jtrain_step_fn = jit(train_step_fn)

    if validate:
        jvalidate_step_fn = jit(validate_step_fn)
    else:
        jvalidate_step_fn = None

    optimizer = get_optimizer_fn(jtrain_step_fn, jvalidate_step_fn)

    #make a dictionary of functions to return
    out_function_dict = {'forward_work_fn' : in_forward_work_fn,
                         'backward_work_fn' : in_backward_work_fn,
                         'loss_fn' : loss,
                         'train_pull_samples_fn' : train_pull_samples_fn,
                         'train_step_fn' : train_step_fn,
                         'validate_pull_samples_fn' : validate_pull_samples_fn,
                         'optimizer_fn' : optimizer,
                         'get_params_fn': get_params,
                         'opt_init_fn': opt_init,
                         'opt_state' : opt_state,
                         'initd_forward_rnvp' : initd_forward_rnvp,
                         'initd_backward_rnvp' : initd_backward_rnvp,
                         'force_loss': force_loss,
                         'u_forward_flow': u_forward_flow,
                         'u_backward_flow': u_backward_flow,
                         'forward_flow_forces': forward_flow_forces,
                         'backward_flow_forces': backward_flow_forces,
                         'forward_force_l2_norm': forward_force_l2_norm,
                         'backward_force_l2_norm': backward_force_l2_norm}

    return out_function_dict

# annealed flow transport? (forward only)
