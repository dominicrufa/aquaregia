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

from aquaregia.utils import Array, ArrayTree
MLPFn = Callable[[ArrayTree], ArrayTree] # default MLP returned from a fn

# Helper Fns
def get_mlp_by_activation(activation : Optional[MLPFn] = nn.relu) -> Module:
    """get an mlp with a specified activation fn"""
    class MLP(nn.Module):
        """A flax MLP."""
        features: Sequence[int]

        @nn.compact
        def __call__(self, inputs):
            x = inputs
            for i, lyr in enumerate([nn.Dense(feat, dtype=jnp.float64) for feat in self.features]):
                x = lyr(x)
                if i != len(self.features) - 1:
                    x = activation(x)
            return x
    return MLP

def make_mlp(features : Sequence[int],
             activation : Optional[MLPFn] = nn.tanh) -> MLPFn:
    @jraph.concatenated_args
    def update_fn(inputs):
        return get_mlp_by_activation(activation)(features)(inputs)
    return update_fn

def scalar_mlp_kernel_init(val : float) -> Callable:
    return lambda seed, shape: val * jnp.ones(shape)

class ScalarMLP(nn.Module):
    """
    define an mlp that just multiplies the input Array by a scalar.
    This is useful when we want to initialize a normalizing flow close to identity.
    """
    num : int
    kernel_init : Callable[[Array, ...], Array]

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param('kernel',
                        self.kernel_init, # Initialization function
                           ((self.num, 1)))  # shape info.
        y = kernel * inputs
        return y

def make_scalar_mlp(num, kernel_init) -> MLPFn:
    @jraph.concatenated_args
    def update_fn(inputs):
        return ScalarMLP(num, kernel_init)(inputs)
    return update_fn

"""Here lies a deprecated `ScalarMLP`"""
# class ScalarMLP(nn.Module):
#     """
#     define an mlp that just multiplies the input Array by a scalar.
#     This is useful when we want to initialize a normalizing flow close to identity.
#     """
#     kernel_init : Callable = partial(random.uniform, minval=1e-2, maxval=2e-2)
#
#     @nn.compact
#     def __call__(self, inputs):
#         kernel = self.param('kernel',
#                         self.kernel_init, # Initialization function
#                            ())  # shape info.
#         y = (kernel**2) * inputs
#         return y

# def make_scalar_mlp() -> MLPFn:
#     @jraph.concatenated_args
#     def update_fn(inputs):
#         return ScalarMLP()(inputs)
#     return update_fn

# this should be a util...
def polynomial_switching_fn(r : Array, r_cutoff : float, r_switch : float) -> float:
    x = (r - r_switch) / (r_cutoff - r_switch)
    switched_x = lax.cond(r < r_switch,
                   lambda _x: 1.,
                   lambda _x: 1. + (_x**3) * (-10. + _x * (15. - (6. * _x))),
                   x)
    return lax.cond(r < r_cutoff, lambda _x: switched_x, lambda _x: 0., switched_x)

def get_stacked_mat_from_vector(vec : Array)-> Array:
    """
    utility to make a stacked vector (e.g. for hs stacking)

    Arguments:
    vec : Array(shape=N,M)

    Returns:
    Array : Array(shape=N,N,M)
    """

    num_attrs, attr_features = vec.shape
    tiled_vec = jnp.tile(vec, (num_attrs,1,1))
    swapped_tiled_vec = jnp.swapaxes(tiled_vec, 1,0)
    return jnp.concatenate([swapped_tiled_vec,tiled_vec], axis=-1)

def make_default_message_fn(mlp_e : MLPFn, # mlp_e : R^{2*nf_h + 1 + 1} -> R^{nf_m}
                            displacement_or_metric : space.DisplacementOrMetricFn,
                            rbf_kwargs : Optional[ArrayTree] = {'mu_ks' : jnp.linspace(0,1,16), 'gamma' : 1.}) -> Callable[[Graph], ArrayTree]:
    """
    function that returns a GNMessageFn (i.e. Eq.3)
    TODO : add support for decaying coupling fn for messages!
    """
    from aquaregia.utils import radial_basis
    rbf = partial(radial_basis, **rbf_kwargs)
    vmetric = vmap(vmap(space.canonicalize_displacement_or_metric(displacement_or_metric), in_axes = (None,0)), (0, None))

    def gn_message_fn(xs, hs, edges):
        N = xs.shape[0]
        distances = vmetric(xs, xs)[..., jnp.newaxis] # make a new axis so we can push to rbf
        rbs = jnp.apply_along_axis(rbf, axis=2, arr=distances)
        received_sent_hs = get_stacked_mat_from_vector(hs)
        messages = mlp_e(received_sent_hs, rbs, edges) # are edges a symmetric matrix?
        return messages

    return gn_message_fn

def make_default_update_h_fn(mlp_h : MLPFn, # mlp_h : R^{nf_h + nf_m} -> R^{nf_h}
                             aggregate_self_messages : Optional[bool] = False
                            ) -> Callable[[ArrayTree, Array, Array], ArrayTree]:
    """function that returns Eq.6"""
    if aggregate_self_messages:
        def aggregate_message_fn(messages): return jnp.sum(messages, axis=1)
    else:
        def aggregate_message_fn(messages):
            message_features = messages.shape[-1]
            zero_features = jnp.zeros(message_features)
            mod_messages = ops.index_update(messages, ops.index[jnp.diag_indices(messages.shape[0])], zero_features)
            return jnp.sum(mod_messages, axis=1)

    def gn_update_h_fn(hs,
                       messages):
        aggregated_message_attributes = aggregate_message_fn(messages) #sum over the senders aggregated on receivers
        updated_hs = mlp_h(hs, aggregated_message_attributes) # pass hs and aggregated message attrs to the h-updating mlp
        return updated_hs

    return gn_update_h_fn

# RNVP velocity update function generator
def make_velocity_RNVP_fns(message_fn,
                           h_fn,
                           passable_message_fn,
                           mlp_x,
                           mlp_v,
                           displacement_fn,
                           shift_fn,
                           C,
                           use_vs_make_scalars = False,
                           make_scalars_vals = {'update_xs': 1e-6, 'velocity_t_fn': 1e-6, 'velocity_log_s_fn': 1e-6}):
    """create a function that will make the scalar shift update vectors for velocity from positions (and graph hs, edges).
       critically, the output is exclusively a function of xs and immutable graph attributes.

       All equation references are from https://arxiv.org/pdf/2102.09844.pdf

       This function is effectively a modified version of the last term in RHS of Eq. 7.

       The update will work as follows:
       1. create messages m_{i,j} from Eq. 3
       2. update the node hs from Eq. 6
       3. recreate messages m_{i,j} with the updated nodes so that the nodes contain position information
       4. generate velocity scalar updates (rightmost term on Eq. 7 RHS)

       TODO : `use_vs_make_scalars`: do we want this to multiply the input along the batch dim, or a single scalar?
    """

    vmetric = vmap(vmap(space.canonicalize_displacement_or_metric(displacement_fn), in_axes = (None,0)), (0, None))
    vdisplacement = vmap(vmap(displacement_fn, in_axes = (None,0)), (0, None))

    make_scalars_vals_keys = list(make_scalars_vals.keys())

    assert 'update_xs' in make_scalars_vals_keys

    if use_vs_make_scalars:
        assert 'velocity_t_fn' in make_scalars_vals_keys
        assert 'velocity_log_s_fn' in make_scalars_vals_keys

    # sort out the make
    i_fn = lambda x: x
    def get_i_fn(*args, **kwargs) : return i_fn
    log_s_make_scalar = make_scalar_mlp if use_vs_make_scalars else get_i_fn
    t_make_scalar = make_scalar_mlp if use_vs_make_scalars else get_i_fn

    # default loaders
    def update_vs(vs, ts, log_s):
        """we do not use the shift function to update velocities because velocities are never periodic(?)"""
        return vs * jnp.exp(log_s) + ts

    def update_xs(xs, vs):
        """
        update the positions from the velocities; there is no log s fn for xs (as a function of velocity)
        since that probably doesnt make sense.
        NOTE : we use a scalar mlp
        a scalar mlp is just a function that multiplies the input by a (trainable) scalar. this is because at the start of training,
        we want the transformation to be close to identity.
        """
        return shift_fn(xs,
                        make_scalar_mlp(num = vs.shape[0], kernel_init = scalar_mlp_kernel_init(val=make_scalars_vals['update_xs'])) (vs))

    def message_and_hs_fn(xs, hs, edges):
        # first thing to do is create messages
        messages = message_fn(xs, hs, edges)

        #second: update node hs from messages
        updated_hs = h_fn(hs, messages)

        #third: make new messages with updated nodes
        new_messages = passable_message_fn(xs, updated_hs, edges)

        return updated_hs, new_messages

    def velocity_t_fn(xs, messages):
        # normalize and protect from nans
        num_positions, dimension = xs.shape
        offset_metrics = vmetric(xs, xs) + C
        aug_normalized_x_distances = vdisplacement(xs, xs) / offset_metrics[..., jnp.newaxis]
        mlp_messages = mlp_x(messages)
        #print(f"mlp messages shape: {mlp_messages.shape}")

        summands = aug_normalized_x_distances * mlp_messages
        summands = ops.index_update(summands, ops.index[jnp.diag_indices(num_positions)], jnp.zeros(dimension))
        updated_vectors = jnp.sum(summands, axis=1)
        updated_vectors = t_make_scalar(num = xs.shape[0], kernel_init = scalar_mlp_kernel_init(val=make_scalars_vals['velocity_t_fn']))(updated_vectors)
        return updated_vectors

    def velocity_log_s_fn(hs, dimension):
        log_multipliers = mlp_v(hs) #compute log multipliers
        log_multipliers = log_s_make_scalar(num = hs.shape[0], kernel_init = scalar_mlp_kernel_init(val=make_scalars_vals['velocity_log_s_fn']))(log_multipliers) # multiply this by a scalar if we need
        tiled_multipliers = dimension * log_multipliers
        logdetJ = tiled_multipliers.sum() #this is a single summation
        return log_multipliers, logdetJ

    def RNVP_wrapper(xs0, vs0, og_hs, edges, dimension):
        """
        write the full RNVP in a wrapper fn
        In particular, we are performing 3 affine couplings/transforms.

        NOTES :
        1. geometric info is necessarily compressed in the HM1/2 stages, meaning that the V1/2 steps of the
           logs and t velocity updaters are lossy. This might be resolved by augmenting the dimension of the hs and messages
           to reduce lossiness.
        2. we can compound as many `RNVP_wrapper`s as we want since the logdetJ computations are always additive.
        3. should we be
        """

        ##HM1##
        hs1, messages1 = message_and_hs_fn(xs0, og_hs, edges)

        ##V1##
        vs_t1 = velocity_t_fn(xs0, messages1) #get v scalar update
        vs_logs1, logdetJs1 = velocity_log_s_fn(hs1, dimension)
        vs1 = update_vs(vs0, vs_t1, vs_logs1)

        ##R1##
        xs1 = update_xs(xs0, vs1)

        ##HM2##
        hs2, messages2 = message_and_hs_fn(xs1, og_hs, edges)

        ##V2##
        vs_t2 = velocity_t_fn(xs1, messages2) #get v scalar update
        vs_logs2, logdetJs2 = velocity_log_s_fn(hs2, dimension)
        vs2 = update_vs(vs1, vs_t2, vs_logs2)

        # collect logdetJs
        logdetJs = logdetJs1 + logdetJs2

        return xs1, vs2, logdetJs

    return RNVP_wrapper


# make RNVP Module
def make_RNVP_module(graph : Graph,
                     message1_fn_features : Sequence[int],
                     message2_fn_features : Sequence[int],
                     h_fn_features: Sequence[int],
                     mlp_x_features : Sequence[int],
                     mlp_v_features : Sequence[int],
                     displacement_fn : Callable,
                     shift_fn : Callable,
                     seed : Optional[Array] = random.PRNGKey(13),
                     C : Optional[float] = 1.,
                     use_vs_make_scalars : Optional[bool] = False,
                     make_scalars_vals : Optional[ArrayTree] = {'update_xs': 1e-6, 'velocity_t_fn': 1e-6, 'velocity_log_s_fn': 1e-6},
                     num_VRV_repeats : Optional[int] = 1,

                     # optional activation fns
                     message1_fn_activation : Optional[MLPFn] = nn.relu,
                     message2_fn_activation : Optional[MLPFn] = nn.relu,
                     h_fn_activation: Optional[MLPFn] = nn.relu,
                     mlp_x_activation : Optional[MLPFn] = nn.swish,
                     mlp_v_activation : Optional[MLPFn] = nn.swish,
                     message_rbf_kwargs : Optional[ArrayTree] = {'mu_ks' : jnp.linspace(0,1,16), 'gamma' : 1.}
                    ):
    # make RNVP_wrapper
    RNVP_wrapper = make_velocity_RNVP_fns(message_fn = make_default_message_fn(mlp_e = make_mlp(message1_fn_features, message1_fn_activation), displacement_or_metric=displacement_fn, rbf_kwargs = message_rbf_kwargs),
                                          h_fn = make_default_update_h_fn(mlp_h=make_mlp(h_fn_features, h_fn_activation)),
                                          passable_message_fn = make_default_message_fn(mlp_e = make_mlp(message1_fn_features, message1_fn_activation), displacement_or_metric=displacement_fn, rbf_kwargs = message_rbf_kwargs),
                                          mlp_x = make_mlp(mlp_x_features, mlp_x_activation),
                                          mlp_v = make_mlp(mlp_v_features, mlp_v_activation),
                                          displacement_fn = displacement_fn,
                                          shift_fn = shift_fn,
                                          C = C,
                                          use_vs_make_scalars=use_vs_make_scalars,
                                          make_scalars_vals=make_scalars_vals)

    # pull the graph apart
    hs, _xs, _vs, edges = graph # pull out the graph
    dimension = _xs.shape[-1]
    assert dimension == _vs.shape[-1]


    class RNVP(nn.Module):
        """
        make RNVP module to perform N * (V R V) updates
        """

        @nn.compact
        def __call__(self, xs, vs):
            logdetJs = 0.
            for i in range(num_VRV_repeats):
                xs, vs, _logdetJs = RNVP_wrapper(xs, vs, hs, edges, dimension)
                logdetJs = _logdetJs + logdetJs
            return xs, vs, logdetJs

    rnvp = RNVP()
    params = rnvp.init(seed, _xs, _vs)

    return rnvp, params
