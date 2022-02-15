"""haiku-based constructors of rotational-, translational- and permutational-equivariant RealNVP-based deterministic leapfrog integrators with support for both free and periodic boundary conditions"""
from jax.config import config
config.update("jax_enable_x64", True)
from jax import numpy as jnp
from aquaregia.utils import Array, ArrayTree
from jax import random
import jax
import haiku as hk
import jraph
import functools
import copy
from typing import Optional, Sequence, Callable, Dict

from jax_md import partition

from haiku._src import utils
hk.get_channel_index = utils.get_channel_index

"""
default hk.Module kwarg dicts
"""
DEFAULT_BATCHNORM_KWARGS = {'create_scale': True, 'create_offset': True, 'decay_rate': 0.9, 'RNVP': False}
DEFAULT_PHI_E_KWARGS = {'output_sizes': [8,8,8],
                          'Linear_kwargs': {},
                          'BatchNorm_bool': True,
                          'BatchNorm_kwargs': DEFAULT_BATCHNORM_KWARGS,
                          'activation_fn': jax.nn.swish,
                          'BatchNorm_last_layer': True,
                          'mask_pad': 0.,
                          'activate_last_layer': False}
DEFAULT_PHI_X_KWARGS = {'output_sizes': [8,8,1],
                          'Linear_kwargs': {},
                          'BatchNorm_bool': True,
                          'BatchNorm_kwargs': DEFAULT_BATCHNORM_KWARGS,
                          'activation_fn': jax.nn.swish,
                          'BatchNorm_last_layer': False,
                          'mask_pad': 0.,
                          'activate_last_layer': False}
DEFAULT_PHI_V_KWARGS = DEFAULT_PHI_X_KWARGS


DEFAULT_PHI_H_KWARGS = DEFAULT_PHI_E_KWARGS

def reduce_dense_neighbor_list_idx(neighbor_list_idx):
    num_nodes, max_pads = neighbor_list_idx.shape
    receiver_sender_edges = jax.vmap(
                                     jax.vmap(lambda _x, _y: Array([_x, _y]),
                                              in_axes=(None,0)),
                            in_axes=(0,0))(jnp.arange(num_nodes, dtype=jnp.int32), neighbor_list_idx).reshape((num_nodes * max_pads, 2))
    mask = jnp.all(receiver_sender_edges != num_nodes, axis=1)
    receivers = receiver_sender_edges[:,0]
    senders = receiver_sender_edges[:,1]
    return receivers, senders, mask

def get_identical_system_GraphsTuple(positions,
                                     stacked_neighbor_list_indices,
                                     hs_features,
                                     velocities):
    """
    basically a fancy `jraph.utils.pad_with_graphs` function that should be jittable
    """
    dimension = positions.shape[-1]
    num_stacks, num_nodes_per_stack, max_neighbors_per_stack = stacked_neighbor_list_indices.shape

    n_unmasked_nodes = jnp.ones(num_stacks, dtype=jnp.int32) * num_nodes_per_stack
    total_num_unmasked_nodes = jnp.sum(n_unmasked_nodes)
    node_rename_template = jnp.concatenate([Array([0], dtype=jnp.int32), jnp.cumsum(n_unmasked_nodes)[:-1]])[..., jnp.newaxis]

    _global_attrs = jnp.zeros(num_stacks+1)[..., jnp.newaxis] # there is an extra (zeroed) global for the last masking node


    stacked_receivers, stacked_senders, stacked_mask = jax.vmap(reduce_dense_neighbor_list_idx)(stacked_neighbor_list_indices)
    receivers = jnp.where(stacked_mask, stacked_receivers + node_rename_template, total_num_unmasked_nodes).flatten()
    senders = jnp.where(stacked_mask, stacked_senders + node_rename_template, total_num_unmasked_nodes).flatten()

    edges = jnp.zeros(len(senders))[..., jnp.newaxis] # the edges carry no information

    pad_n_edge = jnp.count_nonzero(stacked_mask, axis=1) # get the number of nonmasked edges in each stack element
    pads_edges = jnp.count_nonzero(jnp.invert(stacked_mask)) # get the total number of masked edges in the whole stack


    # construct the nodes. there is a single masked node with zeros.
    unmasked_node_dict = {
                          'hs': jnp.vstack([jnp.tile(hs_features, (num_stacks,1)), jnp.zeros_like(hs_features[0])]),
                          'xs': jnp.vstack([jnp.concatenate(positions, axis=0), jnp.zeros(dimension)]),
                          'vs': jnp.vstack([jnp.concatenate(velocities, axis=0), jnp.zeros(dimension)])
                         }

    graph = jraph.GraphsTuple(
                              n_node = jnp.concatenate([n_unmasked_nodes, Array([1], dtype=jnp.int32)]), # nodes per graph plus an extra
                              n_edge = jnp.concatenate([pad_n_edge, Array([pads_edges], dtype=jnp.int32)]), # edges plus `pads_edges` extra
                              nodes = unmasked_node_dict, # add the nodes
                              edges = edges,
                              globals = _global_attrs,
                              senders = senders,
                              receivers = receivers
                             )
    node_mask, edge_mask = jnp.concatenate([jnp.ones(total_num_unmasked_nodes, dtype=jnp.bool_), Array([0], dtype=jnp.bool_)]), stacked_mask.flatten()
    return graph, node_mask, edge_mask

def modify_metrics_squared(displacements, edge_mask, is_training, MaskedBatchNorm_module, r_squared_cutoff = 1.):
    unscaled_metrics_squared = jax.vmap(lambda _disp: _disp.dot(_disp))(displacements)[..., jnp.newaxis]
    if MaskedBatchNorm_module is not None:
        scaled_metrics_squared = MaskedBatchNorm_module(inputs=unscaled_metrics_squared, mask=edge_mask, is_training=is_training)
    else:
        scaled_metrics_squared = unscaled_metrics_squared / r_squared_cutoff
    return unscaled_metrics_squared, scaled_metrics_squared

def calculate_aggregated_scaled_and_normed_displacements(phi_xs, receivers, displacements, metrics_squared, partialed_segment_fn, normalization_offset=1e-5):
    scaled_and_normalized_displacements = displacements * phi_xs / (jnp.sqrt(metrics_squared) + normalization_offset)
    aggregated_scaled_and_normalized_displacements = partialed_segment_fn(scaled_and_normalized_displacements, receivers) #should we use a masked mean here?
    return aggregated_scaled_and_normalized_displacements

def graph_extractor(graph):
    # pylint: disable=g-long-lambda
    nodes, edges, receivers, senders, globals_, n_node, n_edge = graph
    sent_attributes = jax.tree_util.tree_map(lambda n: n[senders], nodes)
    received_attributes = jax.tree_util.tree_map(lambda n: n[receivers], nodes)
    return sent_attributes, received_attributes

def get_edge_padding_mask(graph):
    node_pad_idx = jnp.sum(graph.n_node) - 1
    receiver_mask = jnp.where(graph.receivers != node_pad_idx, True,False)
    sender_mask = jnp.where(graph.senders != node_pad_idx, True, False)
    return jax.vmap(lambda _q: jnp.all(_q))(jnp.transpose(jnp.vstack([receiver_mask, sender_mask])))

def graph_generator(positions, velocities, hs, base_neighbor_list, neighbor_list_update_fn):
    stacked_neighbor_list_indices = jax.vmap(lambda _xs: neighbor_list_update_fn(_xs, base_neighbor_list).idx)(positions)
    return get_identical_system_GraphsTuple(positions = positions,
                                            stacked_neighbor_list_indices = stacked_neighbor_list_indices,
                                            hs_features=hs,
                                            velocities = velocities)


"""specialized MLPs"""

class RNVPExponentialMovingAverage(hk.ExponentialMovingAverage):
    """a special ExponentialMovingAverage that carries two hidden states
    """
    def __init__(self,
                decay,
                zero_debias: bool = True,
                warmup_length: int = 0,
                name: Optional[str] = None,
                state_init: Optional[Callable] = jnp.zeros):
        super().__init__(decay=decay, zero_debias=zero_debias, warmup_length=warmup_length, name=name)
        self._state_init = state_init

    def initialize(self, shape, dtype=jnp.float32):
        """If uninitialized sets the average to ``zeros`` of the given shape/dtype."""
        if hasattr(shape, "shape"):
            warnings.warn("Passing a value into initialize instead of a shape/dtype "
                    "is deprecated. Update your code to use: "
                    "`ema.initialize(v.shape, v.dtype)`.",
                    category=DeprecationWarning)
            shape, dtype = shape.shape, shape.dtype

        hk.get_state("hidden", shape, dtype, init=self._state_init)
        hk.get_state("hidden_tm1", shape, dtype, init=self._state_init)
        hk.get_state("average", shape, dtype, init=self._state_init)
        hk.get_state("average_tm1", shape, dtype, init=self._state_init)

    def __call__(
        self,
        value: jnp.ndarray,
        update_stats: bool = True) -> jnp.ndarray:
        """Updates the EMA and returns the new value.
        Args:
        value: The array-like object for which you would like to perform an
            exponential decay on.
        update_stats: A Boolean, whether to update the internal state
            of this object to reflect the input value. When `update_stats` is False
            the internal stats will remain unchanged.
        Returns:
        The exponentially weighted average of the input value.
        """
        if not isinstance(value, jnp.ndarray): value = jnp.asarray(value)

        counter = hk.get_state("counter", (), jnp.int32, init=hk.initializers.Constant(-self.warmup_length))
        counter = counter + 1

        decay = jax.lax.convert_element_type(self.decay, value.dtype)
        if self.warmup_length > 0:
            decay = jax.lax.select(counter <= 0, 0.0, decay)

        one = jnp.ones([], value.dtype)
        hidden_tm1 = hk.get_state("hidden", value.shape, value.dtype, init=self._state_init)
        hidden = hidden_tm1 * decay + value * (one - decay)

        average = hidden
        average_tm1 = hidden_tm1
        if self.zero_debias:
            average /= (one - jnp.power(decay, counter))
            average_tm1 /= (one - jnp.power(decay, counter))

        if update_stats:
            hk.set_state("counter", counter)

            hk.set_state("hidden", hidden)
            hk.set_state("hidden_tm1", hidden_tm1)

            hk.set_state("average", average)
            hk.set_state("average_tm1", average_tm1)

        return average_tm1

    @property
    def average(self): return hk.get_state("average")

    @property
    def average_tm1(self): return hk.get_state(f"average_tm1")

var_state_init = lambda shape, dtype: jnp.ones(shape, dtype=dtype)

DEFAULT_RNVPBATCHNORM_KWARGS = {'create_scale': True,
                                'create_offset': True,
                                'decay_rate': 0.9,
                                'RNVP': True,
                                'ema': RNVPExponentialMovingAverage,
                                'scale_init': hk.initializers.Constant(constant = 1.),
                                'mean_ema_kwargs': {},
                                'var_ema_kwargs': {'state_init': var_state_init,
                                                   'zero_debias': True}}

class MaskedBatchNorm(hk.Module):
    """
    hk.BatchNorm module with a masking modifier
    """
    def __init__(
                 self,
                 create_scale: bool,
                 create_offset: bool,
                 decay_rate: float,
                 eps: float = 1e-5,
                 scale_init: Optional[hk.initializers.Initializer] = None,
                 offset_init: Optional[hk.initializers.Initializer] = None,
                 axis: Optional[Sequence[int]] = None,
                 cross_replica_axis: Optional[str] = None,
                 cross_replica_axis_index_groups: Optional[Sequence[Sequence[int]]] = None,
                 data_format: str = "channels_last",
                 name: Optional[str] = None,
                 ema : Optional[hk.Module] = hk.ExponentialMovingAverage,
                 mean_ema_kwargs: Optional[Dict] = {},
                 var_ema_kwargs: Optional[Dict] = {},
                 RNVP: Optional[bool] = False
                 ):
        super().__init__(name=name)
        if not create_scale and scale_init is not None:
            raise ValueError("Cannot set `scale_init` if `create_scale=False`")
        if not create_offset and offset_init is not None:
            raise ValueError("Cannot set `offset_init` if `create_offset=False`")
        if (cross_replica_axis is None and cross_replica_axis_index_groups is not None):
            raise ValueError("`cross_replica_axis` name must be specified"
                       "if `cross_replica_axis_index_groups` are used.")

        self.create_scale = create_scale
        self.create_offset = create_offset
        self.eps = eps
        self.scale_init = scale_init or jnp.ones
        self.offset_init = offset_init or jnp.zeros
        self.axis = axis
        self.cross_replica_axis = cross_replica_axis
        self.cross_replica_axis_index_groups = cross_replica_axis_index_groups
        self.channel_index = utils.get_channel_index(data_format)
        self.mean_ema = ema(decay_rate, name="mean_ema", **mean_ema_kwargs)
        self.var_ema = ema(decay_rate, name="var_ema", **var_ema_kwargs)
        self.RNVP = RNVP

        if self.axis != None:
            raise ValueError(f"`MaskedBatchNorm` only supports `None` axis")

        if data_format != "channels_last":
            raise ValueError(f"`data_format` must be default (i.e. `channels_last`)")

        if RNVP and ema != RNVPExponentialMovingAverage:
            raise ValueError(f"RNVP requires `RNVPExponentialMovingAverage`")


    def __call__(
        self,
        inputs: jnp.ndarray,
        mask : jnp.ndarray,
        is_training: bool,
        test_local_stats: bool = False,
        scale: Optional[jnp.ndarray] = None,
        offset: Optional[jnp.ndarray] = None
        ) -> jnp.ndarray:
        """Computes the normalized version of the input.
        Example:
        >>> _inputs = random.normal(random.PRNGKey(2623), shape = (5,3))
        >>> _mask = Array([1,1,1,1,0], dtype=jnp.bool_)
        >>> def _run(inputs, mask, is_training): return MaskedBatchNorm(True, True, 0.9, name='red')(inputs, mask, is_training=is_training)
        >>> _run = hk.without_apply_rng(hk.transform_with_state(_run))
        >>> params, state = _run.init(random.PRNGKey(362), _inputs,_mask, is_training=True)
        >>> _out, state = _run.apply(params, state, _inputs, _mask, is_training=True)
        """
        axis_mask = jnp.repeat(mask[..., jnp.newaxis], repeats=inputs.shape[-1], axis=-1) #get axis mask
        if self.RNVP:
            inputs = inputs.flatten()[..., jnp.newaxis]
            axis_mask = axis_mask.flatten()[..., jnp.newaxis]

        if self.create_scale and scale is not None:
            raise ValueError("Cannot pass `scale` at call time if `create_scale=True`.")
        if self.create_offset and offset is not None:
            raise ValueError("Cannot pass `offset` at call time if `create_offset=True`.")

        channel_index = self.channel_index
        if channel_index < 0:
            channel_index += inputs.ndim

        if self.axis is not None:
            axis = self.axis
        else:
            axis = [i for i in range(inputs.ndim) if i != channel_index]

        if is_training or test_local_stats:
            mean = jnp.mean(inputs, axis, keepdims=True, where=axis_mask)
            mean_of_squares = jnp.mean(jnp.square(inputs), axis, keepdims=True, where=axis_mask)
            if self.cross_replica_axis:
                mean = jax.lax.pmean(
                    mean,
                    axis_name=self.cross_replica_axis,
                    axis_index_groups=self.cross_replica_axis_index_groups,
                    where = axis_mask)
                mean_of_squares = jax.lax.pmean(
                    mean_of_squares,
                    axis_name=self.cross_replica_axis,
                    axis_index_groups=self.cross_replica_axis_index_groups,
                    where = axis_mask)
            var = mean_of_squares - jnp.square(mean)
        else:
            mean = self.mean_ema.average
            var = self.var_ema.average

        if is_training:
            _mean = self.mean_ema(mean)
            _var = self.var_ema(var)
            if self.RNVP:
                # print(f"ema: {_mean}, {_var}")
                mean, var = _mean.flatten(), _var.flatten() # flatten it because we are keeping dims (because the fucker is hardcoded)

        w_shape = [1 if i in axis else inputs.shape[i] for i in range(inputs.ndim)]
        w_dtype = inputs.dtype

        if self.create_scale:
            scale = hk.get_parameter("scale", w_shape, w_dtype, self.scale_init)
        elif scale is None:
            scale = np.ones([], dtype=w_dtype)

        if self.create_offset:
            offset = hk.get_parameter("offset", w_shape, w_dtype, self.offset_init)
        elif offset is None:
            offset = np.zeros([], dtype=w_dtype)

        eps = jax.lax.convert_element_type(self.eps, var.dtype)
        inv = scale * jax.lax.rsqrt(var + eps)

        if not self.RNVP:
            out = (inputs - mean) * inv + offset
            return jnp.where(axis_mask, out, 0.)
        else: # we use self.RNVP, so we need to return the log scalar update
            return mean.flatten(), inv.flatten(), offset.flatten()

class MaskedMLP(hk.Module):
    def __init__(self,
                 output_sizes,
                 name=None,
                 Linear_kwargs = {},
                 BatchNorm_bool=True,
                 BatchNorm_kwargs = DEFAULT_BATCHNORM_KWARGS,
                 activation_fn = jax.nn.swish,
                 BatchNorm_last_layer=True,
                 mask_pad = 0.,
                 activate_last_layer=False
                ):
        """
        Example:
        >>> def masked_mlp(inputs, mask, is_training): return MaskedMLP(name=f"m_mlp", output_sizes=[4,4,2], activate_last_layer=True)(inputs, mask, is_training)
        >>> masked_mlp = hk.without_apply_rng(hk.transform_with_state(masked_mlp))
        >>> # graph.nodes['xs'] has dimension (251, 3) and last dimension is mask
        >>> # jraph.get_node_padding_mask(graph) returns a 1D array of `True` except the last dimension
        >>> params, state = masked_mlp.init(random.PRNGKey(25), graph.nodes['xs'], jraph.get_node_padding_mask(graph), is_training=True)
        >>> outputs, state = masked_mlp.apply(params, state, graph.nodes['xs'], jraph.get_node_padding_mask(graph), is_training=True)
        >>> # outputs.shape is (251, 2)
        """
        super().__init__(name=name)
        self._num_layers = len(output_sizes)
        self._output_sizes = output_sizes
        self._Linear_kwargs = Linear_kwargs
        self._BatchNorm_bool = BatchNorm_bool
        self._BatchNorm_kwargs = BatchNorm_kwargs
        self._activation_fn = activation_fn
        self._BatchNorm_last_layer = BatchNorm_last_layer
        self._mask_pad = mask_pad
        self._activate_last_layer = activate_last_layer

    def __call__(self, inputs, mask, is_training):
        """we do not mask out Linear/activation layers since it is assumed the output mask is rendered as -inf"""
        for i, output_size in enumerate(self._output_sizes):
            inputs = hk.Linear(output_size=output_size, **self._Linear_kwargs)(inputs)
            if i == self._num_layers - 1:
                if self._BatchNorm_last_layer: inputs = MaskedBatchNorm(**self._BatchNorm_kwargs)(inputs=inputs, mask=mask, is_training=is_training)
                if self._activate_last_layer: inputs = self._activation_fn(inputs)
            else:
                inputs = self._activation_fn(inputs)
                if self._BatchNorm_bool: inputs = MaskedBatchNorm(**self._BatchNorm_kwargs)(inputs = inputs, mask = mask, is_training=is_training)

        #mask
        repeated_mask = jnp.repeat(mask[..., jnp.newaxis], repeats=inputs.shape[-1], axis=-1)
        masked_inputs = jnp.where(repeated_mask, inputs, self._mask_pad)
        return masked_inputs

"""EGCL Module"""
class EGCL(hk.Module):
    def __init__(self,
                 displacement_fn,
                 num_layers,
                 partialed_message_aggregator,
                 phi_e_kwarg_dict = DEFAULT_PHI_E_KWARGS,
                 phi_h_kwarg_dict = DEFAULT_PHI_H_KWARGS,
                 MaskedBatchNorm_kwarg_dict = DEFAULT_BATCHNORM_KWARGS,
                 name=None
                 ):
        import functools
        super().__init__(name=name)
        self._vdisplacement_fn = jax.vmap(displacement_fn, in_axes=(0,0)) #vmap displacement fn

        self._num_layers = num_layers
        self._MaskedBatchNorm_kwarg_dict = MaskedBatchNorm_kwarg_dict
        self._partialed_modify_metrics_squared = functools.partial(modify_metrics_squared, MaskedBatchNorm_module = MaskedBatchNorm(**self._MaskedBatchNorm_kwarg_dict))
        self._message_aggregator = partialed_message_aggregator

        self._phi_e_kwarg_dict = phi_e_kwarg_dict
        self._phi_h_kwarg_dict = phi_h_kwarg_dict



    def __call__(self, graph, node_mask, edge_mask, is_training, metrics_squared = None):
        # extraction
        sent_attributes, received_attributes = graph_extractor(graph)
        if node_mask is None: node_mask = jraph.get_node_padding_mask(graph)
        if edge_mask is None: edge_mask = get_edge_padding_mask(graph)

        # metrics
        if metrics_squared is None:
            displacements = self._vdisplacement_fn(received_attributes['xs'], sent_attributes['xs'])
            unscaled_metrics_squared, scaled_metrics_squared = self._partialed_modify_metrics_squared(displacements = displacements,
                                                                                                      edge_mask = edge_mask,
                                                                                                      is_training=is_training)

        # iteratively make messages, aggregate, and update nodes
        for index in range(self._num_layers):
            nodes, messages, receivers, senders, globals_, n_node, n_edge = graph
            if index != 0:
                sent_attributes, received_attributes = graph_extractor(graph)
            # message generator
            phi_e_features = jnp.concatenate([received_attributes['hs'], sent_attributes['hs'], scaled_metrics_squared, messages], axis=1)
            messages = MaskedMLP(**self._phi_e_kwarg_dict)(inputs=phi_e_features, mask=edge_mask, is_training=is_training)

            # message aggregator
            aggregated_messages = self._message_aggregator(messages, segment_ids=receivers)
            aggregated_messages = MaskedBatchNorm(**self._MaskedBatchNorm_kwarg_dict)(inputs=aggregated_messages, mask=node_mask, is_training=is_training)


            # new hs calculator
            phi_h_features = jnp.concatenate([nodes['hs'], aggregated_messages], axis=1)
            hs = MaskedMLP(**self._phi_h_kwarg_dict)(inputs = phi_h_features, mask = node_mask, is_training=is_training)

            # update graph
            graph = graph._replace(nodes = {'xs': nodes['xs'], 'vs': nodes['vs'], 'hs': hs}, edges = messages)

        return hs, messages, aggregated_messages, displacements, unscaled_metrics_squared


class RNVPVelocityEGCL(hk.Module):
    """a special EGCL that will return log_s and t"""
    def __init__(self,
                 EGCL_kwarg_dict,
                 phi_x_module_kwarg_dict = DEFAULT_PHI_X_KWARGS,
                 phi_v_module_kwarg_dict = DEFAULT_PHI_V_KWARGS,
                 eps = 1e-5,
                 dimension=3,
                 name=None
                 ):
        super().__init__(name=name)
        self._EGCL_module = EGCL(**EGCL_kwarg_dict) #this has some private functions we can use
        self._phi_x_module = MaskedMLP(**phi_x_module_kwarg_dict)
        self._phi_v_module = MaskedMLP(**phi_v_module_kwarg_dict)
        self._eps = eps
        self._dimension = dimension

    def __call__(self, graph, node_mask, edge_mask, is_training):
        hs, messages, aggregated_messages, displacements, unscaled_metrics_squared = self._EGCL_module(graph=graph,
                                                                                                       node_mask = node_mask,
                                                                                                       edge_mask = edge_mask,
                                                                                                       is_training=is_training)

        # compute translations
        phi_xs = self._phi_x_module(inputs = messages, mask = edge_mask, is_training = is_training)
        aggregated_scaled_and_normalized_displacements = calculate_aggregated_scaled_and_normed_displacements(phi_xs = phi_xs,
                                                                                                              receivers = graph.receivers,
                                                                                                              displacements = displacements,
                                                                                                              metrics_squared = unscaled_metrics_squared,
                                                                                                              normalization_offset= self._eps,
                                                                                                              partialed_segment_fn = self._EGCL_module._message_aggregator
                                                                                                             )
        translations = aggregated_scaled_and_normalized_displacements
        # translations = jnp.zeros((hs.shape[0], 3)) # for no change

        #compute log_s
        #phi_vs = self._phi_v_module(inputs=hs, mask=node_mask, is_training=is_training)
        #log_s = jnp.repeat(phi_vs, repeats=self._dimension, axis=-1)
        log_s = jnp.zeros((hs.shape[0], 3))

        return log_s, translations

class PositionUpdate(hk.Module):
    """
    a simple position update function from velocities;
    subsequently, one must update a neighbor list and regenerate senders/receivers/mask/etc if periodic
    """
    def __init__(self, name, shift_fn, forward=True, shift_scalar = 1e-4):
        super().__init__(name=name)
        self._shift_scalar = shift_scalar
        def update_xs(graph):
            xs, vs = graph.nodes['xs'], graph.nodes['vs']
            shift_vals = vs if forward else -vs
            new_xs = jax.vmap(shift_fn, in_axes=(0,0))(xs, self._shift_scalar * shift_vals) # dt is presumed to be 1ps
            return graph._replace(nodes = {'xs': new_xs, 'vs': vs, 'hs': graph.nodes['hs']})

        self._update_xs = update_xs

    def __call__(self, graph): return self._update_xs(graph) # then one must create a new graph

class VelocityUpdate(hk.Module):
    def __init__(self,
                 name,
                 RNVPVelocityEGCL_kwarg_dict,
                 partialed_segment_sum,
                 # segment_ids = jnp.concatenate([Array([i]*num_nodes) for i, num_nodes in enumerate(graph.n_node)])
                 # functools.partial(jraph.segment_sum, segment_ids = segment_ids, num_segments = len(num_nodes))
                 forward=True,
                 RNVPBatchNorm=True,
                 RNVPBatchNorm_kwargs = DEFAULT_RNVPBATCHNORM_KWARGS
                 ):
        super().__init__(name=name)
        self._RNVPVelocityEGCL_module = RNVPVelocityEGCL(**RNVPVelocityEGCL_kwarg_dict)
        self._RNVPBatchNorm = RNVPBatchNorm

        # make the basic function for velocity update
        if forward:
            def update(v, log_s, t): return v * jnp.exp(log_s) + t
        else:
            def update(v, log_s, t): return (v - t) * jnp.exp(-log_s)

        if RNVPBatchNorm:
            RNVPBatchNorm_module = MaskedBatchNorm(**RNVPBatchNorm_kwargs) # make the MaskedBatchNorm
            if forward:
                def norm_update(v, log_s, t, mask, is_training):
                    new_vs = update(v, log_s, t)
                    mean, inv, offset = RNVPBatchNorm_module(inputs=new_vs, mask=mask, is_training=is_training)
                    new_mod_vs, log_inv = (new_vs - mean) * inv + offset, jnp.log(inv)
                    #print(f"new_mod_vs: {new_mod_vs[:-1].mean()}, {new_mod_vs[:-1].var()}")
                    return new_mod_vs, log_inv

            else:
                def norm_update(v, log_s, t, mask, is_training):
                    new_vs = update(v, log_s, t)
                    mean, inv, offset = RNVPBatchNorm_module(inputs=new_vs, mask=mask, is_training=is_training) # is this correct for "backward" pass?
                    new_mod_vs, log_inv = (((v - offset) / inv) + mean - t) * jnp.exp(-log_s), jnp.log(inv)
                    #print(f"new_mod_vs: {new_mod_vs[:-1].mean()}, {new_mod_vs[:-1].var()}")
                    return new_mod_vs, log_inv

        else: # retain single function because forward/backward functionality was previously handled
            def norm_update(v, log_s, t, mask, is_training):
                return update(v, log_s, t), 0.

        self._update = norm_update
        self._partialed_segment_sum = partialed_segment_sum


    def __call__(self, graph, node_mask, edge_mask, is_training):
        log_s, translations = self._RNVPVelocityEGCL_module(graph = graph, node_mask = node_mask, edge_mask=edge_mask, is_training = is_training)
        # print(f"log s: {jnp.max(jnp.abs(log_s))}")
        # print(f"translations: {jnp.max(jnp.abs(translations))}")
        xs, vs = graph.nodes['xs'], graph.nodes['vs']
        new_vs, log_invs = self._update(v = vs, log_s = log_s, t = translations, mask=node_mask, is_training=is_training)
        logdetJs = self._partialed_segment_sum(log_s).sum(axis=-1) + graph.n_node * vs.shape[-1] * log_invs
        # print((graph.n_node * vs.shape[-1] * log_invs).shape)
        return graph._replace(nodes = {'xs': xs, 'vs': new_vs, 'hs': graph.nodes['hs']}), logdetJs

class RNVP(hk.Module):
    def __init__(self,
                 name,
                 shift_fn,
                 num_repeats,
                 graph_regenerator_fn,
                 PositionUpdate_kwarg_dict,
                 VelocityUpdate_kwarg_dict,
                 forward=True,
                 ):
        super().__init__(name=name)

        # assert forward direction
        PositionUpdate_kwarg_dict['forward'] = True if forward else False
        VelocityUpdate_kwarg_dict['forward'] = True if forward else False
        self._forward = forward

        # equip shift fn with PositionUpdate
        PositionUpdate_kwarg_dict['shift_fn'] = shift_fn

        self._PositionUpdate_kwarg_dict = PositionUpdate_kwarg_dict
        self._VelocityUpdate_kwarg_dict = VelocityUpdate_kwarg_dict

        self._graph_regenerator_fn = graph_regenerator_fn
        self._num_repeats = num_repeats


    def __call__(self, graph, node_mask, edge_mask, is_training):
        init_name = f"V_0" if self._forward else f"V_{self._num_repeats*2}"
        graph, _logdetJ = VelocityUpdate(name=init_name, **self._VelocityUpdate_kwarg_dict)(graph = graph, node_mask = node_mask,
                                                                                        edge_mask = edge_mask, is_training = is_training)
        logdetJ = _logdetJ

        range_seq = range(1, self._num_repeats*2, 2) if self._forward else range(1, self._num_repeats*2, 2)[::-1]
        for i in range_seq:
            graph = PositionUpdate(name=f"R_{i}", **self._PositionUpdate_kwarg_dict)(graph = graph)
            graph, node_mask, edge_mask = self._graph_regenerator_fn(positions = graph.nodes['xs'], velocities = graph.nodes['vs'])
            inter_iter = i+1 if self._forward else i-1
            graph, _logdetJ = VelocityUpdate(name=f"V_{inter_iter}", **self._VelocityUpdate_kwarg_dict)(graph = graph, node_mask = node_mask,
                                                                                                 edge_mask = edge_mask, is_training = is_training)
            logdetJ = logdetJ + _logdetJ
        return graph, logdetJ



"""no batchnorm egcl"""
from aquaregia.unction import modify_metrics_squared, graph_extractor
class _EGCL(hk.Module):
    def __init__(self,
                 displacement_fn,
                 num_layers,
                 partialed_message_aggregator,
                 phi_e_mlp_kwargs = {'output_sizes': [8,8,8], 'activation': jax.nn.relu},
                 phi_h_mlp_kwargs = {'output_sizes': [8,8,8], 'activation': jax.nn.relu},
                 name=None
                 ):
        from aquaregia.unction import modify_metrics_squared
        super().__init__(name=name)
        self._vdisplacement_fn = jax.vmap(displacement_fn, in_axes=(0,0)) #vmap displacement fn

        self._num_layers = num_layers
        self._partialed_modify_metrics_squared = functools.partial(modify_metrics_squared, MaskedBatchNorm_module = None) # may want to add a radial basis function split?
        self._message_aggregator = partialed_message_aggregator #this is already partialed out

        # make the mlps
        self._phi_e_mlp_kwargs = phi_e_mlp_kwargs
        self._phi_h_mlp_kwargs = phi_h_mlp_kwargs



    def __call__(self, graph, node_mask, edge_mask, is_training, metrics_squared = None):
        # extraction
        sent_attributes, received_attributes = graph_extractor(graph)

        # metrics
        if metrics_squared is None: # then we need to compute the modified and unmodified metrics squared
            displacements = self._vdisplacement_fn(received_attributes['xs'], sent_attributes['xs'])
            unscaled_metrics_squared, scaled_metrics_squared = self._partialed_modify_metrics_squared(displacements = displacements,
                                                                                                      edge_mask = edge_mask,
                                                                                                      is_training=False) # we aren't using the BatchNorm here

        # iteratively make messages, aggregate, and update nodes
        for index in range(self._num_layers): #iterate over the number of layers
            nodes, messages, receivers, senders, globals_, n_node, n_edge = graph
            if index != 0:
                sent_attributes, received_attributes = graph_extractor(graph)

            # message generator
            phi_e_features = jnp.concatenate([received_attributes['hs'], sent_attributes['hs'], scaled_metrics_squared, messages], axis=1)
            messages = hk.nets.MLP(**self._phi_e_mlp_kwargs)(inputs=phi_e_features)
            messages = jnp.where(jnp.repeat(edge_mask[..., jnp.newaxis], repeats=messages.shape[-1], axis=-1), messages, 0.) # mask the messages


            # message aggregator
            aggregated_messages = self._message_aggregator(messages, segment_ids=receivers)


            # new hs calculator
            phi_h_features = jnp.concatenate([nodes['hs'], aggregated_messages], axis=1)
            hs = hk.nets.MLP(**self._phi_h_mlp_kwargs)(inputs = phi_h_features)
            hs = jnp.where(jnp.repeat(node_mask[..., jnp.newaxis], repeats=hs.shape[-1], axis=-1), hs, 0.) # mask the hs

            # update graph
            graph = graph._replace(nodes = {'xs': nodes['xs'], 'vs': nodes['vs'], 'hs': hs}, edges = messages)
        return hs, messages, aggregated_messages, displacements, unscaled_metrics_squared
