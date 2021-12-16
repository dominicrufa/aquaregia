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
from typing import Optional, Sequence

from jax_md import partition

"""
default hk.Module kwarg dicts
"""
DEFAULT_BATCHNORM_KWARGS = {'create_scale': True, 'create_offset': True, 'decay_rate': 0.9}
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
                          'BatchNorm_last_layer': True,
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
class MaskedBatchNorm(hk.BatchNorm):
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
                 name: Optional[str] = None):
        super().__init__(create_scale=create_scale,
                         create_offset=create_offset,
                         decay_rate=decay_rate,
                         eps=eps,
                         scale_init=scale_init,
                         offset_init=offset_init,
                         axis=axis,
                         cross_replica_axis=cross_replica_axis,
                         cross_replica_axis_index_groups=cross_replica_axis_index_groups,
                         data_format=data_format,
                         name=name
                         )

        if self.axis != None:
            raise ValueError(f"`MaskedBatchNorm` only supports `None` axis")

        if data_format != "channels_last":
            raise ValueError(f"`data_format` must be default (i.e. `channels_last`)")

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
            self.mean_ema(mean)
            self.var_ema(var)

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
        return (inputs - mean) * inv + offset

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
