"""test aquaregia.unction"""
from jax import numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
from aquaregia.utils import Array, ArrayTree
from jax import random
import jax
import haiku as hk
import jraph

def get_random_positions_and_neighbor_lists(seed, R, num_stacks, shift_fn, neighbor_fns, periodic):
    """generate random positions and neighbor list indices for testing's sake to generate identical graphs"""
    import functools
    if neighbor_fns.allocate is None: # vacuum
        from aquaregia.utils import get_vacuum_neighbor_list
        base_neighbor_list = get_vacuum_neighbor_list(R.shape[0])
        nbr_update_fn = lambda _x, neighbors: base_neighbor_list
    else:
        base_neighbor_list = neighbor_fns.allocate(R)
        nbr_update_fn = neighbor_fns.update

    _update = functools.partial(nbr_update_fn, neighbors = base_neighbor_list)
    def update(_q): return _update(_q).idx
    def new_posits(seed, R): return jax.vmap(shift_fn, in_axes=(0,0))(R, random.normal(seed, shape=R.shape)*1e-4)
    stacked_positions = jax.vmap(lambda _seed, _x: new_posits(_seed, _x), in_axes=(0, None))(jax.random.split(seed, num=num_stacks), R)
    stacked_neighbor_list_indices = jax.vmap(update)(stacked_positions)
    return stacked_positions, stacked_neighbor_list_indices

def make_vac_rot_translated_graph(seed, graph):
    translation_vec = random.normal(seed, shape=(3,))
    positions, velocities = graph.nodes['xs'], graph.nodes['vs']
    from jraph.examples.higgs_detection import get_random_rotation_matrix
    rot_matrix = get_random_rotation_matrix()[1:, 1:]
    new_positions = positions @ rot_matrix + translation_vec
    new_velocities = velocities @ rot_matrix
    return graph._replace(nodes = {'xs': new_positions, 'vs': new_velocities, 'hs': graph.nodes['hs']}), translation_vec, rot_matrix

def test_MaskedBatchNorm():
    """test the `MaskedBatchNorm`; ensure it appropriately masks the right dimension and scales outputs to unit normal"""
    from aquaregia.unction import MaskedBatchNorm
    _inputs = random.normal(random.PRNGKey(2623), shape = (5,3))
    _mask = Array([1,1,1,1,0], dtype=jnp.bool_)
    def _run(inputs, mask, is_training): return MaskedBatchNorm(True, True, 0.9, name='red')(inputs, mask, is_training=True)
    _run = hk.without_apply_rng(hk.transform_with_state(_run))
    params, state = _run.init(random.PRNGKey(362), _inputs,_mask, is_training=True)
    _out, state = _run.apply(params, state, _inputs, _mask, is_training=True)
    assert jnp.allclose(jnp.mean(_out[:4], axis=0), jnp.zeros(3))
    assert jnp.allclose(jnp.var(_out[:4], axis=0), 1., atol=1e-4)

def test_MaskedMLP():
    """test the `MaskedMLP`; ensure it appropriately runs an MLP and scales the outputs to unit normal"""
    from aquaregia.unction import MaskedMLP
    def masked_mlp(inputs, mask, is_training): return MaskedMLP(name=f"m_mlp", output_sizes=[4,4,2], activate_last_layer=False)(inputs, mask, is_training)
    masked_mlp = hk.without_apply_rng(hk.transform_with_state(masked_mlp))
    _inputs = random.normal(random.PRNGKey(2352), shape=(5,3))
    mask = Array([1,1,1,1,0], dtype=jnp.bool_)
    params, state = masked_mlp.init(random.PRNGKey(362), _inputs, mask, is_training=True)
    outputs, state = masked_mlp.apply(params, state, _inputs, mask, is_training=True)
    assert jnp.allclose(outputs[-1], jnp.zeros(2)), f"mask"
    assert jnp.allclose(jnp.mean(outputs[:4], axis=0), jnp.zeros(2))
    assert jnp.allclose(jnp.var(outputs[:4], axis=0), 1., atol=1e-4)

def render_get_identical_system_GraphsTuple(periodic=True, num_stacks=3, seed = random.PRNGKey(3246)):
    from aquaregia.tests.test_tincture import get_periodic_particles
    from aquaregia.unction import get_identical_system_GraphsTuple
    seed, run_seed = random.split(seed)
    R, displacement, shift, nbr_fn, hs, edges, box_size = get_periodic_particles(seed = run_seed,
                           periodic = periodic,
                           particles_per_side = 4,
                           spacing = 0.1,
                           hs_features = 2,
                           r_cutoff = 0.15,
                           dr_threshold = 0.0,
                           capacity_multiplier = 1.25,
                           edges_maxval = 2,
                           dimension = 3)
    seed, run_seed = random.split(seed)
    R = jax.vmap(shift, in_axes=(0,0))(R, random.normal(run_seed, shape=R.shape)*1e-3)
    seed, run_seed = random.split(seed)
    stacked_positions, stacked_neighbor_list_indices = get_random_positions_and_neighbor_lists(run_seed, R, num_stacks, shift, nbr_fn, periodic)
    graph, node_mask, edge_mask = get_identical_system_GraphsTuple(stacked_positions, stacked_neighbor_list_indices, hs, jnp.zeros_like(stacked_positions))

    # node assertions
    assert graph.n_node[-1]==1, f"the last graph must have 1 node"
    assert not node_mask[-1], f"the last node is pad"
    assert jnp.count_nonzero(node_mask) == len(node_mask)-1

    # edge assertions
    assert len(edge_mask) - jnp.count_nonzero(edge_mask) == jnp.count_nonzero(jnp.where(stacked_neighbor_list_indices==R.shape[0], True, False))
    return graph, node_mask, edge_mask, displacement, shift, nbr_fn, hs, edges, box_size

def test_get_identical_system_GraphsTuple():
    """test `get_identical_system_GraphsTuple` in periodic and non periodic regime for a stack of 3 identical graphs. we test the node and edge masks, too"""
    _ = render_get_identical_system_GraphsTuple(periodic=True)
    _ = render_get_identical_system_GraphsTuple(periodic=False)

"""
we still need to test the EGCL and the RNVP modules.
then we have to write the MaskedBatchNorm for the EGCL velocity update.
"""
def render_EGCL(periodic=False, seed = random.PRNGKey(3246), num_layers=4):
    from aquaregia.unction import EGCL
    import functools
    run_seed, seed = random.split(seed)
    graph, node_mask, edge_mask, displacement, shift, nbr_fn, hs, edges, box_size = render_get_identical_system_GraphsTuple(periodic=periodic, seed = run_seed)
    message_aggregator = functools.partial(jraph.segment_sum, num_segments = jnp.sum(graph.n_node))
    def egcl_fn(graph, node_mask, edge_mask, is_training):
        return EGCL(name=f"EGCL", displacement_fn=displacement, num_layers=num_layers, partialed_message_aggregator = message_aggregator)(graph, node_mask, edge_mask, is_training)

    egcl_fn = hk.without_apply_rng(hk.transform_with_state(egcl_fn))
    run_seed, seed = random.split(seed)
    params, state = egcl_fn.init(run_seed, graph, node_mask, edge_mask, True)

    @jax.jit
    def jit_egcl_fn(params, state, graph, node_mask, edge_mask):
        _outputs, _state = egcl_fn.apply(params, state, graph, node_mask, edge_mask, is_training=False)
        return _outputs
    hs, messages, aggregated_messages, displacements, unscaled_metrics_squared = jit_egcl_fn(params, state, graph, node_mask, edge_mask)

    #now for the rotation/translation
    run_seed, seed = random.split(seed)
    mod_graph, _, _ = make_vac_rot_translated_graph(run_seed, graph)
    mod_hs, mod_messages, mod_aggregated_messages, mod_displacements, mod_unscaled_metrics_squared = jit_egcl_fn(params, state, mod_graph, node_mask, edge_mask)

    # hs and agg messages are unchanged
    assert jnp.allclose(hs, mod_hs)
    assert jnp.allclose(aggregated_messages, mod_aggregated_messages)
    assert jnp.allclose(unscaled_metrics_squared, mod_unscaled_metrics_squared)

def test_EGCL():
    """
    test `aquaregia.unction.EGCL` layer (aperiodic only)
    TODO: periodic
    """
    render_EGCL()

def render_RNVPVelocityEGCL(periodic=False, seed = random.PRNGKey(3246)):
    from aquaregia.unction import RNVPVelocityEGCL
    import functools
    run_seed, seed = random.split(seed)
    graph, node_mask, edge_mask, displacement, shift, nbr_fn, hs, edges, box_size = render_get_identical_system_GraphsTuple(periodic=periodic, seed = run_seed)
    EGCL_kwargs = {'displacement_fn': displacement,
                   'num_layers': 2,
                   'partialed_message_aggregator': functools.partial(jraph.segment_sum, num_segments=jnp.sum(graph.n_node))
                  }

    def rnvp_v_fn(graph, node_mask, edge_mask, is_training):
        _fn = RNVPVelocityEGCL(EGCL_kwarg_dict = EGCL_kwargs)(graph, node_mask, edge_mask, is_training)
        return _fn

    egcl_fn = hk.without_apply_rng(hk.transform_with_state(rnvp_v_fn))
    run_seed, seed = random.split(seed)
    params, state = egcl_fn.init(run_seed, graph, node_mask, edge_mask, True)

    @jax.jit
    def jit_egcl_fn(params, state, graph, node_mask, edge_mask):
        _outputs, _state = egcl_fn.apply(params, state, graph, node_mask, edge_mask, is_training=False)
        return _outputs
    log_s, translations = jit_egcl_fn(params, state, graph, node_mask, edge_mask)

    #now for the rotation/translation
    run_seed, seed = random.split(seed)
    mod_graph, translation_vec, rotation_matrix = make_vac_rot_translated_graph(run_seed, graph)
    mod_log_s, mod_translations = jit_egcl_fn(params, state, mod_graph, node_mask, edge_mask)

    # assert log_s is unchanged (should be invariant to rotation/translation)
    assert jnp.allclose(mod_log_s, log_s)

    # assert  that the translations are equivariant about rotation
    rot_translations = translations @ rotation_matrix
    assert jnp.allclose(mod_translations, rot_translations)

def test_RNVPVelocityEGCL():
    """
    test `RNVPVelocityEGCL` in the aperiodic case. we make sure log_s scales are invariant to rotation/translations while translation elements are
    equivariant
    """
    render_RNVPVelocityEGCL()
