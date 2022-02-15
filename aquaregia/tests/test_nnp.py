"""test nnp"""
from jax.config import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from functools import partial
import haiku as hk
from aquaregia.utils import Array, ArrayTree
from aquaregia.nnp import make_energy_module
from aquaregia.tfn import DEFAULT_EPSILON, DEFAULT_VDISPLACEMENT_FN, unit_vectors_and_norms, mask_tensor, SinusoidalBasis

def test_lifting_procedure(seed = random.PRNGKey(2346),
                           num_particles = 100,
                           SinusoidalBasis_kwargs = {'r_switch': 1.,
                                  'r_cut': 1.5,
                                  'basis_init' : hk.initializers.Constant(constant=jnp.linspace(1., 8., 8))
                                 }
                           ):
    """
    test to ensure that a lifted subset of particles is actually decoupled in a system.
    This is performed by computing the energy of a nn model without lifting and again when with lifting past the r_cutoff.
    It should be the case that in the lifted regime, we should recover the same energy upon the rotation/translation of a subset of particles.
    """
    from aquaregia.tests.test_tfn import random_rotation_matrix
    atom_subset_max = 40
    atom_subset = jnp.arange(atom_subset_max)
    init_seed, positions_seed, bias_seed = random.split(seed, num=3)
    positions = jax.random.normal(positions_seed, shape=(num_particles,3)) * 0.5
    feature_dict = { 0:random.normal(init_seed, shape=(num_particles, 8,1)), 1: None}
    energy_fn, constructor = make_energy_module(max_L = 1, # we specifically want this to be one since we want to convolve l >= 1 shapes
                       num_particles = num_particles,
                       lifted_particles = range(atom_subset_max),
                       SinusoidalBasis_kwargs = SinusoidalBasis_kwargs)
    energy_fn = hk.without_apply_rng(hk.transform(energy_fn))

    # initialize and randomize biases (this is a necessity since biases initialized at zero will always )
    init_params = energy_fn.init(init_seed, positions, feature_dict, DEFAULT_EPSILON, 0., 0.)
    from jax.flatten_util import ravel_pytree
    flat_nn_params, back_fn = jax.flatten_util.ravel_pytree(init_params)
    mod_flat_nn_params = random.normal(bias_seed, shape=flat_nn_params.shape)
    init_params = back_fn(mod_flat_nn_params)


    # compute energy unlifted
    unlifted_energy = energy_fn.apply(init_params, positions, feature_dict, DEFAULT_EPSILON, 0., 0.)

    # compute lifted energy
    lifted_energy = energy_fn.apply(init_params, positions, feature_dict, DEFAULT_EPSILON, 0., 2.)

    # rotate position subset and recompute unlifted, lifted energies
    rot_matrix = random_rotation_matrix(np.random.RandomState())
    translation_vec = np.random.normal(3)
    position_subset = positions[:atom_subset_max] @ rot_matrix + translation_vec
    rot_positions = positions.at[:atom_subset_max].set(position_subset)
    lifted_rot_energy = energy_fn.apply(init_params, rot_positions, feature_dict, DEFAULT_EPSILON, 0., 2.)

    # rotate and translate the whole thing
    full_rot_positions = positions @ rot_matrix + translation_vec
    unlifted_rot_energy = energy_fn.apply(init_params, full_rot_positions, feature_dict, DEFAULT_EPSILON, 0., 0.)

    # assert equivariance, first of all
    assert np.isclose(unlifted_energy, unlifted_rot_energy), f"""the unlifted energy ({unlifted_energy})
                                                                 should match the unlifted rotated energy ({unlifted_rot_energy})"""
    assert np.isclose(lifted_energy, lifted_rot_energy), f"""the lifted energy ({lifted_energy})
                                                             should match the lifted rotated energy ({lifted_rot_energy})"""
    assert not np.isclose(unlifted_energy, lifted_energy), f"""the unlifted energy ({unlifted_energy})
                                                               should NOT match the lifted energy ({lifted_energy})"""

def test_SinusoidalBasisMasking(seed = random.PRNGKey(257),
                                num_particles=100,
                                SinusoidalBasis_kwargs = {'r_switch': 1.,
                                                          'r_cut': 1.5,
                                                          'basis_init' : hk.initializers.Constant(constant=jnp.linspace(1., 8., 16))
                                                         }
                               ):
    """
    test that the `SinusoidalBasis` MLP recovers the same output at cutoff as when it is masked
    """
    init_seed, positions_seed = random.split(seed)

    def sinusoid_fn(r_ij, epsilon=DEFAULT_EPSILON, mask_val=0.):
        return SinusoidalBasis(**SinusoidalBasis_kwargs)(r_ij = r_ij, epsilon=epsilon, mask_val=mask_val)

    sinusoid_fn = hk.without_apply_rng(hk.transform(sinusoid_fn))

    positions = jax.random.normal(positions_seed, shape=(num_particles,3)) * 0.5
    r_ij = DEFAULT_VDISPLACEMENT_FN(positions, positions)
    unit_r_ij, norms = unit_vectors_and_norms(r_ij)
    norms = mask_tensor(norms, mask_val=0.)
    norms = jnp.squeeze(norms)
    # print(norms)

    # get the places where the distance is beyond the cutoff
    #past_cutoff_bools = jnp.where(jnp.all(norms >= SinusoidalBasis_kwargs['r_cut']) or jnp.all(norms == 0.), True, False)
    past_cutoff_bools = norms >= SinusoidalBasis_kwargs['r_cut']
    diag_true = jnp.eye(num_particles,dtype=jnp.bool_)
    past_cutoff_bools = jnp.logical_or(past_cutoff_bools, diag_true)
    assert not jnp.all(past_cutoff_bools), f"all of the r_ijs are past the cutoff, so this test is not informative"

    #initialize the network
    init_params = sinusoid_fn.init(init_seed, norms, DEFAULT_EPSILON, 0.)

    # apply
    out_r_feats = sinusoid_fn.apply(init_params, norms, DEFAULT_EPSILON, 0.)
    # print(out_r_feats)

    # assert that the masking cutoff operation works
    zero_pads_locs = np.all(out_r_feats == 0., axis=-1)
    assert jnp.array_equal(zero_pads_locs, past_cutoff_bools), f"""the zero padding locations ({zero_pads_locs})
                                                   are not all equal to the past_cutoff_bools ({past_cutoff_bools})"""

test_SinusoidalBasisMasking()
