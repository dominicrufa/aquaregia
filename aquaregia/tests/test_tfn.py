"""test aquaregia.tfn"""
from jax.config import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from functools import partial
import haiku as hk
from aquaregia.utils import Array, ArrayTree
from aquaregia import tfn

def rotation_matrix(axis, theta):
    import scipy
    return scipy.linalg.expm(np.cross(np.eye(3), axis * theta))

def random_rotation_matrix(numpy_random_state, epsilon=tfn.DEFAULT_EPSILON):
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


def execute_TensorFieldMLP(seed = random.PRNGKey(23), L = 1):
    """dummy mlp to test the `TensorFieldMLP`; this just makes sure there are no errors"""
    random_vec_seed, init_seed = random.split(seed) # split the seed
    output_sizes = [8,4]
    m_dim = tfn.L_to_M_dict[L]
    input_tensor = random.normal(random_vec_seed, (5, 8, m_dim))

    def _wrapper(_input_tensor):
        mlp = tfn.TensorFieldMLP(output_sizes = output_sizes, L = L)
        return mlp(inputs = _input_tensor)

    _fn = hk.without_apply_rng(hk.transform(_wrapper))
    init_params = _fn.init(init_seed, input_tensor)
    _out = _fn.apply(init_params, input_tensor)

def test_TensorFieldMLP(seed = random.PRNGKey(2356), Ls = [0,1]):
    """run `execute_TensorFieldMLP`"""
    for l in Ls:
        run_seed, seed = random.split(seed)
        execute_TensorFieldMLP(seed = run_seed, L = l)

def test_SE3_equivariance(
    seed = jax.random.PRNGKey(434),
    N = 10,
    hs_dim = 8,
    displacement_fn = tfn.DEFAULT_VDISPLACEMENT_FN,
    r_switch = 1.5,
    r_cut = 2.,
    max_L = 1):
    """a simple, composable test that will test the equivariance of different filters"""
    from copy import deepcopy
    import time

    #initialize positions and labels
    pos_seed, label_seed, init_seed, translate_seed, vec_seed, seed = random.split(seed, num=6)
    translation_vec = jax.random.normal(translate_seed, shape=(3,)) # define random translation
    rotation_matrix = random_rotation_matrix(np.random.RandomState()) # get a rotation matrix
    random_positions = random.normal(pos_seed, (N,3)) # generate random positions in 3D
    random_labels = random.normal(label_seed, (N,hs_dim)) # make random labels
    random_vecs = random.normal(label_seed, (N,hs_dim, 3)) # the input L=1 tensor
    transformed_positions = random_positions @ rotation_matrix + translation_vec # rotate and translate randomly the input

    # test convolution
    def _wrapper(positions, input_dict):
        r_ij = tfn.DEFAULT_VDISPLACEMENT_FN(positions, positions)
        unit_r_ij, norms = tfn.unit_vectors_and_norms(r_ij)
        norms = jnp.maximum(norms, tfn.DEFAULT_EPSILON)

        #compute RBFs
        rbf_module = tfn.SinusoidalBasis(r_switch = r_switch, r_cut = r_cut, name="SinusoidBasis")
        rbfs = rbf_module(r_ij=jnp.squeeze(norms))

        # convolution
        convolution_module = tfn.Convolution(filter_mlp_dicts = {0: {'output_sizes': [8,8], 'activation': jax.nn.swish}, 1:{'output_sizes': [8,8], 'activation': jax.nn.swish}})
        convolution_dict = convolution_module(in_tensor_dict = input_dict, rbf_inputs = rbfs, unit_vectors = unit_r_ij)

        # TensorNetworkMLP
        output_sizes = {0: [8,1], 1: [8,4]}
        out_dict = {key : tfn.TensorFieldMLP(output_sizes=output_sizes[key], L = key)(inputs = val) for key, val in convolution_dict.items()}

        return out_dict

    # transform convolution
    in_dict = {0: random_labels[..., jnp.newaxis], 1: random_vecs}
    _fn = hk.without_apply_rng(hk.transform(_wrapper))
    init_params = _fn.init(init_seed, random_positions, in_dict)
    _out_dict = _fn.apply(init_params, random_positions, in_dict)

    # can we check the rotation functional
    rot_in_dict = deepcopy(in_dict)
    rot_in_dict[1] = in_dict[1] @ rotation_matrix
    _out_rot_dict = _fn.apply(init_params, transformed_positions, rot_in_dict)

    # assert rotation equivariance upon L=0,1, convolution
    assert jnp.allclose(_out_dict[0], _out_rot_dict[0])
    assert jnp.allclose(_out_dict[1] @ rotation_matrix, _out_rot_dict[1])


    # last thing to do is see if we can take the grad and val of the function...
    @jax.jit
    def func(positions, input_dict, params):
        out_dict = _fn.apply(params, positions, input_dict)
        out_energy = jnp.sum(out_dict[0])
        return out_energy

    jit_time = time.time()
    out_energy = func(random_positions, in_dict, init_params)
    print(time.time() - jit_time)

    jit_time = time.time()
    out_energy = func(random_positions, in_dict, init_params)
    print(time.time() - jit_time)



    out_energy, out_forces = jax.value_and_grad(func)(random_positions, in_dict, init_params)
    print(f"out energy, forces:", out_energy, out_forces)

    return _out_dict, _out_rot_dict, rotation_matrix
