"""constructors of neural network potential"""
from jax.config import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from functools import partial
import haiku as hk
from typing import Sequence, Callable, Dict, Tuple, Optional, Iterable, Any
import functools
from aquaregia.utils import Array, ArrayTree, polynomial_switching_fn
from aquaregia import tfn

def make_energy_module(max_L : int,
                       num_particles : int,
                       lifted_particles : Optional[Sequence] = [],
                       SinusoidalBasis_kwargs : Optional[dict] = {'r_switch': 2., 'r_cut': 2.5},
                       feature_mlp_kwargs : Optional[dict] = None,
                       conv_shapes_dict : Optional[dict] = {0: {'output_sizes': [8,8], 'activation': jax.nn.swish},
                                                            1: {'output_sizes': [8,8], 'activation': jax.nn.swish}},
                       tf_mlp_shapes_dict : Optional[dict] = {0: {0: {'output_sizes': [8,8], 'nonlinearity': jax.nn.swish},
                                                                  1:{'output_sizes': [8,8], 'nonlinearity': jax.nn.swish}},
                                                              1: {0: {'output_sizes': [8,1], 'nonlinearity': jax.nn.swish}}},
                       mask_output=False):
    """
    TODO: reorganize the inputs (maybe as a class?); allow for other masking schema!

    `conv_shapes_dict` should look like {<layer_int>: {"output_sizes": [8,8], "activation": jax.nn.swish}}
    `tf_mlp_shapes_dict` should look like {<layer_int>: {L: {"output_sizes": [8,8], "activation": jax.nn.swish}}}}

    Example:
    # L=0
    energy_fn, out_constructor = make_energy_module(max_L = 0,
                                                    num_particles = 22,
                                                    SinusoidalBasis_kwargs = {'r_switch': 2.,
                                                                              'r_cut': 2.5,
                                                                              'basis_init' : hk.initializers.Constant(constant=jnp.linspace(1., 8., 16))},
                                                    conv_shapes_dict = {0: {'output_sizes': [16,16], 'activation': jax.nn.swish},
                                                                        1: {'output_sizes': [16,16], 'activation': jax.nn.swish}},
                                                    tf_mlp_shapes_dict =  {0: {0: {'output_sizes': [16,16], 'nonlinearity': jax.nn.swish}},
                                                                           1: {0: {'output_sizes': [16,1], 'nonlinearity': jax.nn.swish}}},
                                                    mask_output=False)

    # L=0,1
    energy_fn, out_constructor = make_energy_module(max_L = 1,
                                                    num_particles = 22,
                                                    SinusoidalBasis_kwargs = {'r_switch': 2.,
                                                                              'r_cut': 2.5,
                                                                              'basis_init' : hk.initializers.Constant(constant=jnp.linspace(1., 8., 16))},
                                                    conv_shapes_dict = {0: {'output_sizes': [16,16], 'activation': jax.nn.swish},
                                                                        1: {'output_sizes': [16,16], 'activation': jax.nn.swish}},
                                                    tf_mlp_shapes_dict =  {0: {0: {'output_sizes': [16,16], 'nonlinearity': jax.nn.swish},
                                                                               1: {'output_sizes': [16,16], 'nonlinearity': jax.nn.swish}},
                                                                           1: {0: {'output_sizes': [16,1], 'nonlinearity': jax.nn.swish}}},
                                                    mask_output=False)

    energy_fn = hk.without_apply_rng(hk.transform(energy_fn))
    init_positions = random.normal(random.PRNGKey(25), shape=(N, 3))
    feature_dict = { 0:random.normal(random.PRNGKey(253), shape=(N, hs_feature_size,1)), 1: None}

    """
    print(f"making energy module...")
    num_layers = len(conv_shapes_dict)
    conv_switching_fn = functools.partial(polynomial_switching_fn,
                                     r_cutoff = SinusoidalBasis_kwargs['r_cut'],
                                     r_switch = SinusoidalBasis_kwargs['r_switch'])

    if max_L > 1: # this isnt yet implemented
        raise NotImplementedError(f"L > 1 is not yet implemented")

    if num_layers != len(tf_mlp_shapes_dict): #check the number of layers are equal
        raise ValueError(f"""each `tf_mlp` layer should be preceded by a `conv` layer,
                             so the number of layers must match;
                             however, there are {num_layers} `conv` layers and only {len(tf_mlp_shapes_dict)} `tf_mlp` layers
                          """)

    # check the last tf_mlp_shapes_dict
    last_tf_mlp_layer = tf_mlp_shapes_dict[num_layers-1]
    if 0 not in last_tf_mlp_layer.keys():
        raise ValueError(f"the last tf_mlp_layer must have an L=0 mlp")
    if last_tf_mlp_layer[0]['output_sizes'][-1] != 1:
        raise ValueError(f"the last tf_mlp_layer's L=0 mlp must output a shape of 0")

    #finally, check to make sure that if max_L == 1, each tf_mlp_layer has a L=1 mlp (except the last one, since the last convolution only returns an L=0)
    for tf_mlp_layer_idx in tf_mlp_shapes_dict.keys():
        if tf_mlp_layer_idx != num_layers - 1:
            tf_Ls = set(tf_mlp_shapes_dict[tf_mlp_layer_idx].keys())
            expected_Ls = set(range(max_L+1))
            if not expected_Ls.issubset(tf_Ls):
                raise ValueError(f"""tf_mlp layer {tf_mlp_layer_idx}  provided with Ls {tf_Ls} must be a subset of {expected_Ls};
                it does not have a mlp for each convolution layer""")
        else:
            if set(tf_mlp_shapes_dict[tf_mlp_layer_idx].keys()) != {0}:
                raise ValueError(f"the last tf_mlp may only contain a convolution on L=0")

    # decide the angular number args
    if max_L == 0: # we ony do a schnet-like activation
        combination_dict = {0: {0: [0]}}
        final_combination_dict = combination_dict
    elif max_L == 1: # convolve l=1
        combination_dict = {0: {0: [0], 1: [1]}, 1: {0: [1], 1: [0,1]}}
        final_combination_dict = {0: {0: [0]}, 1: {1: [0]}}
    else:
        raise NotImplementedError(f"L_max > 1 is not implemented")

    # construct conv dicts
    conv_dict_list = [{'filter_mlp_dicts': {_L: conv_shapes_dict[idx] for _L in range(max_L+1)},
                       'name' : f"tfn_convolution_{idx}",
                       'mask_output': mask_output,
                       'switching_fn': conv_switching_fn,
                       'combination_dict' : combination_dict if idx != num_layers-1 else final_combination_dict} for idx in range(num_layers)]

    # construct mlp dicts
    mlp_dict_list = tf_mlp_shapes_dict
    for layer_idx in mlp_dict_list.keys():
        layer_dict = mlp_dict_list[layer_idx]
        for L_dict_idx in layer_dict:
            L_dict = layer_dict[L_dict_idx]
            L_dict['L'] = L_dict_idx
            L_dict['name'] = f"tf_mlp_layer_{layer_idx}_L{L_dict_idx}"


    # package mlp lists in to a returnable dict
    out_constructor_dict = {'Convolution': conv_dict_list, 'TensorFieldMLP': mlp_dict_list}

    # handle lifting particles; create lifting dimension
    lifting_dimension = np.zeros(num_particles)
    if len(lifted_particles) != 0:
        lifting_dimension[lifted_particles] = 1.
    lifting_dimension = Array(lifting_dimension)[..., jnp.newaxis]



    class EnergyModule(hk.Module):
        """
        `hk.Module` that will return a scalar (sum of scalars from every particle in the input)
        """
        def __init__(self,
                     num_particles,
                     name : Optional[str] = f"energy",
                     feature_mlp_kwargs : Optional[dict] = None,
                    ):
            """
            initializer for `EnergyModule`
            feature_mlp_kwargs : args to an MLP that convolve the input L=0 of the feature dict
            """
            super().__init__(name=name)
            self.SinusoidalBasis_module = tfn.SinusoidalBasis(**SinusoidalBasis_kwargs)
            if feature_mlp_kwargs is not None:
                self.feature_convolution_module = hk.nets.MLP(**feature_mlp_kwargs)
            else:
                self.feature_convolution_module = None

            self._num_particles = num_particles

            # setup the for loop
            layers = {}
            for layer in range(num_layers):
                conv_module = tfn.Convolution(**conv_dict_list[layer])
                tfn_module_dict = {_L: tfn.TensorFieldMLP(**mlp_dict_list[layer][_L]) for _L in mlp_dict_list[layer].keys()}
                layers[layer] = {'conv': conv_module, 'mlp': tfn_module_dict}
            self.layers = layers

        def __call__(self,
                     positions : Array,
                     feature_dictionary : dict,
                     epsilon : Optional[float] = tfn.DEFAULT_EPSILON,
                     mask_value : Optional[float] = 0.,
                     lifting_value : Optional[float] = 0.):
            """
            arguments:

            positions : N x 3 positions array
            feature_dictionary : {L : Array(N,channels,m)} of inputs
            epsilon : zero padding float
            mask_val : masking value for diagonal
            lifting_particles : set of particle indices to lift into 4th dimension
            lifting_value : distance to lift particles into 4th dimension

            returns:
            output energy (float)
            """
            # check the number of particles
            if positions.shape[0] != self._num_particles:
                raise ValueError(f"given positions of {positions.shape[0]} particles does not match expected number of particles ({self._num_particles})")

            # process the input positions
            augmented_positions = jnp.concatenate([positions, lifting_dimension*lifting_value], axis=-1)
            r_ij = tfn.DEFAULT_VDISPLACEMENT_FN(augmented_positions, augmented_positions)
            unit_r_ij, norms = tfn.unit_vectors_and_norms(r_ij) # compute unit vectors and norms
            unit_r_ij = tfn.mask_tensor(unit_r_ij[:,:,:-1], mask_val = mask_value) # remove the 4th dimension since we only want the projection onto 3D
            norms = tfn.mask_tensor(norms, mask_val = mask_value)
            squeezed_norms = jnp.squeeze(norms)

            # rbf convolution
            rbf_inputs = self.SinusoidalBasis_module(r_ij = squeezed_norms, epsilon=epsilon, mask_val=mask_value)

            # loop over convolutions and tf_mlps
            in_tensor_dict = {key: val for key, val in feature_dictionary.items()} # make a copy
            if self.feature_convolution_module is not None:
                if len(feature_dictionary[0].shape) != 2:
                    raise ValueError(f"if passing features to be convolved in-module, then the raw L0 inputs should have a dimension of 2, but instead is of shape ({feature_dictionary[0].shape})")
                convolved_features = self.feature_convolution_module(feature_dictionary[0])
                # print(f"convolved L0 features shape: ", convolved_features.shape)
                in_tensor_dict[0] = convolved_features[..., jnp.newaxis]

            for layer_idx in range(len(self.layers)):
                out_tensor_dict = {}
                layer_dict = self.layers[layer_idx]
                conv_dict = layer_dict['conv'](in_tensor_dict=in_tensor_dict, rbf_inputs=rbf_inputs, unit_vectors=unit_r_ij, r_ij = squeezed_norms, epsilon = tfn.DEFAULT_EPSILON)
                for _L in conv_dict.keys():
                # for _L, mlp in layer_dict['mlp'].items(): #iterate over the mlps/angular numbers
                    mlp = layer_dict['mlp'][_L]
                    p_array = mlp(inputs=conv_dict[_L], epsilon=epsilon) # pass convolved arrays through mlp
                    out_tensor_dict[_L] = p_array # populate out dict

                in_tensor_dict=out_tensor_dict # rename for passing

            # global pool the out tensor dict's zeroth output
            out_particle_array = in_tensor_dict[0] #angular l=0 only!
            if out_particle_array.shape[-2] != 1:
                raise ValueError(f"the last feature of L=0 must have a channel dimension of 1")
            return jnp.sum(out_particle_array)

    # wrap the EnergyModule
    def energy_fn(positions, feature_dictionary, epsilon, mask_val, lifting_val):
        return EnergyModule(num_particles=num_particles, feature_mlp_kwargs = feature_mlp_kwargs)(positions = positions,
                                                         feature_dictionary = feature_dictionary,
                                                         epsilon = epsilon,
                                                         mask_value = mask_val,
                                                         lifting_value = lifting_val)

    return energy_fn, out_constructor_dict


"""training helper"""
