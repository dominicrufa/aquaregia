"""constructors of SE(3) and E(3) equivariant tensor field networks"""
from jax.config import config
config.update("jax_enable_x64", True) # perhaps we want to disable this in the future
from typing import Sequence, Callable, Dict, Tuple, Optional, Iterable, Any
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import haiku as hk
from aquaregia.utils import Array, ArrayTree
import jax_md
MLPFn = Callable[[ArrayTree], ArrayTree] # default MLP returned from a fn

# some defaults for fully-connected networks
DEFAULT_DISPLACEMENT_FN, DEFAULT_SHIFT_FN = jax_md.space.free()
DEFAULT_METRIC_FN = jax_md.space.canonicalize_displacement_or_metric(DEFAULT_DISPLACEMENT_FN)
DEFAULT_VDISPLACEMENT_FN = jax.vmap(jax.vmap(DEFAULT_DISPLACEMENT_FN, in_axes=(None,0)), in_axes=(0, None))
DEFAULT_VMETRIC_FN = jax.vmap(jax.vmap(DEFAULT_METRIC_FN, in_axes=(None,0)), in_axes=(0, None))

DEFAULT_EPSILON = 1e-8

def get_levi_civita():
    import numpy as np
    eijk_ = np.zeros((3,3,3))
    eijk_[0,1,2] = eijk_[1,2,0] = eijk_[2,0,1] = 1.
    eijk_[0,2,1] = eijk_[2,1,0] = eijk_[1,0,2] = -1.
    return Array(eijk_)

LEVI_CIVITA_TENSOR = get_levi_civita()
M_to_L_dict = {1: 0, 3: 1, 5: 2, 7: 3}
L_to_M_dict = {val: key for key, val in M_to_L_dict.items()}

def mask_tensor(tensor, mask_val = 0.):
    """TODO : make this more expressive than just masking a diagonal"""
    _fold_mask = jnp.invert(jnp.eye(tensor.shape[0], dtype=jnp.bool_))
    mask = jnp.repeat(_fold_mask[..., jnp.newaxis], repeats=tensor.shape[-1], axis=-1)
    return jnp.where(mask, tensor, mask_val)

def check_channel_shape(filter_channel_size, layer_input_channel_size):
    if filter_channel_size != layer_input_channel_size:
        raise ValueError(f"the filter channel size ({filter_channel_size}) must be equal to the layer input channel size ({layer_input_channel_size})")

def check_combination_dict(combination_dict):
    for in_L, filter_L_dict in combination_dict.items():
        for filter_L, out_L_list in filter_L_dict.items():
            lower_bound = jnp.abs(in_L - filter_L)
            upper_bound = in_L + filter_L
            for out_L in out_L_list:
                if out_L < lower_bound: raise ValueError(f"out_L ({out_L}) is outside the lower bound {lower_bound}")
                if out_L > upper_bound: raise ValueError(f"out_L ({out_L}) is outside the upper bound {upper_bound}")

def unit_vectors_and_norms(r_ij, epsilon = DEFAULT_EPSILON):
    """special function to normalize and compute nan-safe norm for a displacement matrix"""
    r_ij_squares = jnp.square(r_ij)
    sums = jnp.sum(r_ij_squares, axis=-1, keepdims=True)
    norms = jnp.sqrt(jnp.maximum(sums, epsilon**2))
    return r_ij / norms, norms

def Y_0(r_ij): #propto unity (scalar)
    return 1.

def Y_1(r_ij): #propto r_ij (3vec)
    return r_ij

def Y_2(r_ij, epsilon = DEFAULT_EPSILON):
    """
    this is a near copy;
    TODO: check this works?
    """
    x,y,z = r_ij
    r2 = jnp.maximum(jnp.square(r_ij).sum(), epsilon)
    output = Array([x*y/r2, y*z/r2, (-jnp.square(x)-jnp.square(y) + 2. * jnp.square(z))/(2 * jnp.sqrt(3.) * r2), z*x/r2, (jnp.square(x) - jnp.square(y)) / (2. * r2)])
    return output

"""Abstract MLP"""
class R(hk.Module):
    """
    radial function `R_c^(l_f, l_i)(r)`implemented as an MLP that returns an Array of shape [N,N,output_dim]
    """
    def __init__(self, mlp_constructor_dict, name=None):
        # mlp_constructor_dict = {'output_sizes': [8,8,8], 'activation': jax.nn.swish} for example
        super().__init__(name=name)
        self._mlp = hk.nets.MLP(**mlp_constructor_dict)

    def __call__(self, inputs):
        return self._mlp(inputs = inputs)

"""Radial MLPs"""
class SinusoidalBasis(hk.Module):
    def __init__(self,
                 r_switch,
                 r_cut,
                 basis_init = hk.initializers.Constant(constant=jnp.linspace(1., 8., 8)),
                 name="SinusoidBasis"):
        from aquaregia.utils import polynomial_switching_fn
        super().__init__(name=name)
        self._r_switch = r_switch
        self._r_cut = r_cut
        self._basis_init = basis_init
        self._output_size = len(basis_init.constant)

        # parital the polynomial switch function
        poly_switch = partial(polynomial_switching_fn, r_cutoff = r_cut, r_switch = r_switch)

        def singular_B(r_ij, N_bases, epsilon): # check this fn
            unscaled_out = (2. / r_cut) * jnp.sin(jnp.einsum('ijk,k->ijk', jnp.pi * r_ij / r_cut, N_bases))
            nansafe_r_ij = jnp.where(r_ij >= epsilon, r_ij, epsilon)
            scaled_out = poly_switch(nansafe_r_ij) * unscaled_out / nansafe_r_ij
            return scaled_out

        self._singular_B = singular_B

    def __call__(self, r_ij, epsilon = DEFAULT_EPSILON, mask_val = 0.):
        r_ij_shape = r_ij.shape
        dtype = r_ij.dtype

        if len(r_ij_shape) != 2:
            raise ValueError(f"the radial matrix input shape must be an N x N matrix")
        elif r_ij_shape[0] != r_ij_shape[1]:
            raise ValueError(f"the radial matrix input must be N x N; got {r_ij_shape}")

        basis_init = self._basis_init
        b = hk.get_parameter("b", [self._output_size], dtype=dtype, init = basis_init)
        bases = self._singular_B(r_ij = r_ij[..., jnp.newaxis], N_bases = b, epsilon=epsilon)
        return mask_tensor(bases, mask_val=mask_val)

"""Filter Module"""
class F(hk.Module):
    """filter module"""
    def __init__(
        self,
        L,
        R_mlp_constructor_dict,
        mask_output = False,
        switching_fn = None):
        # sanity check for L
        if L not in set(list(L_to_M_dict.keys())):
            raise ValueError(f"L = {L} is not an implemented filter")

        # super the hk.Module
        super().__init__(name=f"F_{L}")
        self._R = R(mlp_constructor_dict = R_mlp_constructor_dict)
        self._L = L
        self._mask_output = mask_output
        self._switching_fn = switching_fn

        # handle different L cases
    def __call__(self, inputs, unit_vectors, r_ij = None, epsilon = DEFAULT_EPSILON):
        radial = self._R(inputs=inputs) # execute vanilla MLP on radial input

        if (self._switching_fn is not None) and (r_ij is not None):
            # if both are passed as not NoneType objects, apply a final radial activation given by the polynomial switching fn
            if r_ij.shape != radial.shape[:-1]: # check shape
                raise ValueError(f"the r_ij input expects a shape of {radial.shape[:-1]} but received a shape of {r_ij.shape}")
            r_ij = r_ij[..., jnp.newaxis]
            nansafe_r_ij = jnp.where(r_ij >= epsilon, r_ij, epsilon)
            radial = self._switching_fn(nansafe_r_ij) * radial

        if not self._mask_output: # if we are not going to mask the output, one must mask the radial output...
            radial = mask_tensor(radial)
        if self._L == 0:
            _out = jnp.expand_dims(radial, axis=-1)
        elif self._L == 1:
            Y_1s = jnp.apply_along_axis(Y_1, arr=unit_vectors, axis=-1) # this function call is a formality
            _out = jnp.expand_dims(Y_1s, axis=-2) * jnp.expand_dims(radial, axis=-1) # [N,N,output_dim, 3]
        else:
            # TODO : implement l=2 and maybe more
            raise ValueError(f"L=2 is not yet tested; please implement and test this in the future!")

        if self._mask_output:
            # the `_out` var is of the shape [N,N,channel,L_to_M[L]], we want to mask the diagonal
            tensor_mask = jnp.tile(jnp.invert(jnp.eye(5, dtype=jnp.bool_))[..., jnp.newaxis, jnp.newaxis], (1,1,_out.shape[2], _out.shape[3]))
            _out = jnp.where(tensor_mask, _out, 0.)
        return _out

def retrieve_filter(in_L, filter_L, out_L):
    """function that will generate a module"""

    def check_inL(_in_L):
        if _in_L != in_L:
            raise ValueError(f"the input_L ({_in_L}) does not match the filter input L specification ({in_L})")

    def check_filterL(_filter_L):
        if _filter_L != filter_L:
            raise ValueError(f"the filter_L ({_filter_L}) does not match the filter L specification ({filter_L})")

    def check_outL(_out_L):
        if _out_L != out_L:
            raise ValueError(f"the out_L ({_out_L}) does not match the out_L specification ({out_L})")


    def combine(layer_input, F_out):
        neighbor_multiplier = 1. / jnp.sqrt(layer_input.shape[0] - 1) # normalize out by sqrt of number of neighbors
        input_dim = layer_input.shape[-1]
        if filter_L == 0: #L x 0 -> L is scalar multiplication of vector
            cg = jnp.expand_dims(jnp.eye(input_dim), axis=-2) * neighbor_multiplier # divide by the sqrt of the number of neighbors
            out = jnp.einsum('ijk,abfj,bfk->afi', cg, F_out, layer_input) # there can only be one output for l_i=L, l_f=0, and that is l_o = L, so the output has  dim [N,output_dim,2L+1]
        elif filter_L == 1: # the allowable outputs are L x 1 -> {|L-1|, L+1}
            if in_L == 0: # the only allowable out_L is 0 x 1 -> 1; scalar multiplication of vector
                cg = jnp.expand_dims(jnp.eye(3), axis=-1) * neighbor_multiplier
                out = jnp.einsum('ijk,abfj,bfk->afi', cg, F_out, layer_input)
            elif in_L == 1: # the allowable outputs are 1 x 1 -> {0,1,2}, though 2 is not implemented at the moment
                if out_L == 0:
                    cg = jnp.expand_dims(jnp.eye(3), axis=0) * neighbor_multiplier
                    out = jnp.einsum('ijk,abfj,bfk->afi', cg, F_out, layer_input)
                elif out_L == 1:
                    out = jnp.einsum('ijk,abfj,bfk->afi', LEVI_CIVITA_TENSOR * neighbor_multiplier, F_out, layer_input)
                elif out_L == 2:
                    raise NotImplementedError(f"1 x 1 -> 2 is not yet implemented")
                else:
                    raise ValueError(f"out_L = {out_L} is not supported by 1 x 1")
            elif in_L == 2:
                raise NotImplementedError(f"2 x 1 -> {1,2,3} is not currently implemented")
        return out

    class FilterXtoY(hk.Module):
        def __init__(self, R_mlp_constructor_dict, mask_output, switching_fn):
            super().__init__(name=f"In{in_L}_F{filter_L}_to_{out_L}")
            self._F = F(L = filter_L, R_mlp_constructor_dict=R_mlp_constructor_dict, mask_output=mask_output, switching_fn=switching_fn)

        def __call__(self, layer_input, rbf_inputs, unit_vectors, r_ij = None, epsilon = DEFAULT_EPSILON):
            _ = check_inL(M_to_L_dict[layer_input.shape[-1]])
            F_out = self._F(inputs=rbf_inputs, unit_vectors=unit_vectors, r_ij=r_ij, epsilon=epsilon) # gives [N, N, output_dim, 2L+1]
            _ = check_filterL(M_to_L_dict[F_out.shape[-1]])
            _ = check_channel_shape(filter_channel_size = F_out.shape[-2], layer_input_channel_size = layer_input.shape[-2])
            out = combine(layer_input, F_out)
            _ = check_outL(M_to_L_dict[out.shape[-1]])
            return out
    return FilterXtoY

class Convolution(hk.Module):
    """
    convolve a tensor field
    """
    def __init__(self,
                 filter_mlp_dicts,
                 name = None,
                 mask_output = False,
                 switching_fn = None,
                 combination_dict = {0: {0: [0], 1: [1]},
                                    1: {0: [1], 1: [0,1]}} # organized as {in_L: {filter_L: [out_Ls]}}
                 ):
        super().__init__(name=name)
        # create the necessary convolutions...
        filter_dict = {in_L:
                            {filter_L:
                             {out_L:
                              retrieve_filter(in_L = in_L, filter_L = filter_L, out_L = out_L)(R_mlp_constructor_dict = filter_mlp_dicts[filter_L], mask_output = mask_output, switching_fn = switching_fn) for out_L in out_Ls
                             } for filter_L, out_Ls in filter_Ls.items()
                            } for in_L, filter_Ls in combination_dict.items()
                           }

        self._filter_dict = {in_L: jax.tree_util.tree_leaves(entry) for in_L, entry in filter_dict.items()} #flatten the dict
        self._template_dict = {in_L : list(set(jax.tree_util.tree_leaves(_val))) for in_L, _val in combination_dict.items()}

        self._expected_input_Ls = set(list(self._template_dict.keys())) # template_dict's expected input Ls
        self._expected_output_Ls = set([item for sublist in self._template_dict.values() for item in sublist]) # template_dict's expected output Ls

    def __call__(self, in_tensor_dict, rbf_inputs, unit_vectors, r_ij, epsilon = DEFAULT_EPSILON):
        given_input_Ls = set(list(in_tensor_dict.keys()))
        if not self._expected_input_Ls.issubset(given_input_Ls): raise ValueError(f"the prespecified input angular numbers is not a subset of the given input angular numbers")

        out_nest_dict = {_L : None for _L in self._expected_output_Ls} # the output is given _only_ by the specified outputs
        for _L in self._expected_input_Ls:
            _tensor = in_tensor_dict[_L]
            _output_Ls = set(self._template_dict[_L])
            # if _tensor is not None: raise ValueError(f"`None` tensors are not supported")
            if _tensor is not None:
                guard_output_Ls = self._expected_output_Ls.difference(_output_Ls)
                out_tensors = [_function(layer_input = _tensor,
                                         rbf_inputs = rbf_inputs,
                                         unit_vectors = unit_vectors,
                                         r_ij = r_ij,
                                         epsilon = epsilon) for _function in self._filter_dict[_L]]
                out_nest_dict[_L] = {__L : jnp.concatenate([_tens for _tens in out_tensors if M_to_L_dict[_tens.shape[-1]] == __L], axis=-2) for __L in _output_Ls}
                out_nest_dict[_L].update({__L : None for __L in guard_output_Ls})
            else:
                out_nest_dict[_L] = None
                # out_nest_dict[_L] = {__L : None for __L in _output_Ls}

        out_concat_dict = {__L: jnp.concatenate([_dct[__L] for _dct in out_nest_dict.values() if _dct is not None], axis=-2) for __L in self._expected_output_Ls}

        return out_concat_dict

"""Tensor MLP constructors"""

class TensorLinear(hk.Module):
    """create a self interaction layer (linear)"""
    def __init__(self,
                 output_size: int,
                 L : int,
                 w_init : Optional[hk.initializers.Initializer] = None,
                 b_init : Optional[hk.initializers.Initializer] = None,
                 name : Optional[str] = None):
        super().__init__(name=name)
        self.L = L
        self.input_size = None
        self.output_size = output_size
        self.with_bias = (L == 0)
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros

    def __call__(self,
                 inputs : Array,
                 *,
                 precision : Optional[jax.lax.Precision] = None,
                 ) -> Array:
        if not inputs.shape:
            raise ValueError("Inputs must not be a scalar.")
        input_size = self.input_size = inputs.shape[-2]
        output_size = self.output_size
        dtype = inputs.dtype
        w_init = self.w_init

        # get weights
        if w_init is None:
            w_init = hk.initializers.Orthogonal()
        w = hk.get_parameter("w", [output_size, input_size], dtype, init=w_init)

        # get bias
        if self.with_bias:
            b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
        else:
            b = jnp.zeros(output_size)
        b = jnp.broadcast_to(b, (inputs.shape[0], L_to_M_dict[self.L], self.output_size))

        out = jnp.transpose(jnp.einsum('afi,gf->aig', inputs, w, precision=precision) + b, axes=(0,2,1))
        return out

class TensorNonLinear(hk.Module):
    def __init__(self,
                 L : int,
                 nonlinearity : Optional[Callable] = jax.nn.swish,
                 name : Optional[str] = None,
                 b_init : Optional[hk.initializers.Initializer] = None,
                ):
        super().__init__(name=name)
        self.L = L
        self.with_bias = (L != 0)
        self.b_init = b_init or jnp.zeros
        self.nonlin = nonlinearity

    def __call__(self,
                 inputs : Array,
                 epsilon : Optional[float] = DEFAULT_EPSILON
                 ):
        shape = inputs.shape
        representation_index = shape[-1]
        channels = shape[-2]
        dtype = inputs.dtype

        if self.with_bias:
            b = hk.get_parameter("b", [channels], dtype, init = self.b_init)
            b = jnp.broadcast_to(b, inputs.shape[:-1])
            norm_across_m = jnp.linalg.norm(inputs, axis=-1, keepdims=False) # take the euclidean norm across the last dim
            norm_across_m = jnp.where(norm_across_m <= epsilon, norm_across_m + epsilon, norm_across_m) # fix with epsilon
            nonlin_out = self.nonlin(norm_across_m + b)
            factor = nonlin_out / norm_across_m
            nonlin_out = inputs * factor[..., jnp.newaxis]
        else:
            nonlin_out = self.nonlin(inputs)
        return nonlin_out

"""Tensor Field MLPs"""

class TensorFieldMLP(hk.Module):
    """an implementation of `hk.nets.MLP` which respects equivariant w.r.t. tensors"""
    def __init__(self,
                 output_sizes : Iterable[int],
                 L : int,
                 name : Optional[str] = None,
                 w_init : Optional[hk.initializers.Initializer] = None,
                 b_init : Optional[hk.initializers.Initializer] = None,
                 nonlinearity : Optional[Callable] = jax.nn.swish
                 ):
        super().__init__(name=name)
        layers = []
        output_sizes = tuple(output_sizes)
        for index, output_size in enumerate(output_sizes):
            linear_layer = TensorLinear(output_size = output_size,
                                        L = L,
                                        w_init = w_init,
                                        b_init = b_init,
                                        name = f"linear_{index}"
                                       )
            if index != (len(output_sizes) - 1): # we pass a nonlinearity
                nonlinear_layer = TensorNonLinear(L = L,
                                                  nonlinearity = nonlinearity,
                                                  name = f"nonlinear_{index}",
                                                  b_init = b_init)
            else: # we do not pass a nonlinearity
                nonlinear_layer = lambda _x, _epsilon : _x

            layers.append((linear_layer, nonlinear_layer))
            self.layers = tuple(layers)

    def __call__(self,
                 inputs : Array,
                 epsilon : Optional[float] = DEFAULT_EPSILON):
        out = inputs
        for index, layer in enumerate(self.layers):
            linear_layer, nonlinear_layer = layer # pull apart the tuple
            linear_out = linear_layer(inputs = out) # pass the linear layer
            out = nonlinear_layer(linear_out, epsilon) # pass the nonlinear layer, be sure to call it `out`
        return out
