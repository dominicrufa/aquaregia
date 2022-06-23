"""
simple realnvp
"""

from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from functools import partial
import haiku as hk
from aquaregia.utils import Array, ArrayTree, polynomial_switching_fn, kinetic_energy
import diffrax
from jax.example_libraries.optimizers import adam
from typing import Iterable, Optional, Callable, Sequence
from aquaregia import tfn


class VectorModule(hk.Module):
    """
    a vector module for SE(3) equivariance
    """
    def __init__(self,
                 feature_dictionary : dict, # this is permanent
                 conv_dict_list : Iterable,
                 mlp_dict_list : Iterable,
                 SinusoidalBasis_kwargs : dict,
                 time_convolution_kwargs : dict,
                 epsilon : Optional[float] = tfn.DEFAULT_EPSILON,
                 mask_value : Optional[float] = 0.,
                 scalar_multiplier_update : Optional[bool] = True,
                 name : Optional[str] = f"vector"):
        super().__init__(name=name)
        self.SinusoidalBasis_module = tfn.SinusoidalBasis(**SinusoidalBasis_kwargs) # define sinusoid

        num_particles, input_L0_channel_dimension = feature_dictionary[0].shape[:2]
        self._num_particles = num_particles
        self._input_L0_channel_dimension = input_L0_channel_dimension
        self._epsilon = epsilon
        self._mask_value = mask_value
        self._scalar_multiplier_update = scalar_multiplier_update
        self._feature_dictionary = feature_dictionary


        if self._feature_dictionary[0].shape != (self._num_particles, self._input_L0_channel_dimension, 1):
            raise ValueError(f"""the feature dictionary's L=0 input ({feature_dictionary[0].shape})
                                 is not equal to the specified channel dimension
                                 ({(self._num_particles, self._input_L0_channel_dimension, 1)})""")
        elif self._feature_dictionary[1] is not None:
            raise ValueError(f"the input feature dictionary should have no annotations on the L=1 input")

        # time convolution
        self.time_convolution_module = hk.nets.MLP(**time_convolution_kwargs)

        # setup the for loop
        layers = {}
        num_layers = len(conv_dict_list)
        for layer in range(num_layers):
            conv_module = tfn.Convolution(**conv_dict_list[layer])
            tfn_module_dict = {_L: tfn.TensorFieldMLP(**mlp_dict_list[layer][_L]) for _L in mlp_dict_list[layer].keys()}
            layers[layer] = {'conv': conv_module, 'mlp': tfn_module_dict}
        self.layers = layers

    def __call__(self,
                 t : float,
                 positions : Array,
                 velocities : Array,
                 ):
        if positions.shape[0] != self._num_particles:
            raise ValueError(f"the number of positions ({positions}) is not equal to the number of particles")
        if velocities is not None:
            if velocities.shape[0] != self._num_particles:
                raise ValueError(f"the number of positions ({positions}) is not equal to the number of particles")
            if velocities.shape[-1] != 3:
                raise ValueError(f"the velocity must satisfy a 3-dimensional constraint.")

        r_ij = tfn.DEFAULT_VDISPLACEMENT_FN(positions, positions)
        unit_r_ij, norms = tfn.unit_vectors_and_norms(r_ij) # compute unit vectors and norms
        norms = tfn.mask_tensor(norms, mask_val = self._mask_value)
        norms = jnp.squeeze(norms)

        rbf_inputs = self.SinusoidalBasis_module(r_ij = norms, epsilon=self._epsilon, mask_val=self._mask_value)

        # concat feature dictionary with time convolutions.
        time_convolution = self.time_convolution_module(Array([t]))
        repeated_time_convolution = jnp.repeat(time_convolution[jnp.newaxis, ..., jnp.newaxis], repeats=self._num_particles, axis=0)
        aug_L0 = jnp.hstack([self._feature_dictionary[0], repeated_time_convolution])
        in_tensor_dict = {L : _tensor for L, _tensor in self._feature_dictionary.items()}
        in_tensor_dict[0] = aug_L0
        # do we want to allow this annotation?
        in_tensor_dict[1] = velocities

        for layer_idx in range(len(self.layers)):
            out_tensor_dict = {}
            layer_dict = self.layers[layer_idx]
            conv_dict = layer_dict['conv'](in_tensor_dict=in_tensor_dict, rbf_inputs=rbf_inputs, unit_vectors=unit_r_ij, r_ij = norms, epsilon = self._epsilon)
            for _L in conv_dict.keys(): #iterate over the mlps/angular numbers
                mlp = layer_dict['mlp'][_L]
                p_array = mlp(inputs=conv_dict[_L], epsilon=self._epsilon) # pass convolved arrays through mlp
                out_tensor_dict[_L] = p_array # populate out dict
            in_tensor_dict = out_tensor_dict

        scales = jnp.repeat(in_tensor_dict[0][:,0,:], repeats=3, axis=-1)
        translations = in_tensor_dict[1][:,0,:]
        #print(f"scales: {scales}")
        return scales, center_coordinate(translations)


class TimeIncrementMLP(hk.Module):
    """
    mlp that returns a timestep float from the (time, velocity) as an input
    """
    def __init__(self,
                 output_sizes : Optional[Sequence] = [4,4,1],
                 activation : Optional[Callable] = jax.nn.swish,
                 name : Optional[str] = None,
                 **kwargs):
        super().__init__(name=name)
        self._mlp = hk.nets.MLP(output_sizes=output_sizes, activation=activation)

    def __call__(self, t, v):
        v2 = v.dot(v)
        _input = jnp.array([t, v2])
        return self._mlp(_input)**2


class RNVPModule(hk.Module):
    """
    wrapper to run the forward/backward RNVP module
    """
    def __init__(self,
                 VectorModule_kwargs : dict,
                 num_iters : int,
                 TimeIncrementMLP_kwargs : Optional[dict] = {},
                 name : Optional[str] = f"RNVP"):
        super().__init__(name=name)
        self._num_iters = num_iters
        self._time_vector = jnp.linspace(0., 1., num_iters)
        self._VectorModule = VectorModule(**VectorModule_kwargs)
        self._TimeIncrementMLP = TimeIncrementMLP(**TimeIncrementMLP_kwargs)

        def velocity_update(x, v, t, forward_bool):
            log_scales, translations = self._VectorModule(t, x, None)
            logdetJ = log_scales.sum()
            if forward_bool:
                out_velocities = v * jnp.exp(log_scales) + translations
            else:
                out_velocities = (v - translations) * jnp.exp(-log_scales)
            return out_velocities, logdetJ

        self._velocity_update = velocity_update

        def position_update(x, v, t, forward_bool):
            timesteps = hk.vmap(self._TimeIncrementMLP, in_axes=(None, 0))(t, v)
            if forward_bool:
                out_positions = x + timesteps * v
            else:
                out_positions = x - timesteps * v
            return out_positions, 0. # logdetJ is zero

        self._position_update = position_update

    def __call__(self, in_x, in_v, forward_bool):
        logdetJ = 0.
        if forward_bool:
            time_vector = self._time_vector
        else:
            time_vector = self._time_vector[::-1]

        for t in time_vector:
            if forward_bool:
                in_v, v_logdetJ = self._velocity_update(in_x, in_v, t, forward_bool=True)
                in_x, x_logdetJ = self._position_update(in_x, in_v, t, forward_bool=True)
                logdetJ = v_logdetJ + x_logdetJ + logdetJ
            else:
                in_x, x_logdetJ = self._position_update(in_x, in_v, t, forward_bool=False)
                in_v, v_logdetJ = self._velocity_update(in_x, in_v, t, forward_bool=False)
                logdetJ = v_logdetJ + x_logdetJ + logdetJ
            # print(f"x_logdetJ: {x_logdetJ}")
            # print(f"logdetJ: {logdetJ}")
        return in_x, in_v, logdetJ

def center_coordinate(_three_coordinate):
    return _three_coordinate - _three_coordinate.mean(axis=0)[jnp.newaxis, ...]

def build_velocity_augmented_sampler(data, velocity_sampler, center_data, center_velocity):
    num_datapoints, data_dimension = data.shape[0], data.shape[1:]
    def sampler(key):
        idx_key, v_key = jax.random.split(key)
        index = jax.random.choice(idx_key, jnp.arange(num_datapoints, dtype=jnp.int32))
        velocities = jax.lax.cond(center_velocity, lambda _key : center_coordinate(velocity_sampler(_key)), lambda _key : velocity_sampler(_key), v_key)
        out_data = jax.lax.cond(center_data, lambda _index: center_coordinate(data[_index]), lambda _index: data[_index], index)
        return out_data, velocities
    return sampler


class NFFactory(object):
    def __init__(self,
                 position_dimension,
                 logp_position_posterior,
                 posterior_train_samples,
                 posterior_validate_samples,
                 RNVPModule_kwargs
                 ):
        from functools import partial
        from aquaregia.utils import kinetic_energy
        from aquaregia.integrators import thermalize

        self._position_dimension = position_dimension
        self._thermalizer = partial(thermalize, masses = jnp.ones(self._position_dimension[0]), kT=1., dimension=3)
        self._zero_center_thermalizer = lambda _key: center_coordinate(self._thermalizer(_key))
        self._kinetic_energy = partial(kinetic_energy, mass = jnp.ones(self._position_dimension[0]))
        self._logp_position_posterior = logp_position_posterior
        self._posterior_train_samples = posterior_train_samples
        self._posterior_validate_samples = posterior_validate_samples
        self._posterior_sampler = build_velocity_augmented_sampler(posterior_train_samples, self._thermalizer, False, False)
        def prior_sampler(key):
            x_key, v_key = jax.random.split(key)
            return self._zero_center_thermalizer(x_key), self._zero_center_thermalizer(v_key)
        self._prior_sampler = prior_sampler

        # make `u`s (-logp_prior/-logp_posterior)
        self._u_prior = lambda _x, _v : self._kinetic_energy(_x) + self._kinetic_energy(_v)
        self._u_posterior = lambda _x, _v : -self._logp_position_posterior(_x) + self._kinetic_energy(_v)

        # make rnvp module
        def _wrapper(x, v, forward_bool): # outputs (x, v, logdetJ)
            return RNVPModule(**RNVPModule_kwargs)(x, v, forward_bool)
        rnvp_init, rnvp_apply = hk.without_apply_rng(hk.transform(_wrapper))
        self._rnvp_init = rnvp_init
        self._rnvp_apply = rnvp_apply

        # make loss function
        def loss_function(params, x, v, forward_bool):
            # loss is KL-divergence
            u_start = jax.lax.cond(forward_bool,
                                   lambda _x, _v: self._u_prior(_x, _v),
                                   lambda _x, _v: self._u_posterior(_x, _v),
                                   x, v)
            out_x, out_v, logdetJ = self._rnvp_apply(params, x, v, forward_bool)
            out_x, out_v = jax.lax.cond(forward_bool,
                                        lambda _x, _v: (_x, _v), # do not center posterior samples
                                        lambda _x, _v: (center_coordinate(_x), center_coordinate(_v)), # center prior samples
                                        out_x, out_v)
            u_end = jax.lax.cond(forward_bool,
                                 lambda _x, _v: self._u_posterior(_x, _v),
                                 lambda _x, _v: self._u_prior(_x, _v),
                                 out_x, out_v)
            lossy_logdetJ = jax.lax.cond(forward_bool, lambda _in: -_in, lambda _in: _in, logdetJ)
            loss = u_end - u_start + lossy_logdetJ
            return loss

        self._loss_function = loss_function
        self._backward_batch_loss_function = lambda _params, _xs, _vs: jax.vmap(self._loss_function, in_axes=(None,0,0,None))(_params, _xs, _vs, False)
        self._forward_batch_loss_function = lambda _params, _xs, _vs: jax.vmap(self._loss_function, in_axes=(None,0,0,None))(_params, _xs, _vs, True)
        self._train_loss_function = lambda _params, _xs, _vs: self._backward_batch_loss_function(_params, _xs, _vs).mean()


    def train(self, key, batch_size, optimizer, optimizer_kwargs, clip_grad_max_norm, num_iters):
        """
        train by transforming data into noise
        """
        import tqdm
        import time
        assert batch_size < self._posterior_train_samples.shape[0], f"the batch size must be less than the amount of data"
        init_fun, update_fun, get_params = optimizer(**optimizer_kwargs)

        # init
        init_key, sample_key, key = jax.random.split(key, num=3)
        init_params = self._rnvp_init(init_key, *self._posterior_sampler(sample_key), forward_bool=False)

        # make step function
        @jax.jit
        def step(key, _iter, opt_state):
            in_params = get_params(opt_state)
            xs, vs = jax.vmap(self._posterior_sampler)(jax.random.split(key, num=batch_size))
            mean_val, param_grads = jax.value_and_grad(self._train_loss_function)(in_params, xs, vs)
            new_opt_state = update_fun(_iter, jax.example_libraries.optimizers.clip_grads(param_grads, clip_grad_max_norm), opt_state)
            return mean_val, new_opt_state

        def _train(init_params, key):
            train_vals = []
            opt_state = init_fun(init_params)
            trange = tqdm.trange(num_iters, desc=f"Bar desc", leave=True)
            for i in trange:
                run_key, key = jax.random.split(key)
                try:
                    train_val, opt_state = step(run_key, i, opt_state)
                    train_vals.append(train_val)
                    trange.set_description(f"{train_val}")
                    trange.refresh()
                except Exception as e:
                    print(f"retrieved exception in training: {e}")
                    print(f"returning latest parameters...")
                    break
            return get_params(opt_state), np.array(train_vals)

        print(f"training...")
        out_params, train_vals = _train(init_params, key)
        return out_params, train_vals
