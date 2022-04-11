#!/usr/bin/env python
# coding: utf-8

# # 3D Harmonic CNF

# In[ ]:


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
from typing import Iterable, Optional
from aquaregia import tfn


# test on a 3d system...harmonic spring.

# In[ ]:


from aquaregia.tests.test_integrators import get_diatom_parameters_dict, get_diatom_equilibrium_cache, DEFAULT_TEMPERATURE, DEFAULT_TIMESTEP, kT
from aquaregia.tests.test_fec import diatom_free_energy 
from jax.example_libraries.optimizers import adam


# In[ ]:


kT


# In[ ]:


init_diatom_parameters_dict = get_diatom_parameters_dict()
final_diatom_parameters_dict = get_diatom_parameters_dict()


# In[ ]:


init_diatom_parameters_dict['u_params']


# In[ ]:


final_diatom_parameters_dict['u_params']['HarmonicBondForce']['length'] = Array([0.])


# let's make a neural network for the diatom.

# In[ ]:


positions = jax.random.normal(jax.random.PRNGKey(261), shape=(2,3))
velocities = jax.random.normal(jax.random.PRNGKey(235), shape=(2,3))

positions_and_velocities = jnp.vstack([positions[jnp.newaxis, ...], velocities[jnp.newaxis, ...]])
feature_dict = { 0:random.normal(jax.random.PRNGKey(243), shape=(2,2,1)), 1: None}


# In[ ]:


from aquaregia.cnf import VectorModule, make_diff_fn_inits
from aquaregia.integrators import thermalize


# In[ ]:


VectorMLP_kwargs = make_diff_fn_inits(feature_dictionary=feature_dict)


# we need to generate a cache of training data and a velocity sampler

# In[ ]:


# logps
thermalizer = partial(thermalize, masses = init_diatom_parameters_dict['masses'], kT = kT, dimension=3)
ke = partial(kinetic_energy, mass = init_diatom_parameters_dict['masses'])

prior_logp = lambda y : -init_diatom_parameters_dict['u_fn'](y[0], init_diatom_parameters_dict['neighbor_list'], init_diatom_parameters_dict['u_params']) - ke(y[1])
posterior_logp = lambda y : -final_diatom_parameters_dict['u_fn'](y[0], final_diatom_parameters_dict['neighbor_list'], final_diatom_parameters_dict['u_params']) - ke(y[1])


# In[ ]:


# prior_data
prior_displacements, prior_kes, prior_posits = get_diatom_equilibrium_cache(seed=jax.random.PRNGKey(45), _dict=init_diatom_parameters_dict, num_samples=200, steps_per_sample=1000)
posterior_displacements, posterior_kes, posterior_posits = get_diatom_equilibrium_cache(seed=jax.random.PRNGKey(43), _dict=final_diatom_parameters_dict, num_samples=200, steps_per_sample=1000)


# In[ ]:


from aquaregia.cnf import KinematicCNFFactory
import diffrax


# In[ ]:


factory = KinematicCNFFactory(
                 support_dimension = (2,3), # this is doubled to account for velocity
                 logp_prior = prior_logp,
                 logp_posterior = posterior_logp,
                 prior_train_data = prior_posits[:100],
                 velocity_sampler = thermalizer,
                 posterior_train_data = posterior_posits[:100],
                 diff_func_module = VectorModule, # this module needs to return a
                 diff_func_builder_kwargs = VectorMLP_kwargs,
                 batch_size = 4,
                 optimizer = adam,
                 optimizer_kwargs = {'step_size': 1e-3},
                 prior_validate_data = prior_posits[100:110],
                 posterior_validate_data = posterior_posits[110:110],
                 stepsize_controller_module = diffrax.PIDController,
                 stepsize_controller_module_kwarg_dict = {'rtol':1e-3, 'atol':1e-6},
)


# In[ ]:


factory


# In[ ]:


init_y = factory._prior_train_sampler(jax.random.PRNGKey(265))


# In[ ]:


util_fns = factory.get_utils(jax.random.PRNGKey(253), regulariser_lambda=0.)


# In[ ]:


init_params = util_fns['init_parameters']
pretraining_fn = util_fns['pretrain_fn']
train_fn = util_fns['train_fn']


# In[ ]:


canonical_diff_fn = util_fns['canonical_differential_fn']


# In[ ]:


ode_solver = util_fns['ode_solver']


# In[ ]:


forward_loss_fn = util_fns['forward_loss_fn']
backward_loss_fn = util_fns['backward_loss_fn']


# In[ ]:


train_fn(init_params, jax.random.PRNGKey(2346), num_iters=1000, clip_gradient_max_norm=1e1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


sol = ode_solver(init_params, init_y, jax.random.PRNGKey(253), 0., 1.)


# In[ ]:


forward_loss_fn(init_params, init_y, jax.random.PRNGKey(56), 0.)


# In[ ]:


backward_loss_fn(init_params, init_y, jax.random.PRNGKey(56), 0.)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


pretrained_opt_state, pretrain_vals = pretraining_fn(init_params, jax.random.PRNGKey(34), num_iters=1000)


# In[ ]:




