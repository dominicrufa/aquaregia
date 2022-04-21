#!/usr/bin/env python
# coding: utf-8

# generate droplet data

# In[1]:


from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True) 
config.parse_flags_with_absl()
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from functools import partial
import haiku as hk
from aquaregia.utils import Array, ArrayTree


# can we make a lj/WCA fluid droplet like in the targeted fep paper?

# In[2]:


WCA_sigma = 0.34
WCA_epsilon = 0.997736
mass = 39.9
from aquaregia.tfn import DEFAULT_EPSILON


# In[3]:


def get_droplet_potential(nparticles, 
                                 reduced_density, 
                                 wca_sigma, 
                                 wca_epsilon,
                                 harmonic_k,
                                 w_scale):
    import jax_md
    from openmmtools.testsystems import DoubleWellDimer_WCAFluid
    from jax import vmap
    from functools import partial
    from aquaregia.openmm import lifted_vacuum_lj
    from aquaregia.utils import get_periodic_distance_calculator, get_mask
    
    volume = nparticles * wca_sigma**3 / reduced_density
    select_length = (volume/(4*jnp.pi/3))**(1./3.)

    r_cutoff=100.
    r_switch=r_cutoff-1
    displacement_fn, shift_fn = jax_md.space.free()
    base_metric_fn = jax_md.space.canonicalize_displacement_or_metric(displacement_fn)
    vmetric = get_periodic_distance_calculator(base_metric_fn, r_cutoff)
    vsigma = vmap(vmap(lambda x, y : 0.5 * (x + y), in_axes = (None, 0)))
    vepsilon = vmap(vmap(lambda x, y : jnp.sqrt(x * y), in_axes = (None, 0)))
    
    #we actually don't want to lift these sterics yet.
    steric_fn = partial(lifted_vacuum_lj, r_cutoff=r_cutoff, r_switch=r_switch)
    
        # get the parameters right
    sigmas = Array([wca_sigma]*nparticles)
    epsilons = Array([wca_epsilon]*nparticles)
    nb_params = {'sigma': sigmas, 'epsilon': epsilons, 'w': jnp.ones(nparticles)*w_scale, 'k': harmonic_k, 'length': select_length}
    
    def harm_fn(dr, k, length):
        return jax.lax.cond(dr >= length, lambda _dr: 0.5*k*(dr-length)**2, lambda _dr: 0., dr)
    
    def energy_fn(R, neighbor_list, parameter_dict):
        #steric energy
        sigmas, epsilons, ws = parameter_dict['sigma'], parameter_dict['epsilon'], parameter_dict['w']
        augmented_R = jnp.concatenate([R, ws[..., jnp.newaxis]], axis=-1)
        drs = vmetric(augmented_R, neighbor_list) #compute lifted drs
        drs = jnp.where(drs <= DEFAULT_EPSILON, drs + DEFAULT_EPSILON, drs) # pad drs
        steric_energies = jnp.vectorize(steric_fn)(drs,
                                           vsigma(sigmas, sigmas[neighbor_list.idx]),
                                           vepsilon(epsilons, epsilons[neighbor_list.idx])
                                        )
        overfill_mask = get_mask(neighbor_list)
        energies = 0.5 * jnp.sum(jnp.where(overfill_mask, steric_energies, 0.))
        
        #harmonic energy
        dr_array = drs[0,1:]
        length, k = parameter_dict['length'], parameter_dict['k']
        harm_es = jnp.vectorize(partial(harm_fn, k=k, length=length))(dr_array)


        return energies + jnp.sum(harm_es)
    
    return nb_params, energy_fn, displacement_fn, shift_fn, r_cutoff, base_metric_fn


# In[4]:


num_particles = 10


# In[5]:


nb_params, energy_fn, displacement_fn, shift_fn, r_cutoff, base_metric_fn = get_droplet_potential(nparticles=num_particles, 
                                                                                                  reduced_density=.96,
                                                                                                  wca_sigma=WCA_sigma,
                                                                                                  wca_epsilon=WCA_epsilon,
                                                                                                  harmonic_k=121377.84,
                                                                                                  w_scale=1e-3)

mod_nb_params, _, _, _, _, _ = get_droplet_potential(nparticles=num_particles, 
                                                                                                  reduced_density=.96,
                                                                                                  wca_sigma=WCA_sigma,
                                                                                                  wca_epsilon=WCA_epsilon,
                                                                                                  harmonic_k=121377.84,
                                                                                                  w_scale=1e-3)


# In[6]:


new_length = (7*(WCA_sigma**3) + nb_params['length']**3)**(1./3.)


# In[7]:


new_length


# In[8]:


mod_nb_params['sigma'] = Array([WCA_sigma*2] + [WCA_sigma]*(num_particles-1))
mod_nb_params['length'] = new_length


# In[9]:


mod_nb_params


# In[10]:


#nb_params


# In[11]:


from aquaregia.utils import get_vacuum_neighbor_list


# In[12]:



nbr_lst = get_vacuum_neighbor_list(num_particles)
test_posits = Array(jax.random.normal(jax.random.PRNGKey(34), shape=(num_particles, 3)))


# In[13]:


#test_posits


# In[14]:


energy_fn(test_posits, nbr_lst, nb_params)


# In[15]:


from scipy.optimize import minimize


# In[16]:


@jax.jit
def mod_energy_fn(flat_xs, params):
    xs = flat_xs.reshape((num_particles,3))
    return energy_fn(xs, nbr_lst, params)


# In[17]:


unmod_res = minimize(mod_energy_fn, test_posits.flatten(), jac=jax.jit(jax.grad(mod_energy_fn)), args=(nb_params))
mod_res = minimize(mod_energy_fn, test_posits.flatten(), jac=jax.jit(jax.grad(mod_energy_fn)), args=(mod_nb_params))


# In[18]:


unmod_res.fun


# In[19]:


mod_res.fun


# In[20]:


assert unmod_res.success and mod_res.success


# In[21]:


unmod_in_xs = Array(unmod_res.x).reshape((num_particles,3))
mod_in_xs = Array(mod_res.x).reshape((num_particles, 3))


# In[ ]:





# In[22]:


from aquaregia.integrators import BAOABIntegratorGenerator


# In[23]:


from openmmtools.constants import kB
from simtk import unit
temperature = (0.824 * WCA_epsilon * unit.kilojoule_per_mole / kB).value_in_unit(unit.kelvin)
kT = (kB * temperature*unit.kelvin).value_in_unit_system(unit.md_unit_system)
reduced_density = 0.96 #as per the paper
tau = jnp.sqrt(WCA_sigma**2 * mass / WCA_epsilon)
timestep = 0.002 * tau
collision_rate = tau ** (-1)


# In[24]:


int_generator = BAOABIntegratorGenerator(canonical_u_fn = energy_fn,
                                         neighbor_list = nbr_lst,
                                         dt = timestep,
                                         masses = jnp.ones(num_particles)*mass,
                                         kT = kT,
                                         shift_fn = shift_fn,
                                         collision_rate=collision_rate)


# In[25]:


integrator = int_generator.integrator(30000)


# In[26]:


jax_int = jax.jit(partial(integrator, neighbor_list=nbr_lst))


# In[27]:


# unmod_out_dict = jax_int(unmod_in_xs, nb_params, jax.random.PRNGKey(346))
# mod_out_dict = jax_int(mod_in_xs, mod_nb_params, jax.random.PRNGKey(346))


# In[28]:


#all_unmod_outs = jax.vmap(jax_int, in_axes=(None, None, 0))(unmod_in_xs, nb_params, jax.random.split(jax.random.PRNGKey(5645), num=500))


# In[29]:


#all_mod_outs = jax.vmap(jax_int, in_axes=(None, None, 0))(mod_in_xs, mod_nb_params, jax.random.split(jax.random.PRNGKey(586), num=500))


# In[30]:


# unmod_energies = jax.vmap(energy_fn, in_axes=(0, None, None))(all_unmod_outs['xs'], nbr_lst, nb_params)
# mod_energies = jax.vmap(energy_fn, in_axes=(0, None, None))(all_mod_outs['xs'], nbr_lst, mod_nb_params)


# In[31]:


# from matplotlib import pyplot as plt


# In[32]:


# plt.hist(np.array(unmod_energies), alpha=0.5)
# plt.hist(np.array(mod_energies), alpha=0.5)


# In[33]:


# from pymbar import timeseries


# In[34]:


# [t0, g, Neff_max] = timeseries.detectEquilibration(np.array(mod_energies))


# In[35]:


# Neff_max


# compute forward works

# In[36]:


# forward_energies = jax.vmap(energy_fn, in_axes=(0, None, None))(all_unmod_outs['xs'], nbr_lst, mod_nb_params)


# In[37]:


# forward_works = (forward_energies - unmod_energies)/kT


# In[38]:


# plt.hist(np.array(forward_works))


# backward_ener

# In[39]:


# backward_energies = jax.vmap(energy_fn, in_axes=(0, None, None))(all_mod_outs['xs'], nbr_lst, nb_params)


# In[40]:


# backward_works = (backward_energies - mod_energies)/kT


# In[41]:


# plt.hist(np.array(forward_works), alpha=0.5)
# plt.hist(-np.array(backward_works), alpha=0.5)


# neat so those definitely aren't converged. can we make them do that with the `FlussFactory`?
# 

# but first, let's save this training data...

# In[42]:


# out_dict = {'unmod_positions' : np.array(all_unmod_outs['xs']),
#             'mod_positions' : np.array(all_mod_outs['xs'])}


# In[43]:


# np.savez(f"10p_droplet.data.npz", out_dict)


# In[ ]:





# In[ ]:





# In[ ]:





# now we can use fluss...

# In[77]:


in_dict = jnp.load('10p_droplet.data.npz', allow_pickle=True)['arr_0'].item()


# In[78]:


prior_positions = Array(in_dict['unmod_positions'])
posterior_positions = Array(in_dict['mod_positions'])


# In[79]:


from aquaregia.cnf import SequentialFlussFactory, SequentialVectorModule, make_diff_fn_inits, FlussFactory, VectorModule


# In[80]:


logp_prior = lambda _x : -energy_fn(_x, nbr_lst, nb_params)/kT
logp_posterior = lambda _x : -energy_fn(_x, nbr_lst, mod_nb_params)/kT


# In[81]:


naught_features = jnp.concatenate([jnp.zeros((1,4,1)), jnp.ones((num_particles-1, 4, 1))], axis=0)


# In[82]:


feature_dict = {0: naught_features, 1: None}


# In[244]:


import haiku as hk
module_kwargs = make_diff_fn_inits(feature_dict, 
                                   num_layers = 1,
                                   SinusoidalBasis_kwargs = {'r_switch': 1., 'r_cut': 8., 'basis_init' : hk.initializers.Constant(constant=jnp.linspace(1., 8., 8))},
                                   time_convolution_mlp_kwargs = {'output_sizes': [8,8], 'activation': jax.nn.swish},
                                   conv_mlp_kwargs = {'output_sizes': [8,8], 'activation': jax.nn.swish},
                                   tf_mlp_kwargs = {'output_sizes': [8,8], 'nonlinearity': jax.nn.swish})
module_kwargs['num_iters'] = 4


# In[245]:


from jax.example_libraries.optimizers import adam


# In[246]:


step_size_fn = lambda _step : 1e-3# * jnp.exp(-_step/1000.)
factory = SequentialFlussFactory(support_dimension=(num_particles, 3),
                       logp_prior=logp_prior,
                       logp_posterior=logp_posterior,
                       prior_train_data=prior_positions[:250],
                       posterior_train_data=posterior_positions[:250],
                       module=SequentialVectorModule,
                       module_kwargs=module_kwargs,
                       batch_size=32,
                       optimizer=adam,
                       optimizer_kwargs={'step_size': step_size_fn})


# In[247]:


util_dict = factory.get_utils(jax.random.PRNGKey(236), 0.)


# In[248]:


init_y = factory._prior_train_sampler(jax.random.PRNGKey(326))


# In[249]:


pretrain_fn = util_dict['pretrain_fn']
out_params, train_vals = pretrain_fn(util_dict['init_parameters'], key=jax.random.PRNGKey(346), num_iters=2000, clip_gradient_max_norm=1.)


# In[250]:


from matplotlib import pyplot as plt


# In[251]:


plt.plot(np.array(train_vals))
plt.yscale('log')


# can we perform the map and see how much we deviate from the starting position.

# In[252]:


canonical_fn = util_dict['canonical_differential_fn']


# In[253]:


pres_y = canonical_fn(out_params[0], init_y, None, hutch_key = jax.random.PRNGKey(3451))


# In[254]:


pres_y[0] - init_y


# In[255]:


train_fn = util_dict['train_fn']


# In[256]:


opt_state, train_vals, validate_vals, train_timings = train_fn(out_params, jax.random.PRNGKey(346), 15000, clip_gradient_max_norm=1.)


# In[257]:


plt.plot(np.array(train_vals))
plt.yscale('log')


# In[258]:


plt.plot(np.array(train_vals))
plt.yscale('log')


# In[259]:


plt.plot(np.array(train_vals)[1500:])
plt.yscale('log')


# In[233]:


plt.plot(np.array(train_vals)[1500:])
plt.yscale('log')


# In[234]:


plt.plot(np.array(train_vals)[1500:])
plt.yscale('log')


# In[260]:


plt.plot(np.array(train_vals))
#plt.yscale('log')


# In[261]:


fwd_loss_fn = util_dict['forward_loss_fn']
bkwd_loss_fn = util_dict['backward_loss_fn']
opt_params = util_dict['opt_get_params_fn'](opt_state)


# In[262]:


test_fwd_ys = prior_positions[300:]
test_bkwd_ys = posterior_positions[300:]


# In[263]:


fwd_vals = jax.vmap(fwd_loss_fn, in_axes=(None, 0, 0, None))(opt_params, test_fwd_ys, jax.random.split(jax.random.PRNGKey(1000789), num=200), 0.)
bkwd_vals = jax.vmap(bkwd_loss_fn, in_axes=(None, 0, 0, None))(opt_params, test_bkwd_ys, jax.random.split(jax.random.PRNGKey(200789), num=200), 0.)


# In[264]:


plt.hist(np.array(fwd_vals), alpha=0.5)
plt.hist(-np.array(bkwd_vals), alpha=0.5)
#plt.xlim(-50, 50)


# In[265]:


from pymbar import BAR, EXP


# In[266]:


EXP(fwd_vals)


# In[267]:


EXP(bkwd_vals)


# In[268]:


BAR(fwd_vals, bkwd_vals)


# need to put this onto gpu and run the same experiments

# In[ ]:




