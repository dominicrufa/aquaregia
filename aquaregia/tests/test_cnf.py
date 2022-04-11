"""
run tests on the cnf.py
"""
import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.example_libraries.optimizers import adam
import haiku as hk
from aquaregia.utils import Array, ArrayTree
import diffrax
from aquaregia.cnf import CNFFactory, TimeMLP, KinematicCNFFactory, KinematicTimeMLP, build_velocity_augmented_sampler
from pymbar import BAR

dim=2
batch_size=16
prior_mean, prior_cov=jnp.zeros(dim), jnp.eye(dim)
posterior_mean, posterior_cov=5.*jnp.ones(dim), jnp.eye(dim)*0.1
logp_prior = lambda _x : -0.5 * jnp.dot(_x - prior_mean, jnp.linalg.inv(prior_cov) @ (_x - prior_mean))
logp_posterior = lambda _x : -0.5 * jnp.dot(_x - posterior_mean, jnp.linalg.inv(posterior_cov) @ (_x - posterior_mean))

prior_sampler = partial(jax.random.multivariate_normal, mean=prior_mean, cov=prior_cov) # takes a key
posterior_sampler=partial(jax.random.multivariate_normal, mean=posterior_mean, cov=posterior_cov) # takes a key

prior_train_data = jax.vmap(prior_sampler)(jax.random.split(jax.random.PRNGKey(145), num=batch_size*1000))
posterior_train_data = jax.vmap(posterior_sampler)(jax.random.split(jax.random.PRNGKey(1215), num=batch_size*1000))

prior_validate_data = jax.vmap(prior_sampler)(jax.random.split(jax.random.PRNGKey(1445), num=100))
posterior_validate_data = jax.vmap(posterior_sampler)(jax.random.split(jax.random.PRNGKey(12145), num=100))
mlp_hidden_dim = 8
init_stepsize=1e-3
init_key=jax.random.PRNGKey(13)
time_horizon=1.
stepsize_controller=diffrax.PIDController
stepsize_controller_module_kwarg_dict = {'rtol':1e-3, 'atol':1e-6}
logp_velocity = logp_prior
velocity_sampler = prior_sampler

in_logp_prior = lambda _x : logp_prior(_x[0]) + logp_velocity(_x[1])
in_logp_posterior = lambda _x : logp_posterior(_x[0]) + logp_velocity(_x[1])

def test_CNFFactory():
    # get the exact free energy
    exact_free_energy = jnp.log(jnp.sqrt(jnp.linalg.det(prior_cov))) - jnp.log(jnp.sqrt(jnp.linalg.det(posterior_cov)))

    cnf_instance=CNFFactory(support_dimension = (dim,),
                 logp_prior = logp_prior,
                 logp_posterior = logp_posterior,
                 prior_train_data = prior_train_data,
                 posterior_train_data = posterior_train_data, # posterior_train_data,
                 diff_func_module = TimeMLP,
                 diff_func_builder_kwargs = {'output_sizes' : [mlp_hidden_dim, mlp_hidden_dim, dim]},
                 batch_size = batch_size,
                 optimizer = adam,
                 optimizer_kwargs = {'step_size': 1e-3},
                 init_stepsize = init_stepsize,
                 trace_estimator_method = 'exact',
                 init_key = init_key,
                 prior_validate_data = prior_validate_data,
                 posterior_validate_data = posterior_validate_data,
                 time_horizon = time_horizon,
                 stepsize_controller_module = diffrax.PIDController,
                 stepsize_controller_module_kwarg_dict=stepsize_controller_module_kwarg_dict
                )
    util_fns = cnf_instance.get_utils(jax.random.PRNGKey(253), regulariser_lambda=0.)
    post_sampler, prior_sampler = cnf_instance._posterior_train_sampler, cnf_instance._prior_train_sampler
    bkwd_loss_fn, fwd_loss_fn = util_fns['backward_loss_fn'], util_fns['forward_loss_fn']
    init_params = util_fns['init_parameters']
    train_fn = util_fns['train_fn']
    trained_opt_state, train_vals, validate_vals = train_fn(init_params, seed=jax.random.PRNGKey(457), num_iters=500, validate_frequency=100, clip_gradient_max_norm=1e1)
    final_params = util_fns['opt_get_params_fn'](trained_opt_state)
    prior_test_data = jax.vmap(prior_sampler)(jax.random.split(jax.random.PRNGKey(15), num=100))
    posterior_test_data = jax.vmap(posterior_sampler)(jax.random.split(jax.random.PRNGKey(215), num=100))

    forward_works = jax.vmap(fwd_loss_fn, in_axes=(None, 0,0, None))(final_params, prior_test_data, jax.random.split(jax.random.PRNGKey(451), num=100), 0.)
    backward_works = jax.vmap(bkwd_loss_fn, in_axes=(None, 0,0, None))(final_params, posterior_test_data, jax.random.split(jax.random.PRNGKey(45), num=100), 0.)
    df, ddf = BAR(forward_works, backward_works)
    assert (df + 3.*ddf > exact_free_energy) and (df-3.*ddf < exact_free_energy), f"calc df is {df}, but exact df is {exact_free_energy}"

def test_KinematicCNFFactory():
    exact_free_energy = jnp.log(jnp.sqrt(jnp.linalg.det(prior_cov))) - jnp.log(jnp.sqrt(jnp.linalg.det(posterior_cov)))
    kinetic_cnf_instance = KinematicCNFFactory(support_dimension = (dim,),
                 logp_prior = in_logp_prior,
                 logp_posterior = in_logp_posterior,
                 prior_train_data = prior_train_data[1000:],
                 velocity_sampler = velocity_sampler,
                 posterior_train_data = posterior_train_data[1000:], # posterior_train_data,
                 diff_func_module = KinematicTimeMLP,
                 diff_func_builder_kwargs = {'output_sizes' : [mlp_hidden_dim, mlp_hidden_dim, dim*2]},
                 batch_size = batch_size,
                 optimizer = adam,
                 optimizer_kwargs = {'step_size' : 1e-2},
                 init_stepsize = init_stepsize,
                 trace_estimator_method = 'exact',
                 init_key = jax.random.PRNGKey(4256),
                 prior_validate_data = prior_validate_data,
                 posterior_validate_data = posterior_validate_data,
                 time_horizon = time_horizon,
                 stepsize_controller_module = diffrax.PIDController,
                 stepsize_controller_module_kwarg_dict=stepsize_controller_module_kwarg_dict
                 )

    util_fns = kinetic_cnf_instance.get_utils(jax.random.PRNGKey(253), regulariser_lambda=0.)
    bkwd_loss_fn, fwd_loss_fn = util_fns['backward_loss_fn'], util_fns['forward_loss_fn']
    init_params = util_fns['init_parameters']
    train_fn = util_fns['train_fn']
    # train
    trained_opt_state, train_vals, validate_vals = train_fn(init_params, seed=jax.random.PRNGKey(457), num_iters=2000, validate_frequency=500)
    final_params = util_fns['opt_get_params_fn'](trained_opt_state)
    prior_test_data = prior_train_data[:1000]
    posterior_test_data = posterior_train_data[:1000]

    prior_test_sampler = build_velocity_augmented_sampler(prior_test_data, velocity_sampler)
    posterior_test_sampler = build_velocity_augmented_sampler(posterior_test_data, velocity_sampler)
    prior_data = jax.vmap(prior_test_sampler)(jax.random.split(jax.random.PRNGKey(3218), num=1000))
    posterior_data = jax.vmap(posterior_test_sampler)(jax.random.split(jax.random.PRNGKey(180), num=1000))

    forward_works = jax.vmap(fwd_loss_fn, in_axes=(None, 0,0,None))(final_params, prior_data, jax.random.split(jax.random.PRNGKey(456), num=1000), 0.)
    backward_works = jax.vmap(bkwd_loss_fn, in_axes=(None, 0,0,None))(final_params, posterior_data, jax.random.split(jax.random.PRNGKey(4547), num=1000), 0.)

    df, ddf = BAR(forward_works, backward_works)
    assert (df + 3.*ddf > exact_free_energy) and (df-3.*ddf < exact_free_energy), f"calc df is {df}, but exact df is {exact_free_energy}"
