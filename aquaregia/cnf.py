"""
cnf for gen modeling and fecs
"""
from jax.config import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import random
from jax.example_libraries.optimizers import adam
import numpy as np
from functools import partial
import haiku as hk
from aquaregia.utils import Array, ArrayTree, polynomial_switching_fn
from aquaregia import tfn
import diffrax
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController
from typing import Callable, Union, Sequence, Tuple, Optional, Iterable
KEY = Array
NONE = type(None)
OPTIMIZER = type(adam)
MODULE = type(hk.Module)
EQ_MODULE = type(diffrax.PIDController)
ALLOWED_TRACE_METHODS = ['exact', 'hutch']

"""
simple toy mlps first for non-canonical diff_f fns
"""
class TimeMLP(hk.Module):
    """
    generic mlp for diff_f_fn
    """
    def __init__(self, output_sizes, name=None):
        super().__init__(name=name)
        self._mlp = hk.nets.MLP(output_sizes)
    def __call__(self, t, y):
        return self._mlp(jnp.concatenate([y, jnp.array([t])]))

class KinematicTimeMLP(hk.Module):
    """
    special mlp for kinematic-like diff_fn;
    Notably, the `y` input is an Array of shape (2,N,dim)
    """
    def __init__(self, output_sizes, name=None):
        super().__init__(name=name)
        self._mlp = hk.nets.MLP(output_sizes)
    def __call__(self, t, y):
        xs, vs = y[0], y[1]
        ins = jnp.concatenate([xs, jnp.array([t])])
        outs = self._mlp(jnp.concatenate([xs, jnp.array([t])]))
        mod_shape = int(outs.shape[0] / 2)
        scales, translations = outs[: mod_shape], outs[mod_shape:]
        return jnp.array([vs, scales * vs + translations]), scales, translations

"""
augmenting diff_f fns
"""
def exact_aug_diff_f(t, y, args_tuple):
    """
    create an augmented canonical differential equation function that will integrate the
    equation of motions as well as the divergence. in an exact manner
    """
    _y = y[0] # the first entry is the value of y at time t
    _params, _key, diff_f = args_tuple
    aug_diff_fn = lambda __y : diff_f(t, __y, (_params,))
    _f, _vjp_fn = jax.vjp(aug_diff_fn, _y)
    (size,) = _y.shape
    eye = jnp.eye(size)
    (dfdy,) = jax.vmap(_vjp_fn)(eye)
    trace = jnp.trace(dfdy)
    return _f, trace, jnp.sum(_f**2)

def exact_kinematic_aug_diff_f(t, y, args_tuple):
    """
    """
    _y, _, _ = y
    _params, _key, diff_f = args_tuple
    aug_diff_fn = lambda __y : diff_f(t, __y, (_params,))
    _f, scales, translations = aug_diff_fn(_y)
    trace = jnp.sum(scales)
    return _f, trace, jnp.sum(_f**2)

"""
ode solvers
"""
def ode_solver(_y0, _params, _hutch_key, diff_f, augmented_diff_f, solver, t0, t1, dt0, **kwargs):
    """
    wrapper for ode solver.
    TODO : allow for other args
    """
    term = ODETerm(augmented_diff_f)
    solver = solver()
    sol = diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=_y0, args=(_params, _hutch_key, diff_f), **kwargs)
    return sol.ys

from typing import Callable, Union, Sequence, Tuple, Optional
from aquaregia.utils import Array, ArrayTree
KEY = Array
NONE = type(None)
OPTIMIZER = type(adam)
MODULE = type(hk.Module)
EQ_MODULE = type(diffrax.PIDController)

ALLOWED_TRACE_METHODS = ['exact', 'hutch']

"""
sampler utilities
"""
def build_sampler(data):
    """
    simple utility function to construct a sampler
    """
    num_datapoints, data_dimension = data.shape[0], data.shape[1:]
    def sampler(key):
        index = jax.random.choice(key, jnp.arange(num_datapoints, dtype=jnp.int32))
        return data[index]
    return sampler

def build_velocity_augmented_sampler(data, velocity_sampler):
    num_datapoints, data_dimension = data.shape[0], data.shape[1:]
    def sampler(key):
        idx_key, v_key = jax.random.split(key)
        index = jax.random.choice(idx_key, jnp.arange(num_datapoints, dtype=jnp.int32))
        return jnp.concatenate([data[index][jnp.newaxis, ...], velocity_sampler(v_key)[jnp.newaxis, ...]], axis=0)
    return sampler

"""
cnf factories
"""
class CNFFactory(object):
    """
    create necessary functionality for training, executing a continuous normalising flow
    """
    _allowed_states = ['prior', 'posterior']
    def __init__(self,
                 support_dimension : tuple,
                 logp_prior : Callable[Array, float],
                 logp_posterior : Callable[Array, float],
                 prior_train_data : Array,
                 posterior_train_data : Array,
                 diff_func_module : MODULE,
                 diff_func_builder_kwargs : dict,
                 batch_size : int,
                 optimizer : OPTIMIZER,
                 optimizer_kwargs : dict,
                 init_stepsize : Optional[float] = 1e-3,
                 trace_estimator_method : Optional[str] = 'exact',
                 init_key : Optional[KEY] = jax.random.PRNGKey(4256),
                 prior_validate_data : Optional[Array] = None,
                 posterior_validate_data : Optional[Array] = None,
                 time_horizon : Optional[float] = 1.,
                 stepsize_controller_module : Optional[EQ_MODULE] = None,
                 stepsize_controller_module_kwarg_dict : Optional[dict] = {},
                 **kwargs # for show
                ):

        # set the dimension of the support
        self._support_dimension = support_dimension

        # samplers and logp calculators; take array of support dimension and return a float
        self._logp_prior = logp_prior
        self._logp_posterior = logp_posterior

        # create the attributes for the differential function module
        self._diff_func_module = diff_func_module
        self._diff_func_builder_kwargs = diff_func_builder_kwargs
        self._trace_estimator = trace_estimator_method
        self._stepsize_controller_module = stepsize_controller_module
        self._stepsize_controller_module_kwarg_dict = stepsize_controller_module_kwarg_dict


        self._batch_size = batch_size
        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs

        # build the train samplers
        self._make_train_samplers(prior_train_data, 'prior')
        self._make_train_samplers(posterior_train_data, 'posterior')
        if not (self._forward_train_bool or self._backward_train_bool):
            raise ValueError(f"user must provide at least a prior or posterior train data. Neither was registered")

        # build the validate samplers
        self._make_validate(prior_validate_data, 'prior')
        self._make_validate(posterior_validate_data, 'posterior')

        #hardcode some vals
        self._t0 = 0.
        self._t1 = time_horizon
        self._dt0 = init_stepsize

    def _validate_data_dimensionality(self, dataset):
        num_samples, data_dimension = dataset.shape[0], dataset.shape[1:]
        if data_dimension != self._support_dimension:
            raise ValueError(f"dataset dimension ({data_dimension}) is not equal to the support dimension ({self._support_dimension})")

    def _validate_state(self, state):
        if state not in self._allowed_states:
            raise ValueError(f"state `{state}` is not in allowed states of {self._allowed_states}")

    def _make_train_samplers(self, data, state):
        """
        process the `train` data if it exists; if it is the incorrect shape, set directional train bool to `False` as to avoid the calculation
        """
        self._validate_state(state)
        direct = 'forward' if state=='prior' else 'backward'
        try:
            self._validate_data_dimensionality(data)
            setattr(self, f"_{state}_train_sampler", build_sampler(data))
            setattr(self, f"_{direct}_train_bool", True)
        except Exception as e:
            print(f"setting train sampler for state {state} raised Exception: {e}. omitting {direct} training method")
            setattr(self, f"_{state}_train_sampler", None)
            setattr(self, f"_{direct}_train_bool", False)


    def _make_validate(self, validate_data, state):
        """
        process the "validation" data if it exists; if it is the incorrect shape, set validate bool to `False` to as to avoid validation calculations
        """
        self._validate_state(state)
        direct = 'forward' if state=='prior' else 'backward'
        try:
            self._validate_data_dimensionality(validate_data)
            setattr(self, f"_{state}_validate_data", validate_data)
            setattr(self, f"_{direct}_validate_bool", True)
            def batch_validate_sampler(key): return getattr(self, f"_{state}_validate_data")
            setattr(self, f"_batch_{state}_validate_sampler", batch_validate_sampler)
        except Exception as e:
            print(f"setting validation data for state {state} raised Exception: {e}")
            setattr(self, f"_{state}_validate_data", None)
            setattr(self, f"_{direct}_validate_bool", False)
            def batch_validate_sampler(key): raise NotImplementedError()
            setattr(self, f"_batch_{state}_validate_sampler", batch_validate_sampler)


    def get_canonical_diff_fn_and_parameters(self, init_key):
        """
        to nest a haiku module, we need to create a portable function that will initialize the
        """
        # create a transformable function for the diff_f_module
        def _diff_f_wrapper(t, y):
            diff_f = self._diff_func_module(**self._diff_func_builder_kwargs)
            return diff_f(t, y)

        # transform the differential
        diff_f_init, diff_f_apply = hk.without_apply_rng(hk.transform(_diff_f_wrapper))

        #initialize the differential function
        ys_key, init_key = jax.random.split(init_key)
        if self._forward_train_bool:
            init_y = self._prior_train_sampler(ys_key)
        else:
            init_y = self._posterior_train_sampler(ys_key)
        init_params = diff_f_init(init_key, self._t0, init_y)

        # canonicalize the function
        canonicalized_diff_f_fn = lambda _t, _y, _args_tuple : diff_f_apply(_args_tuple[0], _t, _y)
        return init_params, canonicalized_diff_f_fn

    def get_augmented_diff_func(self):
        """
        augment the canonicalized diff_f to return the divergence calculator, either approximate, exact, etc
        """
        if self._trace_estimator == 'exact':
            aug_diff_f = exact_aug_diff_f
        else:
            raise NotImplementedError(f"{self._trace_estimator} is not implemented")
        return aug_diff_f

    def get_ode_solver(self, init_key):
        """
        wrap the canonicalized differential function into the ode solver
        """
        init_parameters, canonical_diff_fn = self.get_canonical_diff_fn_and_parameters(init_key)
        augmented_diff_func = self.get_augmented_diff_func()

        # stepsize_controller
        if self._stepsize_controller_module is not None:
            stepsize_mod = self._stepsize_controller_module
            stepsize_kwarg_dict = self._stepsize_controller_module_kwarg_dict
        else:
            stepsize_mod = diffrax.ConstantStepSize
            stepsize_kwarg_dict = {'compile_steps' : False}


        def ode_solver(parameters, init_y, hutch_key, start_time, end_time):
            dt = jax.lax.cond(start_time < end_time, lambda _: self._dt0, lambda _: -self._dt0, None) # determine if we are traveling forward/backward in time
            term = ODETerm(augmented_diff_func) # transform the augmented canonicalized differential function
            stepsize_controller = stepsize_mod(**stepsize_kwarg_dict)
            solver = Dopri5() # define a solver; TODO : allow user to change this in future if we need higher precision
            aug_init_y = (init_y, 0., 0.) # the second argument is the trace counter
            sol = diffeqsolve(term, solver, t0=start_time, t1=end_time, dt0=dt, y0=aug_init_y, stepsize_controller = stepsize_controller, args=(parameters, hutch_key, canonical_diff_fn))
            return sol
        return init_parameters, canonical_diff_fn, ode_solver

    def get_loss_functions(self, init_key):
        """
        a simple wrapper to build a vmappable (or pmappable) loss function;
        TODO : place this into a test and test each loss function in training separately because i am
               not totally convinced i have the sign right here.
        """
        init_parameters, canonical_diff_fn, ode_solver = self.get_ode_solver(init_key)

        def backward_loss_fn(parameters, init_y, hutch_key, regulariser_lambda):
            """
            sampling from the posterior, run integrator backward in time to prior and compute loss
            """
            u_start = -self._logp_posterior(init_y)
            sol = ode_solver(parameters=parameters, init_y=init_y, hutch_key=hutch_key, start_time=self._t1, end_time=self._t0) # propagate backward in time
            out_y, div, aug_loss = sol.ys[0][0], sol.ys[1][0], sol.ys[2][0] # for some reason, the outputs are wrapped in a newaxis at index 0
            u_out = -self._logp_prior(out_y)
            return u_out - u_start - div + regulariser_lambda*aug_loss # check to make sure it is (minus div)

        def forward_loss_fn(parameters, init_y, hutch_key, regulariser_lambda):
            """
            sampling from the prior, run integration forward in time to posterior and compute loss
            """
            u_start = -self._logp_prior(init_y)
            sol = ode_solver(parameters=parameters, init_y=init_y, hutch_key=hutch_key, start_time=self._t0, end_time=self._t1) # propagate forward in time
            out_y, div, aug_loss = sol.ys[0][0], sol.ys[1][0], sol.ys[2][0]
            u_out = -self._logp_posterior(out_y)
            return u_out - u_start - div + regulariser_lambda*aug_loss # check to make sure it is (plus div, since there is a discrepancy between running aug ode in fwd/bkwd directions)

        return init_parameters, canonical_diff_fn, ode_solver, backward_loss_fn, forward_loss_fn

    def _handle_directionality(self, forward_fn, backward_fn):
        """simple utility function to query whether we process a singular-directional function and which direction"""
        single_directional_bool = True if forward_fn is None or backward_fn is None else False
        if single_directional_bool:
            direction = 'backward' if forward_fn is None else 'forward'
        else:
            direction = 'both'
        return single_directional_bool, direction


    def get_incorporated_train_loss_fn(self,
                                 forward_singular_loss_fn,
                                 backward_singular_loss_fn):
        """
        take a singular forward and/or backward loss function from `get_loss_fn` method and wrap them into a loss function that
        is directly portable into `step` function for training purposes; it is intrinsically vmapped and batch draws from the
        prior/posterior inside the function.

        Namely, if both forward/backward losses are provided, the loss is given by the sum of the forward/backward works;
        otherwise, the loss is given by the forward/backward work.
        """
        single_directional_bool, direction = self._handle_directionality(forward_singular_loss_fn, backward_singular_loss_fn)
        if single_directional_bool:
            # incorporate the singular forward xor backward sampler into the loss function
            direction = 'backward' if forward_singular_loss_fn is None else 'forward'
            singular_loss_fn = forward_singular_loss_fn if direction == 'forward' else backward_singular_loss_fn
            sampler_fn = self._prior_train_sampler if direction == 'forward' else self._posterior_train_sampler
            def incorporated_loss_fn(parameters, key):
                hutch_key, sampler_key = jax.random.split(key)
                init_y = sampler_fn(sampler_key)
                return singular_loss_fn(parameters = parameters, init_y = init_y, hutch_key = hutch_key)
        else: # bidirectional
            # incorporate both forward and backward sampler into singular loss functions
            prior_sampler, posterior_sampler = self._prior_train_sampler, self._posterior_train_sampler
            def incorporated_loss_fn(parameters, key):
                hutch_key_fwd, hutch_key_bkwd, fwd_sampler_key, bkwd_sampler_key = jax.random.split(key, num=4)
                fwd_init_y, bkwd_init_y = prior_sampler(fwd_sampler_key), posterior_sampler(bkwd_sampler_key)
                fwd_work = forward_singular_loss_fn(parameters=parameters, init_y = fwd_init_y, hutch_key=hutch_key_fwd)
                bkwd_work = backward_singular_loss_fn(parameters=parameters, init_y = bkwd_init_y, hutch_key=hutch_key_bkwd)
                return fwd_work + bkwd_work

        vmapped_fn = lambda parameters, key: jax.vmap(incorporated_loss_fn, in_axes=(None, 0))(parameters, jax.random.split(key, num=self._batch_size)).mean()
        return incorporated_loss_fn, vmapped_fn

    def get_incorporated_validate_loss_fn(self,
                                         forward_singular_loss_fn,
                                         backward_singular_loss_fn):
        """
        see `get_incorporated_train_loss_fn`, except this is exclusively for validation data. namely, there is no stochasticity in the prior/posterior sampler
        since we query the data directly.
        """
        single_directional_bool, direction = self._handle_directionality(forward_singular_loss_fn, backward_singular_loss_fn)
        if single_directional_bool:
            batch_sampler = self._batch_prior_validate_sampler if direction=='forward' else self._batch_posterior_validate_sampler
            singular_loss_fn = forward_singular_loss_fn if direction == 'forward' else backward_singular_loss_fn
            def batch_incorporated_loss_fn(parameters, key):
                hutch_key, sampler_key = jax.random.split(key)
                init_ys = sampler_key(sampler_key)
                losses = jax.vmap(singular_loss_fn, in_axes=(None, 0, 0))(parameters, init_ys, jax.random.split(hutch_key, num=init_ys.shape[0]))
                return losses.mean()
        else:
            def batch_incorporated_loss_fn(parameters, key):
                fwd_hutch_key, bkwd_hutch_key, fwd_sampler_key, bkwd_sampler_key = jax.random.split(key, num=4)
                fwd_init_ys = self._batch_prior_validate_sampler(fwd_sampler_key)
                bkwd_init_ys = self._batch_posterior_validate_sampler(bkwd_sampler_key)
                fwd_losses = jax.vmap(forward_singular_loss_fn, in_axes=(None,0,0))(parameters, fwd_init_ys, jax.random.split(fwd_hutch_key, num=fwd_init_ys.shape[0]))
                bkwd_losses = jax.vmap(backward_singular_loss_fn, in_axes=(None,0,0))(parameters, bkwd_init_ys, jax.random.split(bkwd_hutch_key, num=bkwd_init_ys.shape[0]))
                return fwd_losses.mean() + bkwd_losses.mean()
        return batch_incorporated_loss_fn


    def get_utils(self, init_key, regulariser_lambda):
        """
        internally wrap the train utils for ease of implementation;
        TODO : fix for difference in training vs validation loss functions. this is important.
        """
        init_fun, update_fun, get_params = self._optimizer(**self._optimizer_kwargs) # call the optimizer functions

        # create the parameters, canonical diff fn, ode solver, and backward/forward loss functions regardless of which mode is specified
        init_parameters, canonical_diff_fn, ode_solver, backward_loss_fn, forward_loss_fn = self.get_loss_functions(init_key) # create the un-vmapped loss functions

        # get train loss fn
        in_fwd_loss_fn = partial(forward_loss_fn, regulariser_lambda=regulariser_lambda) if self._forward_train_bool else None
        in_bkwd_loss_fn = partial(backward_loss_fn, regulariser_lambda=regulariser_lambda) if self._backward_train_bool else None
        incorporated_train_loss_fn, vmapped_incorporated_train_loss_fn = self.get_incorporated_train_loss_fn(forward_singular_loss_fn=in_fwd_loss_fn,
                                                                                                             backward_singular_loss_fn=in_bkwd_loss_fn)

        # get validate loss fn; regulariser_lambda is always zero
        in_fwd_loss_fn = partial(forward_loss_fn, regulariser_lambda=0.) if self._forward_validate_bool else None
        in_bkwd_loss_fn = partial(backward_loss_fn, regulariser_lambda=0.) if self._backward_validate_bool else None
        if in_fwd_loss_fn is None and in_bkwd_loss_fn is None:
            batch_validate_loss_fn = None
        else:
            batch_validate_loss_fn = self.get_incorporated_validate_loss_fn(forward_singular_loss_fn=in_fwd_loss_fn,
                                                                            backward_singular_loss_fn=in_bkwd_loss_fn)

        @jax.jit
        def step(key, _iter, opt_state, clip_gradient_max_norm):
            """jittable step function for `train` function's `for` loop"""
            in_params = get_params(opt_state) # extract the parameters from state
            mean_val, param_grads = jax.value_and_grad(vmapped_incorporated_train_loss_fn)(in_params, key) # compute value and grad
            new_opt_state = update_fun(_iter, jax.example_libraries.optimizers.clip_grads(param_grads, clip_gradient_max_norm), opt_state) # update the state with the vals and grads
            return mean_val, new_opt_state

        def train(init_params, seed, num_iters, validate_frequency=None, clip_gradient_max_norm=1e3):
            """
            non-jittable train function that records training/validation data throughout training.
            """
            import tqdm
            train_values, validate_values = [], []
            opt_state = init_fun(init_params) # initialize the opt state
            trange = tqdm.trange(num_iters, desc=f"Bar desc", leave=True)
            for i in trange: # begin for loop
                run_seed, seed = random.split(seed)
                train_value, opt_state = step(key = run_seed, _iter=i, opt_state = opt_state, clip_gradient_max_norm=clip_gradient_max_norm)
                train_values.append(train_value)
                if validate_frequency is not None and i % validate_frequency == 0 and batch_validate_loss_fn is not None:
                    validate_seed, seed = jax.random.split(seed)
                    validate_value = jax.jit(batch_validate_loss_fn)(get_params(opt_state), validate_seed)
                    validate_values.append(validate_value)
                trange.set_description(f"test loss: {train_value}")
                trange.refresh() # halt this?
            return opt_state, Array(train_values), Array(validate_values)

        return_dict = {'init_parameters' : init_parameters,
                       'step_fn' : step,
                       'train_fn' : train,
                       'canonical_differential_fn' : canonical_diff_fn,
                       'ode_solver' : ode_solver,
                       'backward_loss_fn' : backward_loss_fn,
                       'forward_loss_fn' : forward_loss_fn,
                       'opt_init_fn' : init_fun,
                       'opt_update_fn' : update_fun,
                       'opt_get_params_fn' : get_params}

        return return_dict

class KinematicCNFFactory(CNFFactory):
    """
    create necessary functionality for training, executing a continuous normalising flow in the kinematics regime.
    """
    _allowed_states = ['prior', 'posterior']
    def __init__(self,
                 support_dimension : tuple, # this is doubled to account for velocity
                 logp_prior : Callable[Array, float],
                 logp_posterior : Callable[Array, float],
                 prior_train_data : Array,
                 velocity_sampler : Callable[KEY, Array],
                 posterior_train_data : Array,
                 diff_func_module : MODULE, # this module needs to return a
                 diff_func_builder_kwargs : dict,
                 batch_size : int,
                 optimizer : OPTIMIZER,
                 optimizer_kwargs : dict,
                 init_stepsize : Optional[float] = 1e-3,
                 trace_estimator_method : Optional[str] = 'exact',
                 init_key : Optional[KEY] = jax.random.PRNGKey(4256),
                 prior_validate_data : Optional[Array] = None,
                 posterior_validate_data : Optional[Array] = None,
                 time_horizon : Optional[float] = 1.,
                 stepsize_controller_module : Optional[EQ_MODULE] = None,
                 stepsize_controller_module_kwarg_dict : Optional[dict] = {},
                 **kwargs
                ):
        self._velocity_sampler = velocity_sampler
        # holy kwargs, batman!
        super().__init__(support_dimension=support_dimension,
                         logp_prior=logp_prior,
                         logp_posterior=logp_posterior,
                         prior_train_data=prior_train_data,
                         posterior_train_data=posterior_train_data,
                         diff_func_module=diff_func_module,
                         diff_func_builder_kwargs=diff_func_builder_kwargs,
                         batch_size=batch_size,
                         optimizer=optimizer,
                         optimizer_kwargs=optimizer_kwargs,
                         init_stepsize=init_stepsize,
                         trace_estimator_method=trace_estimator_method,
                         init_key=init_key,
                         prior_validate_data=prior_validate_data,
                         posterior_validate_data=posterior_validate_data,
                         time_horizon=time_horizon,
                         stepsize_controller_module=stepsize_controller_module,
                         stepsize_controller_module_kwarg_dict=stepsize_controller_module_kwarg_dict,
                         **kwargs)

        self._validate_velocity_dimensionality()

    def get_augmented_diff_func(self):
        """
        augment the canonicalized diff_f to return the divergence calculator, either approximate, exact, etc
        """
        if self._trace_estimator == 'exact':
            aug_diff_f = exact_kinematic_aug_diff_f
        else:
            raise NotImplementedError(f"{self._trace_estimator} is not implemented")
        return aug_diff_f

    def _make_validate(self, validate_data, state):
        """
        process the "validation" data if it exists; if it is the incorrect shape, set validate bool to `False` to as to avoid validation calculations
        """
        self._validate_state(state)
        direct = 'forward' if state=='prior' else 'backward'
        try:
            self._validate_data_dimensionality(validate_data)
            num_datapoints = validate_data.shape[0]
            setattr(self, f"_{state}_validate_data", validate_data)
            setattr(self, f"_{direct}_validate_bool", True)
            def batch_validate_sampler(key):
                split_keys = jax.random.split(key, num=num_datapoints)
                velocities = jax.vmap(self._velocity_sampler)(split_keys)
                out = jnp.hstack([jnp.expand_dims(getattr(self, f"_{state}_validate_data"), 1), jnp.expand_dims(velocities, 1)])
                return out
            setattr(self, f"_batch_{state}_validate_sampler", batch_validate_sampler)
        except Exception as e:
            print(f"setting validation data for state {state} raised Exception: {e}")
            setattr(self, f"_{state}_validate_data", None)
            setattr(self, f"_{direct}_validate_bool", False)
            def batch_validate_sampler(key): raise NotImplementedError()
            setattr(self, f"_batch_{state}_validate_sampler", batch_validate_sampler)

    def _validate_velocity_dimensionality(self):
        trial_velocity = self._velocity_sampler(jax.random.PRNGKey(13))
        if not trial_velocity.shape == self._support_dimension:
            raise ValueError(f"the velocity sampler returns samples with dimension {trial_velocity.shape}, which is not the same as the support dimension {self._support_dimension}")

    def _make_train_samplers(self, data, state):
        """
        process the `train` data if it exists; if it is the incorrect shape, set directional train bool to `False` as to avoid the calculation
        """
        self._validate_state(state)
        direct = 'forward' if state=='prior' else 'backward'
        try:
            self._validate_data_dimensionality(data)
            setattr(self, f"_{state}_train_sampler", build_velocity_augmented_sampler(data, self._velocity_sampler))
            setattr(self, f"_{direct}_train_bool", True)
        except Exception as e:
            print(f"setting train sampler for state {state} raised Exception: {e}. omitting {direct} training method")
            setattr(self, f"_{state}_train_sampler", None)
            setattr(self, f"_{direct}_train_bool", False)

    def get_pretrain_functions(self, canonical_diff_fn):
        init_fun, update_fun, get_params = self._optimizer(**self._optimizer_kwargs) # call the optimizer functions

        def pull_train_sample(key):
            forward_key, backward_key, bool_key = jax.random.split(key, num=3)
            forward_data = jax.lax.cond(self._forward_train_bool,
                                        lambda _key: self._prior_train_sampler(_key),
                                        lambda _key: self._posterior_train_sampler(_key), forward_key)
            backward_data = jax.lax.cond(self._backward_train_bool,
                                         lambda _key: self._posterior_train_sampler(_key),
                                         lambda _key : self._prior_train_sampler(_key), backward_key)
            return jax.random.choice(bool_key, jnp.vstack([forward_data, backward_data]))


        def loss_fn(parameters, key, scale_target, translation_target):
            time_key, run_key = jax.random.split(key)
            t = jax.random.uniform(time_key, minval=self._t0, maxval=self._t1)
            data = pull_train_sample(run_key)
            _, scale, translation = canonical_diff_fn(t, data, (parameters,))
            return jnp.sum((scale-scale_target)**2) + jnp.sum((translation-translation_target)**2)

        def batch_loss_fn(parameters, key, scale_target, translation_target):
            keys = jax.random.split(key, num=self._batch_size)
            return jnp.sum(jax.vmap(loss_fn, in_axes=(None, 0, None, None))(parameters, keys, scale_target, translation_target))


        @jax.jit
        def step(key, _iter, opt_state, clip_gradient_max_norm, scale_target, translation_target):
            in_params = get_params(opt_state) # extract the parameters from state
            val, param_grads = jax.value_and_grad(batch_loss_fn)(in_params, key, scale_target, translation_target) # compute value and grad
            new_opt_state = update_fun(_iter, jax.example_libraries.optimizers.clip_grads(param_grads, clip_gradient_max_norm), opt_state) # update the state with the vals and grads
            return mean_val, new_opt_state

        def train(init_params, key, num_iters, clip_gradient_max_norm=1e1, scale_target=-1e2, translation_target=0.):
            import tqdm
            train_values = []
            opt_state = init_fun(init_params)
            trange = tqdm.trange(num_iters, desc="Bar desc", leave=True)
            for i in trange:
                run_seed, seed = jax.random.split(key)
                val, opt_state = step(run_seed, _iter=i, opt_state=opt_state, clip_gradient_max_norm=clip_gradient_max_norm, scale_target=scale_target, translation_target=translation_target)
                train_values.append(val)
                trange.set_description(f"test loss: {val}")
                trange.refresh()
            return opt_state, Array(train_values)
        return train


    def get_utils(self, init_key, regulariser_lambda):
        """
        just like the super init, except we also equip the dict with a pre-training function
        """
        out_dict = super().get_utils(init_key, regulariser_lambda) # super it!
        canonical_diff_fn = out_dict['canonical_differential_fn']
        pretrain_fn = self.get_pretrain_functions(canonical_diff_fn)
        out_dict['pretrain_fn'] = pretrain_fn
        return out_dict

# """
# vector module utils for CNF
# """
# class VectorModule(hk.Module):
#     """
#     a vector module for SE(3) equivariance
#     """
#     def __init__(self,
#                  num_particles : int,
#                  input_L0_channel_dimension : tuple,
#                  conv_dict_list : Iterable,
#                  mlp_dict_list : Iterable,
#                  SinusoidalBasis_kwargs : dict,
#                  time_convolution_kwargs : dict,
#                  name : Optional[str] = f"vector"):
#         super().__init__(name=name)
#         self.SinusoidalBasis_module = tfn.SinusoidalBasis(**SinusoidalBasis_kwargs)
#         self._num_particles = num_particles
#         self._input_L0_channel_dimension = input_L0_channel_dimension
#
#         # time convolution
#         self.time_convolution_module = hk.nets.MLP(**time_convolution_kwargs)
#
#         # setup the for loop
#         layers = {}
#         num_layers = len(conv_dict_list)
#         for layer in range(num_layers):
#             conv_module = tfn.Convolution(**conv_dict_list[layer])
#             tfn_module_dict = {_L: tfn.TensorFieldMLP(**mlp_dict_list[layer][_L]) for _L in mlp_dict_list[layer].keys()}
#             layers[layer] = {'conv': conv_module, 'mlp': tfn_module_dict}
#         self.layers = layers
#
#         # make vmap and fori functions
#         def fori_body_fn(layer_idx, val_tuple):
#             layer_dict = self.layers[layer_idx]
#             in_tensor_dict, rbf_inputs, unit_r_ij, norms, epsilon = val_tuple
#             conv_dict = layer_dict['conv'](in_tensor_dict=in_tensor_dict, rbf_inputs=rbf_inputs, unit_vectors=unit_r_ij, r_ij = norms, epsilon = tfn.DEFAULT_EPSILON)
#             reference_dict = {_L : _L for _L in conv_dict.keys()}
#             tree_map_fn = lambda _L : layer_dict['mlp'][_L](inputs=conv_dict[_L], epsilon)
#             return jax.tree_util.tree_map(tree_map_fn, reference_dict), rbf_inputs, unit_r_ij, norms, epsilon
#         self._fori_body_fn = fori_body_fn
#
#
#     def __call__(self,
#                  t : float,
#                  xs_and_vs : Array,
#                  feature_dictionary : ArrayTree,
#                  epsilon : Optional[float] = tfn.DEFAULT_EPSILON,
#                  mask_value : Optional[float] = 0.,
#                  ):
#         positions, velocities = xs_and_vs[0], xs_and_vs[1]
#
#         #simple check on feature dictionary
#         if feature_dictionary[0].shape != (self._num_particles, self._input_L0_channel_dimension, 1):
#             raise ValueError(f"""the feature dictionary's L=0 input ({feature_dictionary[0].shape})
#                                  is not equal to the specified channel dimension
#                                  ({(self._num_particles, self._input_L0_channel_dimension, 1)})""")
#         elif feature_dictionary[1] is not None:
#             raise ValueError(f"the input feature dictionary should have no annotations on the L=1 input")
#
#         r_ij = tfn.DEFAULT_VDISPLACEMENT_FN(positions, positions)
#         unit_r_ij, norms = tfn.unit_vectors_and_norms(r_ij) # compute unit vectors and norms
#         norms = tfn.mask_tensor(norms, mask_val = mask_value)
#         norms = jnp.squeeze(norms)
#
#         rbf_inputs = self.SinusoidalBasis_module(r_ij = norms, epsilon=epsilon, mask_val=mask_value)
#
#         # concat feature dictionary with time convolutions.
#         time_convolution = self.time_convolution_module(Array([t]))
#         repeated_time_convolution = jnp.repeat(time_convolution[jnp.newaxis, ..., jnp.newaxis], repeats=self._num_particles, axis=0)
#         aug_L0 = jnp.hstack([feature_dict[0], repeated_time_convolution])
#         in_tensor_dict = {L : _tensor for L, _tensor in feature_dictionary.items()}
#         in_tensor_dict[0] = aug_L0
#
#         final_out = hk.fori_loop(0, len(self.layers), self._fori_body_fn, (in_tensor_dict, ))
#
#         for layer_idx in range(len(self.layers)):
#             out_tensor_dict = {}
#             layer_dict = self.layers[layer_idx]
#             conv_dict = layer_dict['conv'](in_tensor_dict=in_tensor_dict, rbf_inputs=rbf_inputs, unit_vectors=unit_r_ij, r_ij = norms, epsilon = tfn.DEFAULT_EPSILON)
#             for _L in conv_dict.keys(): #iterate over the mlps/angular numbers
#                 mlp = layer_dict['mlp'][_L]
#                 p_array = mlp(inputs=conv_dict[_L], epsilon=epsilon) # pass convolved arrays through mlp
#                 out_tensor_dict[_L] = p_array # populate out dict
#             in_tensor_dict = out_tensor_dict
#
#         # extract and update
#         scales = in_tensor_dict[1][:,0,:]
#         translations = in_tensor_dict[1][:,1,:]
#         return jnp.array([velocities, scales * velocities + translations]), scales, translations

"""
Vector Modules
"""
class VectorModule(hk.Module):
    """
    a vector module for SE(3) equivariance
    """
    def __init__(self,
                 feature_dictionary : dict,
                 conv_dict_list : Iterable,
                 mlp_dict_list : Iterable,
                 SinusoidalBasis_kwargs : dict,
                 time_convolution_kwargs : dict,
                 epsilon : Optional[float] = tfn.DEFAULT_EPSILON,
                 mask_value : Optional[float] = 0.,
                 name : Optional[str] = f"vector"):
        super().__init__(name=name)
        self.SinusoidalBasis_module = tfn.SinusoidalBasis(**SinusoidalBasis_kwargs)

        num_particles, input_L0_channel_dimension = feature_dictionary[0].shape[:2]
        self._num_particles = num_particles
        self._input_L0_channel_dimension = input_L0_channel_dimension
        self._epsilon = epsilon
        self._mask_value = mask_value
        self._feature_dictionary = feature_dictionary

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
                 y : Array,
                 ):
        positions, velocities = y[0], y[1]

        #simple check on feature dictionary
        if self._feature_dictionary[0].shape != (self._num_particles, self._input_L0_channel_dimension, 1):
            raise ValueError(f"""the feature dictionary's L=0 input ({feature_dictionary[0].shape})
                                 is not equal to the specified channel dimension
                                 ({(self._num_particles, self._input_L0_channel_dimension, 1)})""")
        elif self._feature_dictionary[1] is not None:
            raise ValueError(f"the input feature dictionary should have no annotations on the L=1 input")

        if positions.shape[0] != self._num_particles:
            raise ValueError(f"the number of positions ({positions}) is not equal to the number of particles")
        if positions.shape != velocities.shape:
            raise ValueError(f"the position dimension does not match the velocity dimension")

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

        for layer_idx in range(len(self.layers)):
            out_tensor_dict = {}
            layer_dict = self.layers[layer_idx]
            conv_dict = layer_dict['conv'](in_tensor_dict=in_tensor_dict, rbf_inputs=rbf_inputs, unit_vectors=unit_r_ij, r_ij = norms, epsilon = self._epsilon)
            for _L in conv_dict.keys(): #iterate over the mlps/angular numbers
                mlp = layer_dict['mlp'][_L]
                p_array = mlp(inputs=conv_dict[_L], epsilon=self._epsilon) # pass convolved arrays through mlp
                out_tensor_dict[_L] = p_array # populate out dict
            in_tensor_dict = out_tensor_dict

        # extract and update
        scales = in_tensor_dict[1][:,0,:]
        translations = in_tensor_dict[1][:,1,:]
        return jnp.array([velocities, scales * velocities + translations]), scales, translations

"""
VectorModule helpers
"""
def make_diff_fn_inits(feature_dictionary : dict,
                       SinusoidalBasis_kwargs : Optional[dict] = {'r_switch': 2., 'r_cut': 2.5},
                       num_layers : Optional[int] = 1,
                       time_convolution_mlp_kwargs : Optional[dict] = {'output_sizes': [4,4], 'activation': jax.nn.swish},
                       conv_mlp_kwargs : Optional[dict] = {'output_sizes': [4,4], 'activation': jax.nn.swish},
                       tf_mlp_kwargs : Optional[dict] = {'output_sizes': [4,4], 'nonlinearity': jax.nn.swish},
                       epsilon : Optional[dict] = tfn.DEFAULT_EPSILON,
                       mask_output : Optional[bool] = False,
                       mask_value : Optional[float] = 0.):
    """
    function to build the kwarg dict for `VectorModule`
    """
    import copy
    num_particles, input_L0_channel_dimension = feature_dictionary[0].shape[:2] # pull the input_L0_channel dimension
    assert input_L0_channel_dimension % 2 == 0, f"the input_L0_channel_dimension must be even"
    assert input_L0_channel_dimension == tf_mlp_kwargs['output_sizes'][-1] // 2, f"the input L0 channel dimension must be half of the tf_mlp outputs"
    assert conv_mlp_kwargs['output_sizes'][-1] == tf_mlp_kwargs['output_sizes'][-1], f"we use consistent output dimension convolutions for ease of use"
    time_convolution_mlp_kwargs['output_sizes'].append(int(input_L0_channel_dimension))

    conv_shapes_dict = {layer_idx : copy.deepcopy(conv_mlp_kwargs) for layer_idx in range(num_layers)}
    tf_mlp_shapes_dict = {layer_idx : {0 : copy.deepcopy(tf_mlp_kwargs), 1: copy.deepcopy(tf_mlp_kwargs)} for layer_idx in range(num_layers)}
    tf_mlp_shapes_dict[num_layers-1][0]['output_sizes'] = tf_mlp_kwargs['output_sizes'] + [1]
    tf_mlp_shapes_dict[num_layers-1][1]['output_sizes'] = tf_mlp_kwargs['output_sizes'] + [2]

    conv_switching_fn = partial(polynomial_switching_fn,
                                 r_cutoff = SinusoidalBasis_kwargs['r_cut'],
                                 r_switch = SinusoidalBasis_kwargs['r_switch'])

    for tf_mlp_layer_idx in tf_mlp_shapes_dict.keys():
        tf_Ls = set(tf_mlp_shapes_dict[tf_mlp_layer_idx].keys())
        expected_Ls = set(range(2))
        if not expected_Ls.issubset(tf_Ls):
            raise ValueError(f"""tf_mlp layer {tf_mlp_layer_idx}  provided with Ls {tf_Ls} must be a subset of {expected_Ls};
            it does not have a mlp for each convolution layer""")

    combination_dict = {0: {0: [0], 1: [1]}, 1: {0: [1], 1: [0,1]}}

    # construct conv dicts
    conv_dict_list = [{'filter_mlp_dicts': {_L: conv_shapes_dict[idx] for _L in range(2)},
                       'name' : f"tfn_convolution_{idx}",
                       'mask_output': mask_output,
                       'switching_fn': conv_switching_fn,
                       'combination_dict' : combination_dict} for idx in range(num_layers)]
    # construct mlp dicts
    mlp_dict_list = tf_mlp_shapes_dict
    for layer_idx in mlp_dict_list.keys():
        layer_dict = mlp_dict_list[layer_idx]
        for L_dict_idx in layer_dict:
            L_dict = layer_dict[L_dict_idx]
            L_dict['L'] = L_dict_idx
            L_dict['name'] = f"tf_mlp_layer_{layer_idx}_L{L_dict_idx}"

    VectorModule_kwargs = {'feature_dictionary' : feature_dictionary,
                           'conv_dict_list': conv_dict_list,
                           'mlp_dict_list': mlp_dict_list,
                           'SinusoidalBasis_kwargs': SinusoidalBasis_kwargs,
                           'time_convolution_kwargs': time_convolution_mlp_kwargs,
                           'epsilon': epsilon,
                           'mask_value' : mask_value,
                           'name' : 'vector_module'}
    return VectorModule_kwargs
