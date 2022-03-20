"""free energy calculation modules"""
from typing import Sequence, Callable, Dict, Tuple, Optional, NamedTuple, Any
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import tqdm
import time
import pymbar
import warnings

from jax.config import config
config.update("jax_enable_x64", True)
from aquaregia.utils import Array, ArrayTree
from aquaregia.openmm import DEFAULT_VACUUM_R_CUTOFF
from jax_md.partition import NeighborList, NeighborFn
from jax.flatten_util import ravel_pytree


def bool_vector_element_fn(_bool : bool,
                           _vector_element : float,
                           _lambda : float,
                           true_fn : Optional[Callable[[float, float], float]] = lambda _lam, _val: _lam*_val,
                           false_fn : Optional[Callable[[float, float], float]] = lambda _lam, _val: _val):
    """a simple vmappable function for performing operation with logic on protocols"""
    return jax.lax.cond(_bool, true_fn, false_fn, _lambda, _vector_element)

class FECProtocol(object):
    """
    generic class to produce protocol parameter dictionaries

    includes the following properties:
        parameter_dict : a vectorized (along leading axis) version of the `coupled_parameter_dict` that is vmappable
                         to a canonical `energy_function`
        vmap_axes_dict : a vectorized version of the parameter dict that specifies which axes to vmap
    """
    def __init__(self,
                 coupled_parameter_dict : dict,
                 lambdas : Optional[Array] = jnp.linspace(1., 0., 10)
                ):
        """
        Args:
            coupled_parameter_dict : canonical energy fn dict
            lambdas : an array of lambdas (typically from 1 to 0 (i.e. interacting to noninteracting))
        """
        flat_params, back_fn = ravel_pytree(coupled_parameter_dict)

        self._back_fn = back_fn
        self._flat_params = flat_params
        self._parameter_dict_template = back_fn(flat_params) # tricky way of making a copy
        self._lambdas = lambdas

    def _get_vmap_axes_dict(self):
        raise NotImplementedError(f"vmap_axes_dict is not implemented")

    def _get_parameter_dict(self):
        raise NotImplementedError(f"parameter dict is not implemented")

    @property
    def vmap_axes_dict(self):
        return self._get_vmap_axes_dict()

    @property
    def parameter_dict(self):
        return self._get_parameter_dict()

class AbsoluteFECProtocol(FECProtocol):
    """
    an alchemical protocol that will
        1. lift a prespecified index set into a 4th dimension (from lambda 1->0) to r_cutoff with a specified w_lift_fn
        2. optionally annihilate sterics
        3. optionally annihilate electrostatics
        4. do (3,4) with a scale function argument

    TODO : support lifting between ValenceForces as well as Nonbonded Exceptions (we currently don't do this.)
    """
    def __init__(self,
                 coupled_parameter_dict : dict,
                 alchemical_particle_indices : Sequence,
                 w_lambdas : Optional[Array] = jnp.linspace(1., 0., 10),
                 r_cutoff : Optional[float] = DEFAULT_VACUUM_R_CUTOFF,
                 annihilate_sterics : Optional[bool] = False,
                 annihilate_electrostatics : Optional[bool] = True,
                 electrostatics_lambdas : Optional[Array] = jnp.linspace(1., 0., 10),
                 sterics_lambdas : Optional[Array] = jnp.linspace(1., 0., 10)
                ):

        super().__init__(coupled_parameter_dict = coupled_parameter_dict, lambdas = w_lambdas)
        self._w_lambdas = self._lambdas
        self._electrostatics_lambdas = electrostatics_lambdas
        self._sterics_lambdas = sterics_lambdas

        self._r_cutoff = r_cutoff
        self._alchemical_particle_indices = alchemical_particle_indices
        self._alchemical_mask = Array([idx in alchemical_particle_indices for idx in range(coupled_parameter_dict['NonbondedForce']['standard']['charge'].shape[0])])
        self._annihilate_sterics = annihilate_sterics
        self._annihilate_electrostatics = annihilate_electrostatics

        # the linear w lift fn: in both cases, the argument order is [mask, param_vec, lambdas]
        self._w_lift_fn = jax.vmap(jax.vmap(partial(bool_vector_element_fn,
                                  true_fn = lambda _lam, _val: (1. - _lam) * _val, # here, _val is `r_cutoff`
                                  false_fn = lambda _lam, _val: 0.), in_axes=(0, 0, None)), in_axes=(None, None, 0))

        self._linear_scale_fn = jax.vmap(jax.vmap(partial(bool_vector_element_fn,
                                  true_fn = lambda _lam, _val: _lam * _val, # here, _val is the parameter matrix
                                  false_fn = lambda _lam, _val: _val), in_axes=(0, 0, None)), in_axes=(None, None, 0))

        if self._annihilate_electrostatics:
            self._electrostatics_scale_fn = self._linear_scale_fn

        if annihilate_sterics:
            self._sterics_scale_fn = self._linear_scale_fn

    def _annihilate_electrostatics_parameterizer(self, parameter_dict):
        template_charges = self._parameter_dict_template['NonbondedForce']['standard']['charge']
        template_chargeProds = self._parameter_dict_template['NonbondedForce']['exceptions']['chargeProd']
        new_charges = self._electrostatics_scale_fn(self._alchemical_mask, template_charges, self._electrostatics_lambdas)
        new_chargeProds = self._electrostatics_scale_fn(self._alchemical_mask, template_chargeProds, self._electrostatics_lambdas)
        parameter_dict['NonbondedForce']['standard']['charge'] = new_charges
        parameter_dict['NonbondedForce']['exceptions']['chargeProd'] = new_chargeProds
        return parameter_dict

    def _annihilate_electrostatics_vmapper(self, axes_dict):
        axes_dict['NonbondedForce']['standard']['charge'] = 0
        axes_dict['NonbondedForce']['exceptions']['chargeProd'] = 0
        return axes_dict

    def _annihilate_sterics_parameterizer(self, parameter_dict):
        template_epsilons = self._parameter_dict_template['NonbondedForce']['standard']['epsilon']
        template_exception_epsilons = self._parameter_dict_template['NonbondedForce']['exceptions']['epsilon']
        new_epsilons = self._sterics_scale_fn(self._alchemical_mask, template_epsilons, self._sterics_lambdas)
        new_exception_epsilons = self._sterics_scale_fn(self._alchemical_mask, template_exception_epsilons, self._sterics_lambdas)
        parameter_dict['NonbondedForce']['standard']['epsilon'] = new_epsilons
        parameter_dict['NonbondedForce']['exceptions']['epsilon'] = new_exception_epsilons
        return parameter_dict

    def _annihilate_sterics_vmapper(self, axes_dict):
        axes_dict['NonbondedForce']['standard']['epsilon'] = 0
        axes_dict['NonbondedForce']['exceptions']['epsilon'] = 0
        return axes_dict


    def _get_parameter_dict(self):
        parameter_dict = self._back_fn(self._flat_params)

        # ws (always lift)
        ws = self._parameter_dict_template['NonbondedForce']['standard']['w']
        new_ws = self._w_lift_fn(self._alchemical_mask, jnp.ones_like(ws) * self._r_cutoff, self._w_lambdas)
        parameter_dict['NonbondedForce']['standard']['w'] = new_ws

        # electrostatics
        if self._annihilate_electrostatics:
            parameter_dict = self._annihilate_electrostatics_parameterizer(parameter_dict)
        if self._annihilate_sterics:
            parameter_dict = self._annihilate_sterics_parameterizer(parameter_dict)

        return parameter_dict


    def _get_vmap_axes_dict(self):
        if 'NonbondedForce' not in self._parameter_dict_template.keys():
            raise KeyError(f"not `NonbondedForce` supplied")

        #make the vmap param dict
        vmap_param_dict = {key: None for key in self._parameter_dict_template.keys()}
        vmap_param_dict['NonbondedForce'] = {key: None for key in self._parameter_dict_template['NonbondedForce'].keys()}
        vmap_param_dict['NonbondedForce']['standard'] = {key: None for key in self._parameter_dict_template['NonbondedForce']['standard'].keys()}

        # ws (always lift)
        vmap_param_dict['NonbondedForce']['standard']['w'] = 0

        if self._annihilate_electrostatics or self._annihilate_sterics: # accommodate exception keys
            vmap_param_dict['NonbondedForce']['exceptions'] = {key: None for key in self._parameter_dict_template['NonbondedForce']['exceptions'].keys()}

        if self._annihilate_electrostatics:
            vmap_param_dict = self._annihilate_electrostatics_vmapper(vmap_param_dict)
        if self._annihilate_sterics:
            vmap_param_dict = self._annihilate_sterics_vmapper(vmap_param_dict)
        return vmap_param_dict

class BaseReplicaExchangeSampler(object):
    """
    a base class for replica exchange;
    """
    def __init__(self,
                 num_states,
                 canonical_u_fn : Callable[[Array, NeighborList, ArrayTree], float], # canonical u_fn,
                 kT : float,
                 mappable_u_params : Array, # vmappable u_params
                 mappable_u_params_axes : ArrayTree, # axes of vmappable u_params
                 propagator : Callable,
                 MCMC_save_interval : int
                 ):
        # prep
        self._num_states = num_states
        self._canonical_u_fn = canonical_u_fn
        self._mappable_u_params = mappable_u_params
        self._mappable_u_params_axes = mappable_u_params_axes
        self._propagator = jax.jit(propagator)
        self._kT = kT
        self._MCMC_save_interval = MCMC_save_interval

        # query the number of windows
        leaves = jax.tree_leaves(mappable_u_params_axes)
        int_leaves = [leaf for leaf in leaves if leaf is not None]
        if len(int_leaves) == 0: raise ValueError(f"there must be multiple states; recovered 0 states")
        self._iteration = 0

    def _propagate(self, *args, **kwargs):
        """propagate all replicas"""
        raise NotImplementedError()

    def _mix_replicas(self, *args, **kwargs):
        """mix replicas"""
        raise NotImplementedError()

    def _report(self):
        """an internal reporter function that will update the save state"""
        raise NotImplementedError()

    def execute(self):
        """run an iteration of the MCMC sampler; all parameters are maintained internally"""
        raise NotImplementedError()


def is_nan_safe(arr : Array):
    if jnp.isnan(arr).any():
        out=False
    else:
        out=True
    return out

def metropolize_bool(reduced_work : float, # unitless
                     seed: Array # random seed
                     ) -> bool:
    """from a (unitless) work value and a seed, return accept/reject"""
    log_acceptance_prob = jnp.min(Array([0., -reduced_work]))
    lu = jnp.log(jax.random.uniform(seed))
    accept = (lu <= log_acceptance_prob)
    return accept

def accept_metr_bool_fn(accept_reporter, xs, energy_matrix, i,j):
    out_accept_reporter = accept_reporter.at[([i,j], [j,i])].add(1)
    out_xs = xs.at[(i,j),:,:].set(xs[(j,i),:,:])
    out_energy_matrix = energy_matrix.at[(i,j),:].set(energy_matrix[(j,i),:])
    return out_accept_reporter, out_xs, out_energy_matrix

def reject_metr_bool_fn(accept_reporter, xs, energy_matrix, i,j):
    return accept_reporter, xs, energy_matrix

def _uniform_replica_mix_scanner(carry, _xs):
    energy_matrix, xs, accept_reporter, total_mix_reporter, seed = carry
    num_states = energy_matrix.shape[0] # get the number of states
    run_seed, met_seed, seed = jax.random.split(seed, num=3)
    i,j = jax.random.randint(run_seed, shape=(2,), minval=0, maxval = num_states) # max is exclusive
    u_ii, u_jj, u_ij, u_ji = energy_matrix[i,i], energy_matrix[j,j], energy_matrix[i,j], energy_matrix[j,i]
    out_total_mix_reporter = total_mix_reporter.at[([i,j], [j,i])].add(1)
    reduced_work = (u_ij - u_ii) + (u_ji - u_jj)
    accept_bool = metropolize_bool(reduced_work, met_seed)
    out_accept_reporter, out_xs, out_energy_matrix = jax.lax.cond(accept_bool,
                                                                  accept_metr_bool_fn, # true_fn
                                                                  reject_metr_bool_fn, # false_fn
                                                                  accept_reporter,
                                                                  xs,
                                                                  energy_matrix,
                                                                  i,
                                                                  j)
    return (out_energy_matrix, out_xs, out_accept_reporter, out_total_mix_reporter, seed), None

def compute_mbar(u_matrix_array, decorrelate_timeseries=False, skip_interval=1, **mbar_kwargs):
    """
    run pymbar.mbar.MBAR
    """
    from pymbar import timeseries
    from pymbar.mbar import MBAR

    if decorrelate_timeseries:
        raise NotImplementedError(f"timeseries decorrelation is not currently implemented; at present we assume decorrelation")

    # take nth matrix
    u_matrices = u_matrix_array[::skip_interval]

    #restructure for pymbar
    N_k = np.array([u_matrices.shape[0]] * u_matrices.shape[-1])
    flat_u_matrix = np.concatenate(u_matrices, axis=0)
    u_kn = np.transpose(flat_u_matrix)
    res = MBAR(u_kn, N_k, **mbar_kwargs)
    return res

def compute_acceptance(accept_matrices_arr, total_mix_matrices_arr):
    """
    compute the total acceptance rate and total acceptance rate matrix
    """
    full_accept_matrix, total_mix_matrix = accept_matrices_arr.sum(axis=0), total_mix_matrices_arr.sum(axis=0)
    zero_attempts_mask = np.where(total_mix_matrix == 0., True, False)
    accept_rate_matrix = np.nan_to_num(full_accept_matrix / total_mix_matrix) # make nans go to zero since there were no attempts
    full_accept_rate = full_accept_matrix.sum() / total_mix_matrix.sum()
    return full_accept_rate, accept_rate_matrix


class NonCanonicalUniformLogPReplicaExchangeSampler(BaseReplicaExchangeSampler):
    """
    basic replica exchange sampler where the logp_proposal is uniform and velocities are resampled; there are also no neighbor lists
    """
    def __init__(self,
             num_states : int,
             canonical_u_fn : Callable[[Array, NeighborList, ArrayTree], float], # noncanonical
             kT : float,
             mappable_u_params : Array, # vmappable u_params
             mappable_u_params_axes : ArrayTree, # axes of vmappable u_params
             propagator : Callable, # propagator class
             MCMC_save_interval : int,
             neighbor_list_template : NeighborList):

        super().__init__(num_states = num_states,
                         canonical_u_fn = canonical_u_fn,
                         kT = kT,
                         mappable_u_params = mappable_u_params,
                         mappable_u_params_axes = mappable_u_params_axes,
                         propagator = propagator,
                         MCMC_save_interval = MCMC_save_interval)

        self._neighbor_list_template = neighbor_list_template
        self._u_fn = lambda x, y : self._canonical_u_fn(x, neighbor_list_template, y) # a simple partial out of neighbor_list
        self._row_u_fn = jax.vmap(self._u_fn, in_axes = (None, self._mappable_u_params_axes)) # give a single position array, get energy at all states
        self._matrix_u_fn = jax.jit(jax.vmap(self._row_u_fn, in_axes=(0,None))) # give array of positions, get energy at all states [i,j]
        self._scanner_fn = jax.jit(_uniform_replica_mix_scanner)

        self._reporter_dict = {'energy_matrices' : [],
                               'total_mix_reporter' : [],
                               'accept_reporter' : [],
                               'xs' : [],
                               'propagation_times': [],
                               'energy_matrix_compute_times': [],
                               'mixing_times': []}

    def _propagate(self, batch_xs, seed):
        start_time = time.perf_counter()
        out_prop_dict = jax.vmap(self._propagator, in_axes=(0, self._mappable_u_params_axes,0))(batch_xs,
                                                                                                         self._mappable_u_params,
                                                                                                         jax.random.split(seed, num=self._num_states))
        end_time = time.perf_counter()
        self._reporter_dict['propagation_times'].append(end_time - start_time)

        xs, vs = out_prop_dict['xs'], out_prop_dict['vs']
        xs_nan_safe = is_nan_safe(xs)
        vs_nan_safe = is_nan_safe(vs)
        nan_safe = xs_nan_safe and vs_nan_safe
        return {'xs' : xs, 'vs' : vs, 'nan_safe' : nan_safe}

    def _mix_replicas(self, prop_dict, seed, num_swaps):
        # first compute energy matrix
        xs = prop_dict['xs']
        start_time = time.perf_counter()
        reference_energy_matrix = self._matrix_u_fn(xs, self._mappable_u_params) / self._kT # ixj where i is position @ j thermostate
        end_time = time.perf_counter()
        self._reporter_dict['energy_matrix_compute_times'].append(end_time - start_time)

        start_energy_matrix = jnp.nan_to_num(reference_energy_matrix, nan=np.inf) # make a manipulatable matrix

        # simple reporter
        accept_reporter, total_mix_reporter = jnp.zeros_like(start_energy_matrix), jnp.zeros_like(start_energy_matrix)
        carry = (start_energy_matrix, xs, accept_reporter, total_mix_reporter, seed)

        start_time = time.perf_counter()
        (out_energy_matrix, out_xs, out_accept_reporter, out_total_mix_reporter, _), _ = jax.lax.scan(self._scanner_fn, carry, jnp.arange(num_swaps))
        end_time = time.perf_counter()

        self._reporter_dict['mixing_times'].append(end_time - start_time)

        out_dict = {'xs' : out_xs,
                    'total_mix_reporter' : out_total_mix_reporter,
                    'accept_reporter' : out_accept_reporter,
                    'energy_matrix' : out_energy_matrix
                   }
        return out_dict

    def _report(self, mix_replicas_dict):
        """report in place"""
        if self._iteration % self._MCMC_save_interval == 0:
            self._reporter_dict['xs'].append(np.array(mix_replicas_dict['xs']))
            self._reporter_dict['total_mix_reporter'].append(mix_replicas_dict['total_mix_reporter'])
            self._reporter_dict['accept_reporter'].append(mix_replicas_dict['accept_reporter'])
            self._reporter_dict['energy_matrices'].append(np.array(mix_replicas_dict['energy_matrix']))

    def execute(self, xs, seed, num_iterations, num_swaps_per_mix_iteration):
        """execute"""
        if xs.shape[0] != self._num_states:
            raise ValueError(f"{xs.shape[0]} xs were provided, but this instance has {self._num_states} states")

        for _ in tqdm.trange(num_iterations):
            prop_seed, mix_seed, seed = jax.random.split(seed, num=3)
            propagation_dict = self._propagate(batch_xs=xs, seed=prop_seed)
            # check if nan_safe
            if not propagation_dict['nan_safe']:
                warnings.warn("iteration is not nan-safe; returning propagation dictionary")
                return propagation_dict
            mix_dict = self._mix_replicas(prop_dict=propagation_dict, seed=mix_seed, num_swaps=num_swaps_per_mix_iteration)
            self._report(mix_replicas_dict=mix_dict)
            xs = mix_dict['xs']
            self._iteration += 1
        return mix_dict

    def analyze(self, decorrelate_timeseries=False, skip_interval=1, **mbar_kwargs):
        """analyze the `_reporter_dict`"""
        import copy
        reporter = copy.deepcopy(self._reporter_dict)
        mbar_res = compute_mbar(u_matrix_array = np.array(reporter['energy_matrices']),
                                decorrelate_timeseries=decorrelate_timeseries,
                                skip_interval=1,
                                **mbar_kwargs)
        total_acceptance_rate, acceptance_matrix = compute_acceptance(accept_matrices_arr = np.array(reporter['accept_reporter']),
                                                                      total_mix_matrices_arr = np.array(reporter['total_mix_reporter']))
        return mbar_res, total_acceptance_rate, acceptance_matrix

# make a custom w_lift protocol
"""
the following was copied from https://github.com/proteneer/timemachine/pull/538.
"""
def validate_lambda_schedule(lambda_schedule, num_windows):
    """Must go monotonically from 0 to 1 in num_windows steps"""
    assert lambda_schedule[0] == 0.0
    assert lambda_schedule[-1] == 1.0
    assert len(lambda_schedule) == num_windows
    assert ((lambda_schedule[1:] - lambda_schedule[:-1]) > 0).all()

def interpolate_pre_optimized_protocol(pre_optimized_protocol, num_windows):
    xp = np.linspace(0, 1, len(pre_optimized_protocol))
    x_interp = np.linspace(0, 1, num_windows)
    lambda_schedule = np.interp(x_interp, xp, pre_optimized_protocol)
    validate_lambda_schedule(lambda_schedule, num_windows)
    return lambda_schedule

def construct_pre_optimized_absolute_lambda_schedule_solvent(num_windows):
    solvent_decoupling_protocol = np.array(
        [0., 0.02154097, 0.0305478, 0.03747918, 0.0432925, 0.04841349, 0.05303288, 0.05729336, 0.06128111, 0.0650162,
         0.06854392, 0.07186945, 0.07505386, 0.07809426, 0.08097656, 0.08378378, 0.08652228, 0.08910844, 0.09170097,
         0.09415532, 0.0965975, 0.09894146, 0.10125901, 0.10349315, 0.1057036, 0.10782406, 0.10995297, 0.11196338,
         0.11404105, 0.11597311, 0.11799029, 0.11989214, 0.12179616, 0.12367442, 0.12544245, 0.12730977, 0.12904358,
         0.13080329, 0.13255268, 0.13418286, 0.13594787, 0.13760607, 0.13920917, 0.14090233, 0.14247115, 0.14403571,
         0.14563762, 0.14712597, 0.14863463, 0.1501709, 0.1516045, 0.15306237, 0.15457974, 0.15599668, 0.15739867,
         0.1588833, 0.1602667, 0.16158698, 0.16306219, 0.16443643, 0.16571203, 0.1671053, 0.16844875, 0.16969885,
         0.17095515, 0.17229892, 0.17355947, 0.17474395, 0.17606238, 0.17735235, 0.1785562, 0.1797194, 0.18102615,
         0.18224503, 0.18338315, 0.18454735, 0.18579297, 0.18695968, 0.18805265, 0.18920557, 0.1904094, 0.1915372,
         0.1925929, 0.19370481, 0.19486737, 0.19595772, 0.19698288, 0.19803636, 0.1991899, 0.20028, 0.20131035,
         0.20232168, 0.20348772, 0.20458663, 0.2056212, 0.20659485, 0.20774405, 0.20884764, 0.20989276, 0.2108857,
         0.2120116, 0.21316817, 0.21427184, 0.21532528, 0.21650709, 0.21773745, 0.21890783, 0.22002229, 0.22133134,
         0.2226356, 0.22387771, 0.22515419, 0.22662608, 0.22803088, 0.22940172, 0.23108277, 0.2327005, 0.23438922,
         0.23634133, 0.23822652, 0.2405842, 0.24292293, 0.24588996, 0.24922462, 0.25322387, 0.25836924, 0.26533154,
         0.27964026, 0.29688698, 0.31934273, 0.34495637, 0.37706286, 0.4246625, 0.5712542, 1.])
    lambda_schedule = interpolate_pre_optimized_protocol(solvent_decoupling_protocol, num_windows)
    return Array(lambda_schedule)[::-1]
