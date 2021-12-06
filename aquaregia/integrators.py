"""library of kernels"""
# Imports
from typing import Sequence, Callable, Dict, Tuple, Optional, Union, Any, NamedTuple
from jraph._src.utils import ArrayTree
import jax
import flax.linen as nn
import jax.numpy as jnp
from functools import partial
from jax import grad, vmap, jit, random
import numpy as onp
import jax_md

# configure float64 by default
from jax.config import config
config.update("jax_enable_x64", True)

# Typing
Conf = Params = Array = Seed = jnp.array
from jraph._src.models import ArrayTree
NoneType = type(None)
from aquaregia.openmm import EnergyFn
import jax_md
from jax_md.partition import NeighborFn, NeighborList
from simtk import openmm, unit

def kinetic_energy(vs, mass):
    def ke(_v, _mass):
        return 0.5 * _v.dot(_v) / _mass
    return vmap(ke, in_axes=(0,0))(vs, mass).sum()

def metropolize_bool(reduced_work : float, # unitless
                     seed: Array # random seed
                     ) -> bool:
    """from a (unitless) work value and a seed, return accept/reject"""
    log_acceptance_prob = jnp.min(Array([0., -reduced_work]))
    lu = jnp.log(random.uniform(seed))
    accept = (lu <= log_acceptance_prob)
    return accept

def V_update(xs : Array,
             vs : Array,
             neighbor_list : NeighborList,
             potential_energy_fn : EnergyFn,
             potential_energy_params : ArrayTree,
             dt : float,
             mass : Array) -> Array:
    out_vs = vs + -grad(potential_energy_fn)(xs, neighbor_list, potential_energy_params) * dt / mass[..., jnp.newaxis]
    return out_vs

def R_update(xs : Array,
             vs : Array,
             dt : float,
             shift_fn : jax_md.space.ShiftFn) -> Array:

    return vmap(shift_fn, in_axes=(0,0))(xs, vs * dt)

def O_update(vs : Array,
             noise_seed : Seed,
             mass : Array,
             a : float,
             b : float,
             kT : float) -> Array:
    n, d = vs.shape
    return a * vs + b * jnp.sqrt(kT / mass[..., jnp.newaxis]) * random.normal(noise_seed, shape=(n,d))

# def make_static_BAOAB_kernel(potential_energy_fn : EnergyFn,
#                              neighbor_fn : NeighborFn,
#                              dt : float,
#                              gamma : float,
#                              mass : Array,
#                              shift_fn : jax_md.space.ShiftFn,
#                              kinetic_energy_fn : Optional[EnergyFn] = kinetic_energy,
#                              get_shadow_work : Optional[Union[bool, str]] = False,
#                              ) -> Callable[Tuple[Array, Array, Array, ArrayTree, float], Tuple[Array, Array, float]]:
#     """
#     function that returns a static BAOAB kernel
#     TODO : support neighbor lists.
#     """
#     a, b = jnp.exp(-gamma * dt), jnp.sqrt(1. - jnp.exp(-2. * gamma * dt)) # get the a and b parameters
#
#     #partial kernel fns
#     partial_V_update = partial(V_update,
#                                potential_energy_fn=potential_energy_fn,
#                                dt = dt/2.,
#                                mass = mass)
#     partial_R_update = partial(R_update, dt = dt/2., shift_fn=shift_fn)
#     partial_O_update = partial(O_update, mass = mass, a = a, b = b)
#     partial_ke = partial(kinetic_energy_fn, mass = mass)
#
#     if get_shadow_work is True:
#         ke_fn = partial_ke
#         pe_fn = potential_energy_fn
#     elif get_shadow_work is False:
#         ke_fn = lambda x: 0.
#         pe_fn = lambda x, y, z: 0. # this now generically takes 3 args
#     elif get_shadow_work == 'heat': # return the kinetic energy work of the step
#         ke_fn = partial_ke
#         pe_fn = lambda x, y, z: 0.
#     else:
#         raise ValueError(f"the argument {get_shadow_work} is not supported")
#
#     def run(xs : Array,
#             vs : Array,
#             seed : Array,
#             neighbor_list : NeighborList,
#             potential_energy_params : ArrayTree,
#             kT : float) -> Tuple[Array, Array, float]:
#         """returns new xs, vs, and the (unit'd) shadow work if specified; otherwise is zero"""
#         neighbor_list = neighbor_fn(xs, neighbor_list)
#         e0 = ke_fn(vs) + pe_fn(xs, neighbor_list, potential_energy_params)
#         vs1 = partial_V_update(xs, vs, neighbor_list = neighbor_list, potential_energy_params=potential_energy_params) #V
#         xs1 = partial_R_update(xs, vs1) #R
#         neighbor_list = neighbor_fn(xs1, neighbor_list)
#         ke0 = ke_fn(vs1)
#         vs2 = partial_O_update(vs = vs1, noise_seed=seed, kT=kT) #O
#         ke1 = ke_fn(vs2)
#         xs2 = partial_R_update(xs1, vs2) #R
#         neighbor_list = neighbor_fn(xs2, neighbor_list)
#         vs3 = partial_V_update(xs2, vs2, neighbor_list = neighbor_list, potential_energy_params = potential_energy_params) #V
#         e1 = ke_fn(vs3) + pe_fn(xs2, neighbor_list, potential_energy_params)
#
#         return xs2, vs3, e1 - e0 - (ke1 - ke0)
#
#     return run

def BAOAB_coeffs(kT, dt, gamma, masses):
    scale = jnp.sqrt(kT / masses[..., jnp.newaxis])
    a = jnp.exp(-gamma * dt)
    b = dt / masses[..., jnp.newaxis]
    c = jnp.sqrt(1. - jnp.exp(-2. * gamma * dt)) * scale
    return a,b,c

def make_static_BAOAB_kernel(potential_energy_fn : EnergyFn,
                             dt : float,
                             gamma : float,
                             mass : Array,
                             shift_fn : jax_md.space.ShiftFn,
                             ) -> Callable[Tuple[Array, Array, Array, NeighborList, ArrayTree, float], Tuple[Array, Array, float]]:
    def run(xs : Array,
            vs : Array,
            seed : Array,
            neighbor_list : NeighborList, #this is already equipped with xs
            potential_energy_params : ArrayTree,
            kT : float) -> Tuple[Array, Array, float]:
        a,b,c = BAOAB_coeffs(kT, dt, gamma, mass)
        mid_v = vs + b * -grad(potential_energy_fn)(xs, neighbor_list, potential_energy_params)
        new_v = a * mid_v + c * random.normal(seed, shape = xs.shape)
        new_x = vmap(shift_fn, in_axes=(0,0))(xs, 0.5 * dt * (mid_v + new_v))
        return new_x, new_v
    return run

def make_folded_integrator(integrator,
                           neighbor_fns,
                           mod_potential_params_fn,
                           mod_kT_fn,
                           potential_energy_fn
                          ):
    """make a folded integrator; do not jit this"""

    def scan_int(carry, x):
        """carry is xs, vs, start_pe, seed"""
        in_xs, in_vs, neighbor_list, in_seed = carry #open the carry
        out_seed, run_seed = random.split(in_seed) # split the in_seed

        kT = mod_kT_fn(x) #get kT
        potential_energy_params = mod_potential_params_fn(x) # get potential energy params

        neighbor_list = neighbor_fns.update(in_xs, neighbor_list)
        out_xs, out_vs = integrator(xs = in_xs,
                                    vs = in_vs,
                                    seed = run_seed,
                                    neighbor_list = neighbor_list,
                                    potential_energy_params = potential_energy_params,
                                    kT = kT) # run the integrator

        return (out_xs, out_vs, neighbor_list, out_seed), None

    @jit
    def folded_integrator(xs, vs, neighbor_list, seed, sequence):
        in_carry = (xs, vs, neighbor_list, seed) #no need to jit
        out_carry, _ = jax.lax.scan(scan_int, in_carry, sequence)
        out_xs, out_vs, neighbor_list, seed = out_carry #no need to jit
        return out_xs, out_vs, neighbor_list
    return folded_integrator

def get_folded_equilibrium_integrator(potential_energy_fn : EnergyFn,
                                      neighbor_fns : NeighborFn,
                                      potential_energy_parameters : ArrayTree,
                                      kT : float,
                                      dt : float,
                                      gamma : float,
                                      mass : Array,
                                      shift_fn : jax_md.space.ShiftFn):
    """make an equilibrium simulator"""
    base_integrator = make_static_BAOAB_kernel(potential_energy_fn = potential_energy_fn,
                                               dt = dt,
                                               gamma = gamma,
                                               mass = mass,
                                               shift_fn = shift_fn)

    def mod_potential_params_fn(*args): return potential_energy_parameters
    def mod_kT_fn(*args): return kT
    def dummy_potential_energy_fn(*args): return 0.

    folded_integrator = make_folded_integrator(integrator = base_integrator,
                                               neighbor_fns = neighbor_fns,
                                               mod_potential_params_fn = mod_potential_params_fn,
                                               mod_kT_fn = mod_kT_fn,
                                               potential_energy_fn = dummy_potential_energy_fn)
    return folded_integrator

# utility to thermalize velocities
def thermalize(seed, masses, kT, dimension):
    """the mean is 0 and the standard deviation is sqrt(kT/m) for each particle."""
    std_devs = jnp.sqrt(kT / masses)
    out = random.normal(seed, shape=(len(masses), dimension)) * std_devs[..., jnp.newaxis]
    return out

def get_nonequilibrium_integrator(potential_energy_fn : EnergyFn,
                                  potential_energy_parameters : ArrayTree,
                                  kT : float,
                                  dt : float,
                                  gamma : float,
                                  mass : Array,
                                  mod_potential_params_fn : Callable[[float], ArrayTree], # potential energy protocol,
                                  mod_kT_fn : Callable[[float], float], # kT protocol (effective temperature)
                                  shift_fn : jax_md.space.ShiftFn,
                                  kinetic_energy_fn : Optional[EnergyFn] = kinetic_energy,
                                  get_shadow_work : Optional[bool] = False):
    """make an equilibrium simulator"""
    base_integrator = make_static_BAOAB_kernel(potential_energy_fn = potential_energy_fn,
                             dt = dt,
                             gamma = gamma,
                             mass = mass,
                             shift_fn = shift_fn,
                             kinetic_energy_fn = kinetic_energy_fn,
                             get_shadow_work = get_shadow_work,
                             )


    folded_integrator = make_folded_integrator(integrator = base_integrator,
                                               mod_potential_params_fn=mod_potential_params_fn,
                                               mod_kT_fn=mod_kT_fn,
                                               potential_energy_fn=potential_energy_fn)
    return folded_integrator
