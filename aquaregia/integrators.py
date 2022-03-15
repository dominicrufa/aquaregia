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

# utility to thermalize velocities
def thermalize(seed, masses, kT, dimension):
    """the mean is 0 and the standard deviation is sqrt(kT/m) for each particle."""
    std_devs = jnp.sqrt(kT / masses)
    out = random.normal(seed, shape=(len(masses), dimension)) * std_devs[..., jnp.newaxis]
    return out

class BaseIntegratorGenerator(object):
    """
    create an integrator
    """
    def __init__(self,
                 canonical_u_fn : Callable,
                 neighbor_list : NeighborList,
                 dt : float,
                 masses : Array,
                 kT : float,
                 shift_fn : Callable,
                 neighbor_list_update_fn : Optional[Callable] = None,
                 ):
        from aquaregia.integrators import thermalize
        self._canonical_u_fn = canonical_u_fn
        self._neighbor_list_template = neighbor_list
        self._shift_fn = shift_fn
        self._thermalizer = partial(thermalize, masses = masses, kT = kT, dimension = 3)

        if neighbor_list_update_fn is None:
            self._neighbor_list_update_fn = lambda x,y : self._neighbor_list_template
        else:
            self._neighbor_list_update_fn = neighbor_list_update_fn

        self._dt = dt
        self._kT = kT
        self._masses = masses

    def integrator(self, *args, **kwargs) -> Callable:
        """
        get a jittable integrator fn
        """
        raise NotImplementedError()

class BAOABIntegratorGenerator(BaseIntegratorGenerator):
    """
    generate a BAOAB integrator
    """
    def __init__(self,
                 canonical_u_fn : Callable,
                 neighbor_list : NeighborList,
                 dt : float,
                 masses : Array,
                 kT : float,
                 shift_fn : Callable,
                 collision_rate : float,
                 neighbor_list_update_fn : Optional[Callable] = None,
                 ):
        super().__init__(canonical_u_fn, neighbor_list, dt, masses, kT, shift_fn, neighbor_list_update_fn)
        self._collision_rate = collision_rate

    def integrator(self,
                   num_steps : int,
                   remove_neighbor_list : Optional[bool] = False,
                   **kwargs):
        from aquaregia.integrators import BAOAB_coeffs, make_static_BAOAB_kernel

        # single_step_integrator : Callable[[xs, vs, seed, neighbor_list, u_params, kT], [xs, vs]]
        single_step_integrator = make_static_BAOAB_kernel(potential_energy_fn=self._canonical_u_fn,
                                                          dt = self._dt,
                                                          gamma = self._collision_rate,
                                                          mass = self._masses,
                                                          shift_fn = self._shift_fn)

        def scanner(carry, x):
            """carry xs, vs, neighbor_list, u_params, seed"""
            in_xs, in_vs, neighbor_list, u_params, seed = carry
            out_seed, run_seed = jax.random.split(seed)
            neighbor_list = self._neighbor_list_update_fn(in_xs, neighbor_list)
            out_xs, out_vs = single_step_integrator(in_xs, in_vs, run_seed, neighbor_list, u_params, kT=self._kT)
            return (out_xs, out_vs, neighbor_list, u_params, out_seed), None

        def folded_integrator(xs, u_params, seed, neighbor_list, sequence):
            therm_seed, seed = jax.random.split(seed)
            in_carry = (xs, self._thermalizer(therm_seed), neighbor_list, u_params, seed)
            (out_xs, out_vs, neighbor_list, u_params, out_seed), _ = jax.lax.scan(scanner, in_carry, sequence)
            return {'xs' : out_xs, 'vs' : out_vs, 'neighbor_list': neighbor_list} # we can add to this...

        if remove_neighbor_list:
            out = partial(folded_integrator, neighbor_list = self._neighbor_list_template, sequence = jnp.arange(num_steps))
        else:
            out = partial(folded_integrator, sequence = jnp.arange(num_steps))
        return out
