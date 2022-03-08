"""
test the `aquaregia.integrators` library functionality
"""
from aquaregia.integrators import *

from openmmtools.testsystems import Diatom
from aquaregia.openmm import make_canonical_energy_fn
from aquaregia.utils import kinetic_energy, get_vacuum_neighbor_list
import tqdm
import pytest

# Constants
DEFAULT_TEMPERATURE = 300. #kelvin
DEFAULT_TIMESTEP = 1e-3 #ps
from openmmtools.constants import kB
kT = (DEFAULT_TEMPERATURE * unit.kelvin * kB).value_in_unit_system(unit.md_unit_system)

def get_diatom_parameters_dict():
    displacement_fn, shift_fn = jax_md.space.free()
    metric = jit(jax_md.space.canonicalize_displacement_or_metric(displacement_fn))
    diatom = Diatom()
    system, positions = diatom.system, diatom.positions.value_in_unit_system(unit.md_unit_system)
    masses = Array([system.getParticleMass(i).value_in_unit_system(unit.md_unit_system) for i in range(system.getNumParticles())], dtype=jnp.float64)

    xs, vs = Array(diatom.positions.value_in_unit_system(unit.md_unit_system), dtype=jnp.float64), thermalize(seed = random.PRNGKey(263), masses=masses, kT = kT, dimension=3)

    u_params, u_fn = make_canonical_energy_fn(system=system,
                                              displacement_fn=displacement_fn)

    vacuum_neighbor_list = get_vacuum_neighbor_list(system.getNumParticles())
    neighbor_fn = lambda x, y: vacuum_neighbor_list

    _dict = {'displacement_fn': displacement_fn,
             'shift_fn': shift_fn,
             'u_fn': u_fn,
             'u_params': u_params,
             'masses': masses,
             'xs': xs,
             'vs': vs,
             'neighbor_fn' : neighbor_fn,
             'neighbor_list' : vacuum_neighbor_list
             }

    return _dict


def get_diatom_equilibrium_cache(seed = jax.random.PRNGKey(455),
                                 _dict = get_diatom_parameters_dict(),
                                 num_samples=50,
                                 steps_per_sample=5000):

    displacement_fn, shift_fn = _dict['displacement_fn'], _dict['shift_fn']
    metric = jax.jit(jax_md.space.canonicalize_displacement_or_metric(displacement_fn))
    masses = _dict['masses']
    xs, vs = _dict['xs'], _dict['vs']
    repeat_xs = jnp.repeat(xs[jnp.newaxis,...], repeats=num_samples, axis=0)

    u_params, u_fn = _dict['u_params'], _dict['u_fn']

    # get integrator
    int_generator = BAOABIntegratorGenerator(canonical_u_fn = u_fn,
                                             neighbor_list = _dict['neighbor_list'],
                                             dt = DEFAULT_TIMESTEP,
                                             masses = masses,
                                             kT = kT,
                                             shift_fn = shift_fn,
                                             collision_rate=1.)

    integrator = int_generator.integrator(steps_per_sample)

    jax_int = partial(integrator, neighbor_list=_dict['neighbor_list'], sequence = jnp.arange(steps_per_sample))
    out_dict = jax.vmap(jax_int, in_axes=(0,None,0))(repeat_xs, u_params, jax.random.split(seed, num=num_samples))
    dxs = jax.vmap(metric, in_axes=(0,0))(out_dict['xs'][:,0], out_dict['xs'][:,1])
    kes = jax.vmap(kinetic_energy, in_axes=(0,None))(out_dict['vs'], masses)
    displacements = dxs
    posits = out_dict['xs']
    return displacements, kes, posits

def test_folded_equilibrium_integrator():
    """
    run an equilibrium simulation on a diatomic species and assert kinetic energy
    and displacement statistics are close to analytical value.
    """
    diatom = Diatom()
    displacement_std = jnp.sqrt((1. / diatom.K).value_in_unit_system(unit.md_unit_system))
    displacement_mean = Array((diatom.r0).value_in_unit_system(unit.md_unit_system))
    mean_ke = 3. * kT

    displacements, kes, _ = get_diatom_equilibrium_cache()

    measured_mean_displacement = jnp.mean(displacements)
    displacement_bounds = displacement_std * 3

    # displacement assertions
    assert measured_mean_displacement > displacement_mean - displacement_bounds
    assert measured_mean_displacement < displacement_mean + displacement_bounds

    # kinetic energy assertions
    ke_bound = 1.
    measured_mean_ke = jnp.mean(kes)
    assert measured_mean_ke > mean_ke - ke_bound
    assert measured_mean_ke < mean_ke + ke_bound
