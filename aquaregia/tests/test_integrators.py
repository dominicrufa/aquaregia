"""
test the `aquaregia.integrators` library functionality
"""
from aquaregia.integrators import *

from openmmtools.testsystems import Diatom
from aquaregia.openmm import make_canonical_energy_fn
from aquaregia.utils import kinetic_energy
import tqdm

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
                                              displacement_fn=displacement_fn,
                                              fix_parameters=False)

    _dict = {'displacement_fn': displacement_fn,
             'shift_fn': shift_fn,
             'u_fn': u_fn,
             'u_params': u_params,
             'masses': masses,
             'xs': xs,
             'vs': vs}

    return _dict


def get_diatom_equilibrium_cache(seed = random.PRNGKey(455), _dict = get_diatom_parameters_dict(), num_samples=500, steps_per_sample=50):
    """
    run equilibrium integrator of diatom. returns equilibrium positions, velocities
    """
    displacement_fn, shift_fn = _dict['displacement_fn'], _dict['shift_fn']
    metric = jit(jax_md.space.canonicalize_displacement_or_metric(displacement_fn))
    masses = _dict['masses']
    xs, vs = _dict['xs'], _dict['vs']

    u_params, u_fn = _dict['u_params'], _dict['u_fn']

    #make the integrator
    integrator = get_folded_equilibrium_integrator(potential_energy_fn = u_fn,
                                      potential_energy_parameters = u_params,
                                      kT = kT,
                                      dt = DEFAULT_TIMESTEP,
                                      gamma = 1.,
                                      mass = masses,
                                      shift_fn = shift_fn,
                                      kinetic_energy_fn = kinetic_energy,
                                      get_shadow_work = False)
    jax_int = jit(integrator)
    jkinetic_energy = jit(kinetic_energy)

    displacements = []
    kes = []
    posits = []
    #seed = random.PRNGKey(455)
    for i in tqdm.trange(num_samples):
        seed, run_seed = random.split(seed)
        xs, vs, _ = jax_int(xs, vs, start_pe=0., seed=run_seed, sequence=jnp.arange(steps_per_sample))
        dx = metric(xs[0], xs[1])
        ke = jkinetic_energy(vs, masses)
        displacements.append(dx)
        posits.append(xs)
        kes.append(ke)

    return Array(displacements), Array(kes), Array(posits)

def test_folded_equilibrium_integrator():
    """
    run an equilibrium simulation on a diatomic species and assert kinetic energy
    and displacement statistics are close to analytical value
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


def test_forward_nonequilibrium_integrator():
    """
    do a simple test on a nonequilibrium integrator to show that we improve logZ estimate over IW when performing
    AIS. Also show that the bias decreases.

    In this test case, we are increasing the equilibrium bond length of a diatom from 0.155 to 0.175 nm.

    TODO : can we write a more robust test that will ensure the free energy is absolutely unchanged?
    """
    from aquaregia.utils import logZ_from_works
    _dict = get_diatom_parameters_dict()
    _, _, xs_eq = get_diatom_equilibrium_cache(random.PRNGKey(27491), _dict)

    num_iters = 100
    protocol = jnp.linspace(0.155, 0.175, num_iters)[..., jnp.newaxis]
    #print(protocol[0], protocol[-1])

    def mod_potential_params_fn(_iter):
        _dict['u_params']['HarmonicBondForce']['length'] = protocol[_iter]
        return _dict['u_params']

    def mod_kT_fn(_iter):
        return kT


    neq_int = get_nonequilibrium_integrator(potential_energy_fn=_dict['u_fn'],
                                            potential_energy_parameters=_dict['u_params'],
                                            kT = kT,
                                            dt = DEFAULT_TIMESTEP,
                                            gamma=1.,
                                            mass=_dict['masses'],
                                            mod_potential_params_fn=mod_potential_params_fn,
                                            mod_kT_fn=mod_kT_fn,
                                            shift_fn = _dict['shift_fn'],
                                            kinetic_energy_fn=kinetic_energy,
                                            get_shadow_work=False
                                           )


    partial_neq_int = partial(neq_int, sequence=jnp.arange(num_iters, dtype=jnp.int64)[1:])
    in_seed = random.PRNGKey(270)
    vthermalize = vmap(partial(thermalize, kT=kT, masses = _dict['masses'], dimension=3))

    vs_eq = vthermalize(random.split(random.PRNGKey(236), num=500))

    vpartial_neq_int = vmap(partial_neq_int, in_axes=(0,0,0,0))

    start_pes = vmap(_dict['u_fn'], in_axes=(0,None))(xs_eq, mod_potential_params_fn(0))
    end_pes = vmap(_dict['u_fn'], in_axes = (0,None))(xs_eq, mod_potential_params_fn(num_iters-1))
    imp_outs = (end_pes - start_pes) / kT

    run_seed = random.PRNGKey(380)
    out_xs, out_vs, out_works = vpartial_neq_int(xs_eq, vs_eq, start_pes, random.split(run_seed, num=500))
    print(out_xs.shape, out_vs.shape, out_works.shape)
    cumulative_works = jnp.cumsum(out_works, axis=1)[:,-1] / kT

    bound = 1.
    logZ_estimate = -logZ_from_works(cumulative_works)
    logZ_IW_estimate = -logZ_from_works(imp_outs)

    assert logZ_estimate > -bound
    assert logZ_estimate < bound

    assert logZ_estimate < logZ_IW_estimate
