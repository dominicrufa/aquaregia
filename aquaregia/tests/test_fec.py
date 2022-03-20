import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from jax.config import config
config.update("jax_enable_x64", True)
from aquaregia.utils import Array, ArrayTree
FREE_ENERGY_TOLERANCE = 5e-2

def test_AbsoluteFECProtocol():
    """
    test the AbsoluteFECProtocol; specifically, make sure that it integrates into energy vmapping capabilities.
    """
    from aquaregia.tests.test_openmm import get_cleaned_vacuum_system
    from openmmtools.testsystems import HostGuestVacuum
    from jax_md.space import free
    from aquaregia.utils import Array, get_vacuum_neighbor_list
    from aquaregia.openmm import make_canonical_energy_fn
    from simtk import unit
    from aquaregia.fec import AbsoluteFECProtocol

    # clean
    testsys_instance = HostGuestVacuum(constraint=None)
    get_cleaned_vacuum_system(testsys_instance)

    # get lifted particles
    lifted_res = list(testsys_instance.topology.residues())[1]
    lifted_atom_indices = [atom.index for atom in lifted_res.atoms()]

    # create jax system
    vac_neighbor_list = get_vacuum_neighbor_list(testsys_instance.system.getNumParticles())
    params, energy_fn = make_canonical_energy_fn(system = testsys_instance.system, displacement_fn=free()[0], allow_constraints=True)

    # test energy
    test_jax_energy = energy_fn(testsys_instance.positions / unit.nanometers, vac_neighbor_list, params)


    # create the protocol
    num_windows=10
    protocol_cls = AbsoluteFECProtocol(coupled_parameter_dict = params,
                                       alchemical_particle_indices = lifted_atom_indices,
                                       w_lambdas = jnp.linspace(1., 0., num_windows))

    # get the parameters and axis mapper
    vaxes = protocol_cls.vmap_axes_dict
    parameters = protocol_cls.parameter_dict

    # now check vmap dimensions
    for key, val in vaxes.items():
        if key != 'NonbondedForce':
            assert val is None
        else: #this is NonbondedForce
            assert parameters[key]['standard']['charge'].shape[0] == num_windows
            assert parameters[key]['exceptions']['chargeProd'].shape[0] == num_windows
            assert parameters[key]['standard']['w'].shape[0] == num_windows
            assert jnp.allclose(parameters[key]['standard']['w'][0],params['NonbondedForce']['standard']['w'])

    # call jax energies
    vac_neighbor_list = get_vacuum_neighbor_list(testsys_instance.system.getNumParticles())
    _venergies = jax.vmap(energy_fn, in_axes=(None, None, vaxes))(testsys_instance.positions/unit.nanometers, vac_neighbor_list, parameters)
    assert _venergies.shape[0] == num_windows
    assert jnp.isclose(_venergies[0], energy_fn(testsys_instance.positions/unit.nanometers, vac_neighbor_list, params))

def diatom_free_energy(r0, k0, r1, k1, kT):
    from scipy import integrate
    def u(r, r_, k, kT):
        return r**2 * np.exp(-0.5 * k * (r-r_)**2/kT)

    lower, upper = 0., 1.

    Z0 = integrate.quad(u, lower, upper, (r0, k0, kT))[0]
    Z1 = integrate.quad(u, lower, upper, (r1, k1, kT))[0]
    return -np.log(Z1/Z0)

def test_NonCanonicalUniformLogPReplicaExchangeSampler():
    """test the `NonCanonicalUniformLogPReplicaExchangeSampler` on the `Diatom` testsystem
     where the spring constant is changed over the protocol"""
    from aquaregia.fec import NonCanonicalUniformLogPReplicaExchangeSampler
    from aquaregia.tests.test_integrators import get_diatom_parameters_dict, DEFAULT_TEMPERATURE, DEFAULT_TIMESTEP, kT
    from aquaregia.integrators import BAOABIntegratorGenerator

    # create the diatom testsystem
    diatom_params = get_diatom_parameters_dict()

    # make a propagator
    int_generator = BAOABIntegratorGenerator(canonical_u_fn=diatom_params['u_fn'],
                                         neighbor_list=diatom_params['neighbor_list'],
                                         dt = DEFAULT_TIMESTEP,
                                         masses = diatom_params['masses'],
                                         kT = kT,
                                         shift_fn=diatom_params['shift_fn'],
                                         collision_rate=1.,
                                         neighbor_list_update_fn=diatom_params['neighbor_fn'])

    # make two propagators. the first one exists to run an equilibration before repex
    prelim_propagator = int_generator.integrator(num_steps=5000, remove_neighbor_list=True)
    repex_propagator = int_generator.integrator(num_steps=100, remove_neighbor_list=True)

    num_windows=100 # define the number of windows

    # define the spring constant protocol
    u_params = diatom_params['u_params']
    u_params['HarmonicBondForce']['k'] = jnp.linspace(.5, 1., num_windows) * u_params['HarmonicBondForce']['k']
    u_params_axes = {'HarmonicBondForce': {'p1': None, 'p2': None, 'length': None, 'k' : 0}}

    # create the repex sampler
    repex_sampler = NonCanonicalUniformLogPReplicaExchangeSampler(num_states = num_windows,
             canonical_u_fn = diatom_params['u_fn'],
             kT = kT,
             mappable_u_params = u_params, # vmappable u_params
             mappable_u_params_axes = u_params_axes, # axes of vmappable u_params
             propagator = repex_propagator, # propagator class
             MCMC_save_interval = 1,
             neighbor_list_template = diatom_params['neighbor_list']
             )

    # batch xs
    batch_xs = jnp.repeat(diatom_params['xs'][jnp.newaxis, ...], repeats=num_windows, axis=0)

    # equilibrate
    out_equil_dict = jax.vmap(prelim_propagator, in_axes=(0,u_params_axes,0))(batch_xs, u_params, jax.random.split(jax.random.PRNGKey(25), num=num_windows))
    out_batch_xs = out_equil_dict['xs']

    # execute the repex sampler
    seed = jax.random.PRNGKey(231)
    out_dict = repex_sampler.execute(out_batch_xs, seed, num_iterations=50, num_swaps_per_mix_iteration=10000)

    # analyze the repex sampler
    mbar, total_accept_rate, acc_mat = repex_sampler.analyze()

    # get free energy differences
    free_energy_mat, dfree_energy_mat = mbar.getFreeEnergyDifferences()
    calc_free_energy = free_energy_mat[0,-1]

    # numerically compute free energy on 1d grid
    numerical_free_energy = diatom_free_energy(u_params['HarmonicBondForce']['length'],
                       u_params['HarmonicBondForce']['k'][0],
                       u_params['HarmonicBondForce']['length'],
                       u_params['HarmonicBondForce']['k'][-1],
                       repex_sampler._kT)

    assert abs(numerical_free_energy - calc_free_energy) <= FREE_ENERGY_TOLERANCE, f""" numerical free energy is {numerical_free_energy} while calculated free energy is {calc_free_energy}"""

def test_AlchemicalHostGuestVacuumEndstates():
    """
    test that openmmtools.testsystems.HostGuestVacuum recovers the same energies we would expect with out absolute free energy protocol
    at the coupled and decoupled endstates; we annihilate electrostatics but not sterics
    """
    from openmmtools.testsystems import HostGuestVacuum
    from openmmtools import alchemy
    from simtk import openmm, unit
    from openmmtools.integrators import DummyIntegrator
    from aquaregia.openmm import make_canonical_energy_fn
    from jax_md.space import free
    from aquaregia.utils import get_vacuum_neighbor_list
    from aquaregia.tests.test_openmm import ENERGY_PERCENT_ERROR_TOLERANCE
    from aquaregia.fec import AbsoluteFECProtocol
    disp_fn, shift_fn = free()

    factory = alchemy.AbsoluteAlchemicalFactory(consistent_exceptions=False) # this is typical?
    hgv = HostGuestVacuum(constraints=None)
    alchemical_atoms = range(126, 156) # this was pre-queried
    alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=alchemical_atoms, annihilate_sterics=False, annihilate_electrostatics=True)
    alchemical_system = factory.create_alchemical_system(reference_system=hgv.system, alchemical_regions=alchemical_region)

    integrator = DummyIntegrator()
    context = openmm.Context(alchemical_system, integrator)
    context.setPositions(hgv.positions)

    # coupled energy
    coupled_omm_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit_system(unit.md_unit_system)

    # decoupled energy
    swig_parameters = context.getParameters()
    context_params = {q : swig_parameters[q] for q in swig_parameters}
    context.setParameter('lambda_electrostatics', 0.)
    context.setParameter('lambda_sterics', 0.)

    # decoupled energy
    decoupled_omm_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit_system(unit.md_unit_system)
    del context
    del integrator

    # get protocol energies with jax
    vac_neighbor_list = get_vacuum_neighbor_list(hgv.system.getNumParticles())

    # remove CMMotion force
    hgv.system.removeForce(4)

    u_params, u_fn = make_canonical_energy_fn(hgv.system,
                                          displacement_fn=disp_fn,
                                          nonbonded_kwargs_dict={'vacuum_r_cutoff' : 2., 'vacuum_r_switch' : 1.75},
                                          )

    # make the protocol
    protocol = AbsoluteFECProtocol(coupled_parameter_dict=u_params,
                                   alchemical_particle_indices=list(alchemical_atoms),
                                   r_cutoff = 2.,
                                   )

    vaxes, protocol_parameters = protocol.vmap_axes_dict, protocol.parameter_dict
    jax_energies = jax.vmap(u_fn, in_axes=(None, None, vaxes))(hgv.positions/unit.nanometers, vac_neighbor_list, protocol_parameters)

    # energy asserttion
    omm_energies = np.array([coupled_omm_energy, decoupled_omm_energy])
    percent_differences = 100. * (omm_energies - np.array([jax_energies[0], jax_energies[-1]])) / omm_energies
    assert np.all(percent_differences < ENERGY_PERCENT_ERROR_TOLERANCE), f"percent differences are {percent_differences}"
