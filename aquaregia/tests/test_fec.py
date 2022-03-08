import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from jax.config import config
config.update("jax_enable_x64", True)
from aquaregia.utils import Array, ArrayTree

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
                                       lambdas = jnp.linspace(1., 0., num_windows))

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
