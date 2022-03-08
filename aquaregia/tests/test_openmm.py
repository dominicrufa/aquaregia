"""test aquaregia.openmm"""
from aquaregia.openmm import *
from pkg_resources import resource_filename
from aquaregia.utils import get_vacuum_neighbor_list
ENERGY_PERCENT_ERROR_TOLERANCE = 1e-2
FORCE_PERCENT_ERROR_TOLERANCE = 1e1

def vacuum_energy_canonicalization_test(testsystem : openmm.System,
                                        positions : Quantity,
                                        percent_error_e_threshold : Optional[float] = ENERGY_PERCENT_ERROR_TOLERANCE,
                                        percent_error_f_threshold : Optional[float] = FORCE_PERCENT_ERROR_TOLERANCE):
    """
    check that the canonicalization of energies from OpenMM to JaxMD is consistent.

    Arguments:
    percent_error_e_threshold : float
        a percent error inconsistency that we tolerate in energy
    percent_error_f_threshold : float
        a percent error inconsistency that we tolerate in force

    NOTE: there is a difference in the `percent_error_e_threshold` and the `percent_error_f_threshold`
    since OpenMM cannot compute forces deterministically for some reason; we really just want to make sure that
    energies are reported consistently and that forces are not crazy.
    """
    # make displacement and shift
    disp, shft = space.free()

    #make parameter dict and energy fn
    out_params, energy_fn = make_canonical_energy_fn(testsystem, disp, allow_constraints=True)

    jenergy_fn = jit(energy_fn)

    #grab a vacuum neighbor list
    vacuum_neighbor_list = get_vacuum_neighbor_list(num_particles = testsystem.getNumParticles())

    # get jax energy
    jax_e = jenergy_fn(positions.value_in_unit_system(unit.md_unit_system), vacuum_neighbor_list, out_params)
    jax_f = -1. * grad(energy_fn)(positions.value_in_unit_system(unit.md_unit_system), vacuum_neighbor_list, out_params)

    # get openmm energy
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(testsystem, integrator)
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    omm_e = state.getPotentialEnergy().value_in_unit_system(unit.md_unit_system)
    omm_f = state.getForces(asNumpy=True).value_in_unit_system(unit.md_unit_system)
    del context
    del integrator

    # check the energy:
    percent_error_e = jnp.abs((jax_e - omm_e) / omm_e) * 100.
    assert percent_error_e < percent_error_e_threshold, f"energy percent error of {percent_error_e} is greater than the threshold of {percent_error_e_threshold}"

    # check the forces:
    percent_error_f = jnp.abs((jax_f - omm_f) / omm_f) * 100.
    max_percent_error_f = jnp.max(percent_error_f)
    assert max_percent_error_f < percent_error_f_threshold, f"the max percent error f ({max_percent_error_f}) is greater than the threshold of {percent_error_f_threshold}"

def get_cleaned_vacuum_system(testsystem_instance):
    """remove the non-canonical forces"""
    system = testsystem_instance.system
    forces = system.getForces()
    force_dict = {force.__class__.__name__ : force for force in forces}
    known_list = []
    for key in force_dict.keys():
        if key not in KNOWN_FORCE_fns:
            known = False
        else:
            known = True
        known_list.append(known)

    for i in range(len(forces))[::-1]:
        known_bool = known_list[i]
        if not known_bool:
            system.removeForce(i)

def test_vacuum_energy_canonicalizations():
    """do a jax energy assertion test for AlanineDipeptideVacuum, TolueneVacuum, HostGuestVacuum"""
    from openmmtools import testsystems
    import tqdm

    instances = [testsystems.TolueneVacuum(), testsystems.AlanineDipeptideVacuum(), testsystems.HostGuestVacuum()]

    for i in tqdm.trange(len(instances)):
        testsys_instance = instances[i]
        get_cleaned_vacuum_system(testsys_instance)
        vacuum_energy_canonicalization_test(testsys_instance.system, testsys_instance.positions)

def test_decoupling():
    """
    1. create an energy function/params for host and guest separately.
    2. compute their energies
    3. assert that the energies are the same as the whole system with the first set lifted.
    4. translate/rotate the lifted particles and assert the total energy is unchanged
    """
    from aquaregia.utils import random_rotation_matrix
    from openmmtools.testsystems import HostGuestVacuum
    from jax_md.space import free
    displacement_fn, _ = free()

    hge = HostGuestVacuum(constraints=None)
    hge.system.removeForce(4) # remove the CMMotion Force

    res_indices = [] # get the res indices
    for res in list(hge.topology.residues())[:2]:
        res_indices.append([atom.index for atom in res.atoms()])
    res_indices.append(res_indices[0] + res_indices[1])

    vacuum_neighbor_list = get_vacuum_neighbor_list(sum([len(q) for q in res_indices]))
    fns = []
    energies = []
    for i in range(3):
        particles = res_indices[i]
        params, energy_fn = make_canonical_energy_fn(system = hge.system, displacement_fn=displacement_fn, particle_indices=particles)
        if i != 2:
            params['NonbondedForce']['standard']['w'] = 10. * jnp.ones(len(particles))
        else:
            params['NonbondedForce']['standard']['w'] = jnp.concatenate([jnp.zeros(len(res_indices[0])), 10. * jnp.ones(len(res_indices[1]))])
        partial_energy_fn = partial(energy_fn, parameters = params, neighbor_list = get_vacuum_neighbor_list(len(particles)))
        fns.append(partial_energy_fn)
        in_posits = hge.positions / unit.nanometer
        energies.append(partial_energy_fn(in_posits[particles,:]))

    # assert that the separate energy functions is equal to the decoupled energy function
    assert jnp.isclose(energies[-1], energies[0] + energies[1])

    # rotate the decoupled energy subset and assert that the energy is unchanged
    rot_matrix = random_rotation_matrix(np.random.RandomState())
    rotated_positions = in_posits[res_indices[0], :] @ rot_matrix
    in_posits[res_indices[0], :] = rotated_positions
    rot_energies = partial_energy_fn(in_posits)
    assert jnp.isclose(rot_energies, energies[-1])

def test_COM():
    """
    load a `complex.nc` file, remove all forces except the COM, and assert an equivalence in energy upon canonicalization of the system.
    """
    # deserialize system
    from simtk.openmm import XmlSerializer, Context
    from simtk import unit
    from openmmtools.integrators import DummyIntegrator
    from jax_md.space import free
    displacement_fn, _ = free()

    #system
    sys_xml_filename = resource_filename('aquaregia', 'data/complex.complex.decoupled_False.xml')
    with open(sys_xml_filename, 'r') as infile:
        xml_readable = infile.read()
    system = XmlSerializer.deserialize(xml_readable)

    #positions
    positions_filename = resource_filename('aquaregia', 'data/CB8G0.positions.npz')
    _dict = np.load(positions_filename)
    positions = _dict['arr_0'] * unit.nanometers

    # remove all forces except last
    for force_idx in range(system.getNumForces())[::-1]:
        if system.getForce(force_idx).__class__.__name__ != 'CustomCentroidBondForce':
            system.removeForce(force_idx)
    assert system.getNumForces() == 1 # check to make sure there is only 1 force

    #query atoms in COMForce
    _com_force = system.getForce(0)
    [(g0, g1), (K,)] = _com_force.getBondParameters(0)
    g0_atoms, g0_weights = _com_force.getGroupParameters(g0)
    g1_atoms, g1_weights = _com_force.getGroupParameters(g1)
    particle_indices = sorted(list(g0_atoms + g1_atoms))

    # get the omm energy
    _int = DummyIntegrator()
    context = Context(system, _int)
    context.setPositions(positions)
    omm_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit_system(unit.md_unit_system)
    del context
    del _int

    # get the jax energy
    params, energy_fn = make_canonical_energy_fn(system = system, displacement_fn=displacement_fn, particle_indices=particle_indices)
    vacuum_neighbor_list = get_vacuum_neighbor_list(len(particle_indices))
    jax_energy = energy_fn((positions/unit.nanometers)[particle_indices,:], parameters=params, neighbor_list = vacuum_neighbor_list)
    assert jnp.isclose(omm_energy, jax_energy), f"there is a discrepancy between the jax_energy {jax_energy} and the omm_energy {omm_energy}"


def test_explicit_energy_canonicalization():
    """do a jax energy assertion test for AlanineDipeptideExplicit, HostGuestExplicit"""
    # we have discrepant energies in electrostatics energy
    # also be sure to turn off dispersion correction and _only_ take the direct space PME energy.
    pass
