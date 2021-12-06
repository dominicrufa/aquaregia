"""test aquaregia.openmm"""
from aquaregia.openmm import *
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
    out_params, energy_fn = make_canonical_energy_fn(testsystem,
                             disp,
                             kwargs_dict = {})

    jenergy_fn = jit(energy_fn)

    #grab a vacuum neighbor list
    from aquaregia.utils import get_vacuum_neighbor_list
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
        if key not in KNOWN_FORCE_fns.keys():
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

def test_explicit_energy_canonicalization():
    """do a jax energy assertion test for AlanineDipeptideExplicit, HostGuestExplicit"""
    # we have discrepant energies in electrostatics energy
    # also be sure to turn off dispersion correction and _only_ take the direct space PME energy.
    pass
