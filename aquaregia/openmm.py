from typing import Sequence, Callable, Dict, Tuple, Optional, NamedTuple, Any
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import lax, ops, vmap, jit, grad, random
from jax.scipy.special import erfc
from aquaregia.tincture import polynomial_switching_fn, get_periodic_distance_calculator, get_mask
import tqdm


from jax.config import config
config.update("jax_enable_x64", True)

import jraph
from jraph._src.models import ArrayTree
from jax_md import space
from jax_md.partition import NeighborList, NeighborFn
from jax_md.util import high_precision_sum, maybe_downcast

# Constant
from openmmtools.constants import ONE_4PI_EPS0
from simtk import openmm, unit


# typing
from aquaregia.utils import EnergyFn, Array
Quantity = unit.quantity.Quantity

def make_HarmonicBondForce(openmm_bond_force : openmm.HarmonicBondForce,
                           displacement_or_metric : space.DisplacementOrMetricFn,
                           **kwargs
                          ) -> Tuple[ArrayTree, EnergyFn]:
    """
    from an openmm.HarmonicBondForce, write a function that will compute a harmonic bond energy
    """
    #vmap the displacement or metric fn
    metric = vmap(space.canonicalize_displacement_or_metric(displacement_or_metric), in_axes=(0,0))

    # extract bond parameters
    num_bonds = openmm_bond_force.getNumBonds()

    bond_terms = {'p1': [], 'p2' :[], 'length': [], 'k': []}
    #query the bond parameters
    for idx in range(num_bonds):
        p1, p2, length, k = openmm_bond_force.getBondParameters(idx)
        bond_terms['p1'].append(p1)
        bond_terms['p2'].append(p2)
        bond_terms['length'].append(length.value_in_unit_system(unit.md_unit_system))
        bond_terms['k'].append(k.value_in_unit_system(unit.md_unit_system))

    out_bond_terms = {}
    for key, val in bond_terms.items():
        out_bond_terms[key] = jnp.asarray(val)

    def hookean_spring_energy_fn(R, parameter_dict, metric_fn, **kwargs):
        Ra, Rb = R[parameter_dict['p1'], :], R[parameter_dict['p2'], :]
        lengths, ks = maybe_downcast(parameter_dict['length']), maybe_downcast(parameter_dict['k'])
        drs = metric_fn(Ra, Rb)
        return high_precision_sum(0.5 * ks * jnp.power(drs - lengths, 2))

    return out_bond_terms, partial(hookean_spring_energy_fn, metric_fn = metric)


def make_HarmonicAngleForce(openmm_angle_force : openmm.HarmonicAngleForce,
                            displacement_fn: space.DisplacementFn,
                            **kwargs
                           ) -> Tuple[ArrayTree, EnergyFn]:
    """"""

    vdisplacement_fn = vmap(displacement_fn, in_axes = (0,0))

    # extract angle parameters
    num_angles = openmm_angle_force.getNumAngles()

    angle_terms = {'p1': [], 'p2' :[], 'p3': [], 'theta0': [], 'k': []}

    #query the angle parameters
    for idx in range(num_angles):
        p1, p2, p3, theta0, k = openmm_angle_force.getAngleParameters(idx)
        angle_terms['p1'].append(p1)
        angle_terms['p2'].append(p2)
        angle_terms['p3'].append(p3)
        angle_terms['theta0'].append(theta0.value_in_unit_system(unit.md_unit_system))
        angle_terms['k'].append(k.value_in_unit_system(unit.md_unit_system))

    out_angle_terms = {}
    for key, val in angle_terms.items(): out_angle_terms[key] = jnp.asarray(val)

    def harmonic_angle_energy(R, parameter_dict, vmapped_displacement_fn, **kwargs):
        r1s, r2s, r3s = R[parameter_dict['p1']], R[parameter_dict['p2']], R[parameter_dict['p3']]
        theta0s, ks = maybe_downcast(parameter_dict['theta0']), maybe_downcast(parameter_dict['k'])
        r21s = vmapped_displacement_fn(r1s, r2s)
        r23s = vmapped_displacement_fn(r3s, r2s)

        tops = high_precision_sum(jnp.multiply(r21s, r23s), axis=-1)
        bots = jnp.linalg.norm(r21s, axis=-1) * jnp.linalg.norm(r23s, axis=-1)

        tb = tops / bots

        angles = jnp.arccos(tb)
        return high_precision_sum(0.5 * ks * jnp.power(angles - theta0s, 2))

    return out_angle_terms, partial(harmonic_angle_energy, vmapped_displacement_fn = vdisplacement_fn)

def make_PeriodicTorsionForce(openmm_torsion_force : openmm.PeriodicTorsionForce,
                              displacement_fn : space.DisplacementFn,
                              **kwargs
                             ) -> Tuple[ArrayTree, EnergyFn]:
    vdisplacement_fn = vmap(displacement_fn, in_axes = (0,0))

    # extract angle parameters
    num_torsions = openmm_torsion_force.getNumTorsions()
    torsion_terms = {'p1': [], 'p2' :[], 'p3': [], 'p4': [], 'periodicity': [], 'phase': [], 'k': []}

    #query the angle parameters
    for idx in range(num_torsions):
        p1, p2, p3, p4, per, phase, k = openmm_torsion_force.getTorsionParameters(idx)
        torsion_terms['p1'].append(p1)
        torsion_terms['p2'].append(p2)
        torsion_terms['p3'].append(p3)
        torsion_terms['p4'].append(p4)
        torsion_terms['periodicity'].append(per)
        torsion_terms['phase'].append(phase.value_in_unit_system(unit.md_unit_system))
        torsion_terms['k'].append(k.value_in_unit_system(unit.md_unit_system))

    out_torsion_terms = {}
    for key, val in torsion_terms.items(): out_torsion_terms[key] = jnp.asarray(val)

    def periodic_torsion_energy(R, parameter_dict, vmapped_displacement_fn, **kwargs):
        ci, cj, ck, cl = R[parameter_dict['p1'],:], R[parameter_dict['p2'],:], R[parameter_dict['p3'],:], R[parameter_dict['p4'], :]

        periodicities, phases, ks = parameter_dict['periodicity'], parameter_dict['phase'], parameter_dict['k']
        rij = vmapped_displacement_fn(cj, ci)
        rkj = vmapped_displacement_fn(cj, ck)
        rkl = vmapped_displacement_fn(cl, ck)

        n1 = jnp.cross(rij, rkj)
        n2 = jnp.cross(rkj, rkl)

        y = jnp.sum(jnp.multiply(jnp.cross(n1, n2), rkj / jnp.linalg.norm(rkj, axis=-1, keepdims=True)), axis=-1)
        x = jnp.sum(jnp.multiply(n1, n2), axis=-1)

        torsion_angles = jnp.arctan2(y, x)

        return high_precision_sum(ks * (1. + jnp.cos(periodicities * torsion_angles - phases)))

    return out_torsion_terms, partial(periodic_torsion_energy, vmapped_displacement_fn = vdisplacement_fn)

# potentials

def lifted_vacuum_electrostatics(dr : float,
                                 chargeProd : float) -> float:
    return ONE_4PI_EPS0 * chargeProd / (dr)

def lifted_vacuum_lj(dr : float,
                             sigma : float,
                             epsilon : float,
                             w : Optional[float] = 0.):
    r_eff = dr + w
    red_sigma = (sigma / r_eff) ** 6
    return 4. * epsilon * ( red_sigma**2 - red_sigma )

def lifted_rf_electrostatics(dr, chargeProd, r_cutoff, r_switch, delta):
    # NOTE : this is wrong, strictly speaking
    # WARNING : make this match...
    alpha = jnp.sqrt(-jnp.log(2.*delta)) / r_cutoff #compute alpha
    val = ONE_4PI_EPS0 * chargeProd  * (erfc(alpha * dr)/(dr) - erfc(alpha * r_cutoff)/r_cutoff)
    multiplier = lax.cond(dr > r_cutoff, lambda _x : 0., lambda _x : 1., None)
    return val * multiplier

def lifted_switched_lj(dr : float,
                       sigma : float,
                       epsilon : float,
                       w : Optional[float] = 0.,
                       r_cutoff : Optional[float] = 1.,
                       r_switch : Optional[float] = 0.85):
    vacuum_lj_energy = lifted_vacuum_lj(dr, sigma, epsilon, w)
    polynomial_scalar = polynomial_switching_fn(r = dr, r_cutoff = r_cutoff, r_switch = r_switch)
    return vacuum_lj_energy * polynomial_scalar

def get_box_vectors_from_vec3s(vec3s : Tuple[Quantity, Quantity, Quantity]) -> Array:
    """
    query a tuple object of vec3s to get a box array (for pbc-enabled nbfs)

    Example:
    >>> a,b,c = system.getDefaultPeriodicBoxVectors()
    >>>bvs = get_box_vectors_from_vec3s((a,b,c))
    """
    rank = []
    enumers = [0,1,2]
    for idx, i in enumerate(vec3s):
        rank.append(i[idx].value_in_unit_system(unit.md_unit_system))
        lessers = [q for q in enumers if q != idx]
        for j in lessers:
            assert jnp.isclose(i[j].value_in_unit_system(unit.md_unit_system), 0.), f"vec3({i,j} is nonzero. vec3 is not a cube)"
    return Array(rank)

# nonbonded querying utilities
def handle_exceptions(openmm_nonbonded_force : openmm.NonbondedForce) -> Tuple[NamedTuple, Array, dict]:
    """
    query openmm nonbonded force exceptions and create an immutable neighbor list corresponding to the exceptions, an overfill mask boolean,
    and a dictionary of exceptions

    return a exception neighbor list (immutable)
    """
    num_particles = openmm_nonbonded_force.getNumParticles()
    num_exceptions = openmm_nonbonded_force.getNumExceptions()
    nonbonded_exception_parameter_template = [[] for i in range(num_particles)] #empty

    template_nonbonded_exception_parameters = {'chargeProd': [[] for i in range(num_particles)],
                                  'sigma': [[] for i in range(num_particles)],
                                  'epsilon': [[] for i in range(num_particles)],
                                  'w' : [[] for i in range(num_particles)]
                                 }
    nonbonded_exception_parameters = {}

    # query the exceptions
    _trange = tqdm.trange(num_exceptions, desc = f"querying nonbonded exception particles...", leave=True)
    for idx in _trange:
        p1, p2, chargeProd, sigma, epsilon = openmm_nonbonded_force.getExceptionParameters(idx) #query

        # turn parameters to base units
        chargeProd = chargeProd.value_in_unit_system(unit.md_unit_system)
        sigma = sigma.value_in_unit_system(unit.md_unit_system)
        epsilon = epsilon.value_in_unit_system(unit.md_unit_system)
        w = 0.

        # order particle indices
        particle1 = p1 if p1 < p2 else p2 #order the parameters
        particle2 = p2 if p1 < p2 else p1

        nonbonded_exception_parameter_template[particle1].append(particle2)
        nonbonded_exception_parameter_template[particle2].append(particle1)

        for key, val in zip(template_nonbonded_exception_parameters.keys(), [chargeProd, sigma, epsilon, w]):
            template_nonbonded_exception_parameters[key][particle1].append(val)
            template_nonbonded_exception_parameters[key][particle2].append(val)


    # rework the nonbonded_exception_parameter_template
    arg_max = jnp.argmax(Array([len(q) for q in nonbonded_exception_parameter_template])) #pull the list index with the largest number of exceptions
    max_exceptions = len(nonbonded_exception_parameter_template[arg_max]) #pull out the max number of exceptions
    padded_nonbonded_exception_parameter_template = Array([q + [num_particles]*(max_exceptions - len(q)) for q in nonbonded_exception_parameter_template])
    exception_neighbor_list = NeighborList(idx = padded_nonbonded_exception_parameter_template,
                                    reference_position=None,
                                    did_buffer_overflow=False,
                                    max_occupancy=0.,
                                    cell_list=None,
                                    format=None,
                                    update_fn=None)

    for key in template_nonbonded_exception_parameters.keys():
        nonbonded_exception_parameters[key] = Array([q + [0.]*(max_exceptions - len(q)) for q in template_nonbonded_exception_parameters[key]])

    nonbonded_exception_parameters['exception_template'] = padded_nonbonded_exception_parameter_template

    exception_overfill_mask = exception_neighbor_list.idx != num_particles

    return exception_neighbor_list, exception_overfill_mask, nonbonded_exception_parameters

def handle_standards(openmm_nonbonded_force : openmm.NonbondedForce) -> dict:
    """
    return a dictionary of standard nonbonded force parameters
    """
    num_particles = openmm_nonbonded_force.getNumParticles()
    nonbonded_parameters = {'particle_index': [], 'charge': [], 'sigma': [], 'epsilon': [], 'w': []}

    #query the particle terms
    _trange = tqdm.trange(num_particles, desc = f"querying nonbonded particles", leave=True)
    for idx in _trange:
        charge, sigma, epsilon = openmm_nonbonded_force.getParticleParameters(idx)
        nonbonded_parameters['particle_index'].append(idx)
        nonbonded_parameters['charge'].append(charge.value_in_unit_system(unit.md_unit_system))
        nonbonded_parameters['sigma'].append(sigma.value_in_unit_system(unit.md_unit_system))
        nonbonded_parameters['epsilon'].append(epsilon.value_in_unit_system(unit.md_unit_system))
        nonbonded_parameters['w'].append(0.)

    # make these objects jnp.
    out_nonbonded_parameters = {}
    for key, val in nonbonded_parameters.items():
        out_nonbonded_parameters[key] = Array(val)

    return out_nonbonded_parameters

def make_NonbondedForce(openmm_nonbonded_force : openmm.NonbondedForce,
                        displacement_or_metric : space.DisplacementOrMetricFn,
                        vacuum_r_cutoff : Optional[float] = 100.,
                        r_switch_multiplier : Optional[float] = 0.9,
                        **kwargs)-> Tuple[ArrayTree, EnergyFn]:
    """
    transcribe an openmm.NonbondedForce

    WARNING : the direct space electrostatics PME energies and forces are discrepant with OpenMM right now; however, sterics work.
    """

    # we'll support two kinds of nonbonded methods: nocutoff and cutoff periodic
    nonbonded_method = openmm_nonbonded_force.getNonbondedMethod()
    periodic=False if nonbonded_method == 0 else True

    # extract nonbonded parameters
    num_particles = openmm_nonbonded_force.getNumParticles()
    num_exceptions = openmm_nonbonded_force.getNumExceptions()

    # we need to assert the absence of some attributes...
    if openmm_nonbonded_force.getNumExceptionParameterOffsets() != 0:
        raise ValueError(f"we do not currently support `ExceptionParameterOffsets`")

    if openmm_nonbonded_force.getNumParticleParameterOffsets() != 0:
        raise ValueError(f"we do not currently support `ParticleParameterOffsets`")

    #vmap the displacement or metric fn
    metric = space.canonicalize_displacement_or_metric(displacement_or_metric)

    if periodic:
        r_cutoff = openmm_nonbonded_force.getCutoffDistance().value_in_unit_system(unit.md_unit_system)
        r_switch = openmm_nonbonded_force.getSwitchingDistance().value_in_unit_system(unit.md_unit_system)
        delta = openmm_nonbonded_force.getEwaldErrorTolerance()
        electrostatic_fn = partial(lifted_rf_electrostatics, r_cutoff = r_cutoff, r_switch = r_switch, delta = delta)
        steric_fn = partial(lifted_switched_lj, r_cutoff = r_cutoff, r_switch = r_switch)

        def elec_offset(charges):
            alpha = jnp.sqrt(-jnp.log(2.*delta)) / r_cutoff
            return -1. * jnp.sum(charges**2) * ONE_4PI_EPS0 * alpha / jnp.sqrt(jnp.pi)

    else: #vacuum treatment uses a generous vacuum cutoff and no r_switch
        r_cutoff = vacuum_r_cutoff
        r_switch = None
        electrostatic_fn = lifted_vacuum_electrostatics
        steric_fn = lifted_vacuum_lj
        def elec_offset(charges): return 0.

    vmetric = get_periodic_distance_calculator(metric, r_cutoff)

    # the exception sterics and electrostatics are always vacuum
    exception_electrostatic_fn = lifted_vacuum_electrostatics
    exception_steric_fn = lifted_vacuum_lj

    def find_indices(arr, nbr_list_idx): return jnp.where(jnp.isin(nbr_list_idx, arr), True, False) #returnable logical `True` if in exceptions

    #vmap chargeprod, sigma, epsilon computations
    vchargeProd = vmap(vmap(lambda x, y : x*y, in_axes=(None, 0)))
    vsigma = vmap(vmap(lambda x, y : 0.5 * (x + y), in_axes = (None, 0)))
    vepsilon = vmap(vmap(lambda x, y : jnp.sqrt(x * y), in_axes = (None, 0)))
    v_wlift = vsigma

    # make a parameter_dict; can we pass this to a new function?
    standard_nonbonded_parameters = handle_standards(openmm_nonbonded_force)
    exception_neighbor_list, exception_overfill_mask, exception_nonbonded_parameters = handle_exceptions(openmm_nonbonded_force)

    nonbonded_parameters = {'standard': standard_nonbonded_parameters,
                            'exceptions': exception_nonbonded_parameters}

    def nonbonded_energy_fn(R,
                            neighbor_list,
                            parameter_dict):

        #separate out standard and exception parameters
        standard_parameters = parameter_dict['standard']
        exception_parameters = parameter_dict['exceptions']

        # compute standard
        charges, sigmas, epsilons, ws = standard_parameters['charge'], standard_parameters['sigma'], standard_parameters['epsilon'], standard_parameters['w']
        drs = vmetric(R, neighbor_list) + v_wlift(ws, ws[neighbor_list.idx]) #compute lifted drs
        electrostatic_energies = jnp.vectorize(electrostatic_fn)(drs,
                                                                 vchargeProd(charges, charges[neighbor_list.idx]))
        steric_energies = jnp.vectorize(steric_fn)(drs,
                                                   vsigma(sigmas, sigmas[neighbor_list.idx]),
                                                   vepsilon(epsilons, epsilons[neighbor_list.idx]))

        energies = electrostatic_energies + steric_energies #the diagonals should be nan...they are to be replaced...
        overfill_mask = get_mask(neighbor_list) # True on the particle indices that aren't overfilled
        exception_mask = vmap(find_indices, in_axes=(0,0))(exception_neighbor_list.idx, neighbor_list.idx) # True if in exception
        not_exception_mask = jnp.invert(exception_mask) # False if in exception
        overfill_mask_and_not_exceptions = jnp.logical_and(overfill_mask, not_exception_mask)
        energies = jnp.where(overfill_mask_and_not_exceptions, energies, 0.)

        #compute exceptions
        exception_drs = vmetric(R, exception_neighbor_list) + exception_parameters['w']
        exception_elecrostatics = jnp.vectorize(exception_electrostatic_fn)(exception_drs,
                                                                            exception_parameters['chargeProd'])
        exception_sterics = jnp.vectorize(exception_steric_fn)(exception_drs,
                                                     exception_parameters['sigma'],
                                                     exception_parameters['epsilon'])

        exception_energies = exception_elecrostatics + exception_sterics
        exception_energies = jnp.where(exception_overfill_mask, exception_energies, 0.)

        total_energy = 0.5 * ( high_precision_sum(energies) + high_precision_sum(exception_energies) )
        return total_energy #+ elec_offset(charges)

    return nonbonded_parameters, nonbonded_energy_fn

# Constant
KNOWN_FORCE_fns = {'HarmonicBondForce' : make_HarmonicBondForce,
                   'HarmonicAngleForce' : make_HarmonicAngleForce,
                   'PeriodicTorsionForce' : make_PeriodicTorsionForce,
                   'NonbondedForce' : make_NonbondedForce}

def make_canonical_energy_fn(system : openmm.System,
                             displacement_fn: space.DisplacementFn,
                             kwargs_dict : Optional[Dict] = {},
                            )-> Tuple[ArrayTree, EnergyFn]:
    """write a canonical energy function containing HarmonicBonds, HarmonicAngles, PeriodicTorsions and a NonbondedForce"""
    forces = system.getForces()
    force_dict = {force.__class__.__name__ : force for force in forces}

    out_params = {}
    out_fns = {}

    for force_name, force in force_dict.items():
        try:
            generator_fn = KNOWN_FORCE_fns[force_name]
            params, energy_fn = generator_fn(force, displacement_fn, **kwargs_dict)
            out_params[force_name] = params
            out_fns[force_name] = energy_fn

        except Exception as e:
            raise RuntimeError(f"encountered error: {e}")


    def out_energy_fn(xs : Array, neighbor_list : NeighborList, parameters : ArrayTree):
        running_sum = 0.
        for key in parameters.keys():
            running_sum = running_sum + out_fns[key](R = xs,
                                                     neighbor_list = neighbor_list,
                                                     parameter_dict = parameters[key])

        return running_sum

    return out_params, out_energy_fn

# Utility
def make_scale_system(system : openmm.System,
                      target_ps : Sequence[int],
                      remove_angles : Optional[bool] = True,
                      remove_torsions : Optional[bool] = True,
                      nb_scale_factor : Optional[float] = 1e-3,
                      scale_electrostatics : Optional[bool] = True,
                      scale_sterics : Optional[bool] = True,
                      scale_exceptions : Optional[bool] = True) -> openmm.System:
    """
    make a deecopy of a system and scale the torsion, nb, and nb_exception parameters of all terms containing `target_ps`.
    optionally scale angles, as well
    """
    from copy import deepcopy
    out_system = deepcopy(system)

    torsion_f = out_system.getForces()[-2]
    nbf = out_system.getForces()[-1]

    target_ps_set = set(target_ps)

    # assert there are no constraints.
    assert out_system.getNumConstraints() == 0, f"we need not constraints"

    if remove_angles:
        angle_force = out_system.getForces()[-3]
        #angles
        for angle_idx in range(angle_force.getNumAngles()):
            p1, p2, p3, angle, k = angle_force.getAngleParameters(angle_idx)
            if target_ps_set.issubset({p1, p2, p3}):
                angle_force.setAngleParameters(angle_idx, p1, p2, p3, angle, k * nb_scale_factor)


    # torsions
    if remove_torsions:
        for torsion_idx in range(torsion_f.getNumTorsions()):
            torsion_parameters = torsion_f.getTorsionParameters(torsion_idx)
            torsion_param_set = set(torsion_parameters[:4])
            per, phase, k = torsion_parameters[4:]
            if torsion_param_set.issubset(target_ps_set):
                torsion_f.setTorsionParameters(torsion_idx, *torsion_parameters[:4], per, phase, k * 0.)

    # nonbondeds
    for p_idx in range(nbf.getNumParticles()):
        charge, sigma, eps = nbf.getParticleParameters(p_idx)
        scaled_charge = charge * nb_scale_factor if scale_electrostatics else charge
        scaled_epsilon = eps * nb_scale_factor if scale_sterics else eps
        if p_idx in target_ps:
            nbf.setParticleParameters(p_idx, scaled_charge, sigma, scaled_epsilon)

    #nonbonded exceptions
    if scale_exceptions:
        for nonbonded_exception_idx in range(nbf.getNumExceptions()):
            p1, p2, chargeProd, sigma, eps = nbf.getExceptionParameters(nonbonded_exception_idx)
            scaled_chargeProd = chargeProd * nb_scale_factor if scale_electrostatics else chargeProd
            scaled_epsilon = eps * nb_scale_factor if scale_sterics else eps
            if {p1, p2}.issubset(target_ps_set):
                nbf.setExceptionParameters(nonbonded_exception_idx, p1, p2, scaled_chargeProd, sigma, scaled_epsilon)

    return out_system
