from typing import Sequence, Callable, Dict, Tuple, Optional, NamedTuple, Any
import jax
import jax.numpy as jnp
from functools import partial
from jax import lax, ops, vmap, jit, grad, random

from jax.config import config
config.update("jax_enable_x64", True)

import jraph
from jraph._src.models import ArrayTree
from jax_md import space
from jax_md.util import high_precision_sum, maybe_downcast

# Constant
from openmmtools.constants import ONE_4PI_EPS0
from simtk import openmm, unit


# typing
from aquaregia.utils import EnergyFn, Array
Quantity = unit.quantity.Quantity

def make_HarmonicBondForce(openmm_bond_force : openmm.HarmonicBondForce,
                           displacement_or_metric : space.DisplacementOrMetricFn
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

    def hookean_spring_energy_fn(R, parameter_dict, metric_fn):
        Ra, Rb = R[parameter_dict['p1'], :], R[parameter_dict['p2'], :]
        lengths, ks = maybe_downcast(parameter_dict['length']), maybe_downcast(parameter_dict['k'])
        drs = metric_fn(Ra, Rb)
        return high_precision_sum(0.5 * ks * jnp.power(drs - lengths, 2))

    return out_bond_terms, partial(hookean_spring_energy_fn, metric_fn = metric)


def make_HarmonicAngleForce(openmm_angle_force : openmm.HarmonicAngleForce,
                            displacement_fn: space.DisplacementFn
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

    def harmonic_angle_energy(R, parameter_dict, vmapped_displacement_fn):
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
                              displacement_fn : space.DisplacementFn
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

    def periodic_torsion_energy(R, parameter_dict, vmapped_displacement_fn):
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


def make_NonbondedForce(openmm_nonbonded_force : openmm.NonbondedForce,
                        displacement_or_metric : space.DisplacementOrMetricFn,
                        **kwargs,
                       )-> Tuple[ArrayTree, Callable]:
    """
    transcribe an openmm.NonbondedForce

    TODO : support periodic boundary conditions.
    """
    from jax import ops

    # we'll support two kinds of nonbonded methods: nocutoff and cutoff periodic
    nonbonded_method = openmm_nonbonded_force.getNonbondedMethod()
    periodic=False if nonbonded_method == 0 else True
    if periodic:
        raise RuntimeError(f"we only support non periodic nonbonded methods at the moment")

    # we need to assert the absence of some attributes...
    if openmm_nonbonded_force.getNumExceptionParameterOffsets() != 0:
        raise ValueError(f"we do not currently support `ExceptionParameterOffsets`")

    if openmm_nonbonded_force.getNumParticleParameterOffsets() != 0:
        raise ValueError(f"we do not currently support `ParticleParameterOffsets`")

    #vmap the displacement or metric fn
    metric = space.canonicalize_displacement_or_metric(displacement_or_metric)
    vmetric = vmap(vmap(metric, in_axes = (None,0)), (0, None))
    exception_vmetric = vmap(metric, in_axes=(0,0))

    #vmap chargeprod, sigma, epsilon computations
    vchargeProd = vmap(vmap(lambda x, y : x*y, in_axes = (None, 0)), in_axes=(0,None))
    vsigma = vmap(vmap(lambda x, y : 0.5 * (x + y), in_axes = (None, 0)), in_axes=(0,None))
    vepsilon = vmap(vmap(lambda x, y : jnp.sqrt(x * y), in_axes = (None, 0)), in_axes=(0,None))
    v_wlift = vsigma
    exception_vwlift = vmap(lambda x,y: 0.5 * (x + y), in_axes=(0,0))

    # extract nonbonded parameters
    num_particles = openmm_nonbonded_force.getNumParticles()
    num_exceptions = openmm_nonbonded_force.getNumExceptions()

    # make a parameter_dict
    nonbonded_parameters = {'particle_index': [], 'charge': [], 'sigma': [], 'epsilon': []}
    nonbonded_exception_parameters = {'p1': [], 'p2': [], 'chargeProd': [], 'sigma': [], 'epsilon': []}

    #query the particle terms
    for idx in range(num_particles):
        charge, sigma, epsilon = openmm_nonbonded_force.getParticleParameters(idx)
        nonbonded_parameters['particle_index'].append(idx)
        nonbonded_parameters['charge'].append(charge.value_in_unit_system(unit.md_unit_system))
        nonbonded_parameters['sigma'].append(sigma.value_in_unit_system(unit.md_unit_system))
        nonbonded_parameters['epsilon'].append(epsilon.value_in_unit_system(unit.md_unit_system))

    # query the exceptions
    for idx in range(num_exceptions):
        p1, p2, chargeProd, sigma, epsilon = openmm_nonbonded_force.getExceptionParameters(idx)
        nonbonded_exception_parameters['p1'].append(p1)
        nonbonded_exception_parameters['p2'].append(p2)
        nonbonded_exception_parameters['chargeProd'].append(chargeProd.value_in_unit_system(unit.md_unit_system))
        nonbonded_exception_parameters['sigma'].append(sigma.value_in_unit_system(unit.md_unit_system))
        nonbonded_exception_parameters['epsilon'].append(epsilon.value_in_unit_system(unit.md_unit_system))

    # make these objects jnp.
    out_nonbonded_parameters, out_nonbonded_exception_parameters = {}, {}
    for key, val in nonbonded_parameters.items():
        out_nonbonded_parameters[key] = Array(val)
    for key, val in nonbonded_exception_parameters.items():
        out_nonbonded_exception_parameters[key] = Array(val)

    # and add w terms:
    out_nonbonded_parameters['w'] = jnp.zeros(num_particles)
    out_nonbonded_exception_parameters['w'] = jnp.zeros(num_exceptions)

    nonbonded_parameters = {'standard': out_nonbonded_parameters,
                            'exceptions': out_nonbonded_exception_parameters}

    #let's make an appropriate nonbonded energy fn.
    def lifted_vac_nb(dr, chargeProd, sigma, epsilon, w):
        return lifted_vacuum_electrostatics(dr, chargeProd, w) + lifted_vacuum_lj(dr, sigma, epsilon, w)
    vlifted_vac_nb = vmap(lifted_vac_nb, in_axes=(0, 0, 0, 0, 0))


    def nonbonded_energy_fn(xs, parameters):
        standard_parameters = parameters['standard']
        exception_parameters = parameters['exceptions']

        # compute standard
        charges, sigmas, epsilons, ws = standard_parameters['charge'], standard_parameters['sigma'], standard_parameters['epsilon'], standard_parameters['w']
        drs = vmetric(xs, xs) + v_wlift(ws, ws) #compute lifted drs
        electrostatic_energies = jnp.vectorize(lifted_vacuum_electrostatics)(drs, vchargeProd(charges, charges))
        steric_energies = jnp.vectorize(lifted_vacuum_lj)(drs, vsigma(sigmas, sigmas), vepsilon(epsilons, epsilons))

        energies = electrostatic_energies + steric_energies #the diagonals should be nan...they are to be replaced...
        energies = ops.index_update(energies, ops.index[jnp.diag_indices(xs.shape[0])], 0.)

        #compute exceptions
        exception_index_pairs = jnp.stack([exception_parameters['p1'], exception_parameters['p2']], axis=-1)
        xs1, xs2 = xs[exception_index_pairs[:,0],:], xs[exception_index_pairs[:,1],:]
        exception_drs = exception_vmetric(xs1, xs2) + exception_parameters['w']
        exception_electrostatic_energies = jnp.vectorize(lifted_vacuum_electrostatics)(exception_drs, exception_parameters['chargeProd'])
        exception_steric_energies = jnp.vectorize(lifted_vacuum_lj)(exception_drs, exception_parameters['sigma'], exception_parameters['epsilon'])
        exception_energies = exception_electrostatic_energies + exception_steric_energies

        # remove and replace exceptions
        first_exc, second_exc = exception_index_pairs[:,0], exception_index_pairs[:,1]
        exception_removed_energies = ops.index_update(energies,
                                                  ops.index[jnp.concatenate([first_exc, second_exc]), jnp.concatenate([second_exc, first_exc])],
                                                  jnp.concatenate([exception_energies, exception_energies]))

        total_energy = 0.5 * high_precision_sum(exception_removed_energies)

        return total_energy

    return nonbonded_parameters, nonbonded_energy_fn

# Constant
KNOWN_FORCE_fns = {'HarmonicBondForce' : make_HarmonicBondForce,
                   'HarmonicAngleForce' : make_HarmonicAngleForce,
                   'PeriodicTorsionForce' : make_PeriodicTorsionForce,
                   'NonbondedForce' : make_NonbondedForce}

def make_canonical_energy_fn(system : openmm.System,
                             displacement_fn: space.DisplacementFn,
                             nonbonded_kwargs : Optional[Dict] = {},
                             fix_parameters : Optional[bool] = True
                            )-> Tuple[ArrayTree, EnergyFn]:
    """write a canonical energy function containing HarmonicBonds, HarmonicAngles, PeriodicTorsions and a NonbondedForce"""
    forces = system.getForces()
    force_dict = {force.__class__.__name__ : force for force in forces}

    out_params = {}
    out_fns = {}

    for force_name, force in force_dict.items():
        try:
            generator_fn = KNOWN_FORCE_fns[force_name]
            params, energy_fn = generator_fn(force, displacement_fn)
            out_params[force_name] = params
            out_fns[force_name] = energy_fn

        except Exception as e:
            raise RuntimeError(f"encountered error: {e}")


    def out_energy_fn(xs : Array, parameters : ArrayTree):
        running_sum = 0.
        for key in parameters.keys():
            running_sum = running_sum + out_fns[key](xs, parameters[key])

        return running_sum

    if fix_parameters:
        partial_out_energy_fn = partial(out_energy_fn, parameters = out_params)
        def returnable_fn(xs, placeholder_params):
            return partial_out_energy_fn(xs)
    else:
        returnable_fn = out_energy_fn

    return out_params, returnable_fn

# Utility
def make_scale_system(system : openmm.System,
                      target_ps : Sequence[int],
                      remove_angles : Optional[bool] = True,
                      nb_scale_factor : Optional[float] = 1e-3) -> openmm.System:
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
    for torsion_idx in range(torsion_f.getNumTorsions()):
        torsion_parameters = torsion_f.getTorsionParameters(torsion_idx)
        torsion_param_set = set(torsion_parameters[:4])
        per, phase, k = torsion_parameters[4:]
        if target_ps_set.issubset(torsion_param_set):
            torsion_f.setTorsionParameters(torsion_idx, *torsion_parameters[:4], per, phase, k * 0.)

    # nonbondeds
    for p_idx in range(nbf.getNumParticles()):
        charge, sigma, eps = nbf.getParticleParameters(p_idx)
        if p_idx in target_ps:
            nbf.setParticleParameters(p_idx, charge * nb_scale_factor, sigma, eps * nb_scale_factor)

    #nonbonded exceptions
    for nonbonded_exception_idx in range(nbf.getNumExceptions()):
        p1, p2, chargeProd, sigma, eps = nbf.getExceptionParameters(nonbonded_exception_idx)
        if {p1, p2}.issubset(target_ps_set):
            nbf.setExceptionParameters(nonbonded_exception_idx, p1, p2, chargeProd * nb_scale_factor, sigma, eps * nb_scale_factor)

    return out_system
