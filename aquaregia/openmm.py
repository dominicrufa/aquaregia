from typing import Sequence, Callable, Dict, Tuple, Optional, NamedTuple, Any
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import lax, ops, vmap, jit, grad, random
from jax.scipy.special import erfc
from aquaregia.utils import polynomial_switching_fn, get_periodic_distance_calculator, get_mask
import tqdm


from jax.config import config
config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)

from aquaregia.utils import Array, ArrayTree
from jax_md import space
from jax_md.partition import NeighborList, NeighborFn
from jax_md.util import high_precision_sum, maybe_downcast

# Constant
from openmmtools.constants import ONE_4PI_EPS0
from simtk import openmm, unit

# typing
from aquaregia.utils import EnergyFn, Array
from aquaregia.tfn import DEFAULT_EPSILON
Quantity = unit.quantity.Quantity
KNOWN_FORCE_fns = ['HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce', 'NonbondedForce', 'CustomCentroidBondForce']
DEFAULT_VACUUM_R_CUTOFF = 10.
DEFAULT_VACUUM_R_SWITCH = 9.


class BaseForceConverter(object):
    """
    base class for converting forces from omm to jax
    """
    def __init__(self,
                 omm_force : Any,
                 displacement_fn : space.DisplacementFn,
                 particle_indices : Sequence):
        self._omm_force = omm_force
        self._displacement_fn = displacement_fn
        self._base_metric_fn = space.canonicalize_displacement_or_metric(displacement_fn)

        # in the case of given particles
        self._particle_set = set(particle_indices)
        self._particle_indices = sorted(particle_indices)
        self._given_to_ordered_indices = {given : idx for idx, given in enumerate(self._particle_indices)}
        self._ordered_to_given_indices = {idx : given for idx, given in enumerate(self._particle_indices)}

    def _make_parameter_dict(self):
        return {}

    def _make_jax_force_fn(self):
        def _fn(*args, **kwargs):
            raise NotImplementedError()

    def _get_given_to_ordered_index(self, idx):
        """recover a key error"""
        try:
            out_idx = self._given_to_ordered_indices[idx]
        except:
            out_idx = -1
        return out_idx

    def _omit(self, ordered_idx_sequence):
        """returns False if one ordered index is less than zero"""
        return any(_idx < 0 for _idx in ordered_idx_sequence)

    @property
    def force_fn(self):
        return self._make_jax_force_fn()

    @property
    def parameter_dict(self):
        return self._make_parameter_dict()

    @property
    def parameters_and_fn(self):
        parameter_dict = self._make_parameter_dict()
        fn = self._make_jax_force_fn()
        return parameter_dict, fn


class ValenceForceConverter(BaseForceConverter):
    """
    base class for converting an openmm force to a jax function
    """
    def __init__(self,
                 omm_force : Any,
                 displacement_fn : space.DisplacementFn,
                 particle_indices : Sequence):
        self._metric_fn = vmap(space.canonicalize_displacement_or_metric(displacement_fn), in_axes=(0,0))
        self._vdisplacement_fn = vmap(displacement_fn, in_axes = (0,0))
        super().__init__(omm_force = omm_force, displacement_fn = displacement_fn, particle_indices = particle_indices)

class HarmonicCOMRestraintForce(ValenceForceConverter):
    """convert a `CustomCentroidBondForce` that is shaped like a COM restraining Force"""
    def __init__(self,
                omm_force : openmm.CustomCentroidBondForce,
                displacement_fn : space.DisplacementFn,
                particle_indices : Sequence,
                particle_masses : Array):

        if omm_force.__class__.__name__ != 'CustomCentroidBondForce': raise ValueError("{omm_force.__class__.__name__} is not a `CustomCentroidBondForce`")
        super().__init__(omm_force = omm_force, displacement_fn = displacement_fn, particle_indices = particle_indices)
        assert particle_masses.shape[0] >= len(self._particle_indices), f"you must provide _all_ particle masses to the system"
        self._particle_masses = particle_masses

        # a few more miscellaneous assertions to make sure it is a COMForce
        assert self._omm_force.getEnergyFunction() == 'lambda_restraints * ((K/2)*distance(g1,g2)^2)', f"`CustomCentroidBondForce` has an energy function of the form {self._omm_force.getEnergyFunction()}"
        assert self._omm_force.getNumBonds() == 1, f"only one bond is allowed"
        assert self._omm_force.getNumGroups() == 2, f"only two groups are supported"
        assert self._omm_force.getNumPerBondParameters() == 1, f"only one bond parameter is supported"


    def _make_parameter_dict(self):
        [(g0, g1), (K,)] = self._omm_force.getBondParameters(0)
        g0_atoms, g0_weights = self._omm_force.getGroupParameters(g0)
        g1_atoms, g1_weights = self._omm_force.getGroupParameters(g1)
        assert len(g0_weights) == 0, f"only atom mass weights are allowed"
        assert len(g1_weights) == 0, f"only atom mass weights are allowed"

        total_atom_indices = set(g0_atoms).union(g1_atoms)
        if not total_atom_indices.issubset(self._particle_set):
            raise ValueError(f"the COM atoms are not a subset of the given indices")

        g0_masses = Array([self._particle_masses[_idx] for _idx in sorted(g0_atoms)])
        g1_masses = Array([self._particle_masses[_idx] for _idx in sorted(g1_atoms)])

        ordered_g0_atoms = Array([self._given_to_ordered_indices[_atom] for _atom in sorted(g0_atoms)])
        ordered_g1_atoms = Array([self._given_to_ordered_indices[_atom] for _atom in sorted(g1_atoms)])
        parameter_dict = {'K' : K,
                          'g0_indices' : ordered_g0_atoms,
                          'g1_indices' : ordered_g1_atoms,
                          'g0_masses': g0_masses,
                          'g1_masses': g1_masses,
                          'lambda_restraints' : 1.}

        return parameter_dict

    def _make_jax_force_fn(self):
        def _fn(R, parameter_dict, **kwargs):
            R_g0 = R[parameter_dict['g0_indices'],:]
            R_g1 = R[parameter_dict['g1_indices'],:]
            g0_COM = jnp.sum(R_g0 * parameter_dict['g0_masses'][..., jnp.newaxis], axis=0) / parameter_dict['g0_masses'].sum()
            g1_COM = jnp.sum(R_g1 * parameter_dict['g1_masses'][..., jnp.newaxis], axis=0) / parameter_dict['g1_masses'].sum()
            dr = self._base_metric_fn(g0_COM, g1_COM)
            return 0.5 * parameter_dict['K'] * jnp.power(dr, 2) * parameter_dict['lambda_restraints']
        return _fn

class HarmonicBondForceConverter(ValenceForceConverter):
    """convert a `HarmonicBondForce`"""
    def __init__(self,
                 omm_force : openmm.HarmonicBondForce,
                 displacement_fn : space.DisplacementFn,
                 particle_indices : Sequence,
                 **kwargs):
        if omm_force.__class__.__name__ != 'HarmonicBondForce': raise ValueError("{omm_force.__class__.__name__} is not a `HarmonicBondForce`")
        super().__init__(omm_force = omm_force, displacement_fn = displacement_fn, particle_indices = particle_indices)

    def _make_jax_force_fn(self):
        def _fn(R, parameter_dict, **kwargs):
            Ra, Rb = R[parameter_dict['p1'], :], R[parameter_dict['p2'], :]
            lengths, ks = maybe_downcast(parameter_dict['length']), maybe_downcast(parameter_dict['k'])
            drs = self._metric_fn(Ra, Rb)
            return high_precision_sum(0.5 * ks * jnp.power(drs - lengths, 2))
        return _fn

    def _make_parameter_dict(self):
        num_bonds = self._omm_force.getNumBonds()
        bond_terms = {'p1': [], 'p2' :[], 'length': [], 'k': []}
        for idx in range(num_bonds):
            p1, p2, length, k = self._omm_force.getBondParameters(idx)
            p1, p2 = [self._get_given_to_ordered_index(_p) for _p in [p1, p2]]
            if not self._omit([p1, p2]):
                bond_terms['p1'].append(p1)
                bond_terms['p2'].append(p2)
                bond_terms['length'].append(length.value_in_unit_system(unit.md_unit_system))
                bond_terms['k'].append(k.value_in_unit_system(unit.md_unit_system))

        out_bond_terms = {}
        for key, val in bond_terms.items():
            out_bond_terms[key] = jnp.asarray(val)
        return out_bond_terms

class HarmonicAngleForceConverter(ValenceForceConverter):
    """convert a `HarmonicAngleForce`"""
    def __init__(self,
                 omm_force : openmm.HarmonicAngleForce,
                 displacement_fn : space.DisplacementFn,
                 particle_indices : Sequence,
                 **kwargs):
        if omm_force.__class__.__name__ != 'HarmonicAngleForce': raise ValueError("{omm_force.__class__.__name__} is not a `HarmonicAngleForce`")
        super().__init__(omm_force = omm_force, displacement_fn = displacement_fn, particle_indices = particle_indices)

    def _make_jax_force_fn(self):
        def _fn(R, parameter_dict, **kwargs):
            r1s, r2s, r3s = R[parameter_dict['p1'],:], R[parameter_dict['p2'],:], R[parameter_dict['p3'],:]
            theta0s, ks = maybe_downcast(parameter_dict['theta0']), maybe_downcast(parameter_dict['k'])
            r21s = self._vdisplacement_fn(r1s, r2s)
            r23s = self._vdisplacement_fn(r3s, r2s)
            tops = high_precision_sum(jnp.multiply(r21s, r23s), axis=-1)
            bots = jnp.linalg.norm(r21s, axis=-1) * jnp.linalg.norm(r23s, axis=-1)
            tb = tops / bots
            angles = jnp.arccos(tb)
            return high_precision_sum(0.5 * ks * jnp.power(angles - theta0s, 2))
        return _fn

    def _make_parameter_dict(self):
        num_angles = self._omm_force.getNumAngles()
        angle_terms = {'p1': [], 'p2' :[], 'p3': [], 'theta0': [], 'k': []}
        for idx in range(num_angles):
            p1, p2, p3, theta0, k = self._omm_force.getAngleParameters(idx)
            p1, p2, p3 = [self._get_given_to_ordered_index(_p) for _p in [p1, p2, p3]]
            if not self._omit([p1, p2, p3]):
                angle_terms['p1'].append(p1)
                angle_terms['p2'].append(p2)
                angle_terms['p3'].append(p3)
                angle_terms['theta0'].append(theta0.value_in_unit_system(unit.md_unit_system))
                angle_terms['k'].append(k.value_in_unit_system(unit.md_unit_system))

        out_angle_terms = {}
        for key, val in angle_terms.items():
            out_angle_terms[key] = jnp.asarray(val)
        return out_angle_terms

class PeriodicTorsionForceConverter(ValenceForceConverter):
    """convert a `PeriodicTorsionForce`"""
    def __init__(self,
                 omm_force : openmm.PeriodicTorsionForce,
                 displacement_fn : space.DisplacementFn,
                 particle_indices : Sequence,
                 **kwargs):
        if omm_force.__class__.__name__ != 'PeriodicTorsionForce': raise ValueError(f"{omm_force.__class__.__name__} is not a `PeriodicTorsionsForce`")
        super().__init__(omm_force = omm_force, displacement_fn = displacement_fn, particle_indices = particle_indices)

    def _make_jax_force_fn(self):
        def _fn(R, parameter_dict, **kwargs):
            ci, cj, ck, cl = R[parameter_dict['p1'],:], R[parameter_dict['p2'],:], R[parameter_dict['p3'],:], R[parameter_dict['p4'], :]
            periodicities, phases, ks = parameter_dict['periodicity'], parameter_dict['phase'], parameter_dict['k']
            rij = self._vdisplacement_fn(cj, ci)
            rkj = self._vdisplacement_fn(cj, ck)
            rkl = self._vdisplacement_fn(cl, ck)
            n1 = jnp.cross(rij, rkj)
            n2 = jnp.cross(rkj, rkl)
            y = jnp.sum(jnp.multiply(jnp.cross(n1, n2), rkj / jnp.linalg.norm(rkj, axis=-1, keepdims=True)), axis=-1)
            x = jnp.sum(jnp.multiply(n1, n2), axis=-1)
            torsion_angles = jnp.arctan2(y, x)
            return high_precision_sum(ks * (1. + jnp.cos(periodicities * torsion_angles - phases)))
        return _fn

    def _make_parameter_dict(self):
        num_torsions = self._omm_force.getNumTorsions()
        torsion_terms = {'p1': [], 'p2' :[], 'p3': [], 'p4': [], 'periodicity': [], 'phase': [], 'k': []}
        for idx in range(num_torsions):
            p1, p2, p3, p4, per, phase, k = self._omm_force.getTorsionParameters(idx)
            p1, p2, p3, p4 = [self._get_given_to_ordered_index(_p) for _p in [p1, p2, p3, p4]]
            if not self._omit([p1, p2, p3, p4]):
                torsion_terms['p1'].append(p1)
                torsion_terms['p2'].append(p2)
                torsion_terms['p3'].append(p3)
                torsion_terms['p4'].append(p4)
                torsion_terms['periodicity'].append(per)
                torsion_terms['phase'].append(phase.value_in_unit_system(unit.md_unit_system))
                torsion_terms['k'].append(k.value_in_unit_system(unit.md_unit_system))

        out_torsion_terms = {}
        for key, val in torsion_terms.items():
            out_torsion_terms[key] = jnp.asarray(val)
        return out_torsion_terms

# helper function for nonbonded force
def lifted_vacuum_electrostatics(dr : float,
                                 chargeProd : float,
                                 r_cutoff : float,
                                 r_switch : float) -> float:
    unscaled_out = ONE_4PI_EPS0 * chargeProd / (dr)
    scale = polynomial_switching_fn(r=dr, r_cutoff = r_cutoff, r_switch = r_switch)
    return scale * unscaled_out

def lifted_vacuum_lj(dr : float,
                     sigma : float,
                     epsilon : float,
                     r_cutoff : float,
                     r_switch : float):
    red_sigma = (sigma / dr) ** 6
    unscaled_out = 4. * epsilon * (red_sigma**2 - red_sigma)
    scale = polynomial_switching_fn(r=dr, r_cutoff = r_cutoff, r_switch = r_switch)
    return scale * unscaled_out

def lifted_rf_electrostatics(dr, chargeProd, r_cutoff, r_switch, delta):
    # NOTE : this is wrong, strictly speaking
    # WARNING : make this match...
    alpha = jnp.sqrt(-jnp.log(2.*delta)) / r_cutoff #compute alpha
    val = ONE_4PI_EPS0 * chargeProd  * (erfc(alpha * dr)/(dr) - erfc(alpha * r_cutoff)/r_cutoff)
    multiplier = lax.cond(dr > r_cutoff, lambda _x : 0., lambda _x : 1., None)
    return val * multiplier

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

class NonbondedForceConverter(ValenceForceConverter):
    """
    convert a `NonbondedForce`;
    TODO : implement periodic (especially lifting into a nonperiodic 4th dimension)
    TODO : refactor exception calculation so we don't have to recompute drs twice.
        """
    def __init__(self,
                 omm_force : openmm.NonbondedForce,
                 displacement_fn : space.DisplacementFn,
                 particle_indices : Sequence,
                 vacuum_r_cutoff : Optional[float] = 10.,
                 vacuum_r_switch : Optional[float] = 9.,
                 **kwargs
                 ):
        if omm_force.__class__.__name__ != 'NonbondedForce': raise ValueError("{omm_force.__class__.__name__} is not a `NonbondedForce`")
        super().__init__(omm_force = omm_force, displacement_fn = displacement_fn, particle_indices = particle_indices)

        # we'll support two kinds of nonbonded methods: nocutoff and cutoff periodic
        self._nonbonded_method = self._omm_force.getNonbondedMethod()
        self._periodic=False if self._nonbonded_method == 0 else True

        # extract nonbonded parameters
        self._num_particles = self._omm_force.getNumParticles()
        self._num_exceptions = self._omm_force.getNumExceptions()

        # we need to assert the absence of some attributes...
        if self._omm_force.getNumExceptionParameterOffsets() != 0: raise NotImplementedError(f"we do not currently support `ExceptionParameterOffsets`")
        if self._omm_force.getNumParticleParameterOffsets() != 0: raise NotImplementedError(f"we do not currently support `ParticleParameterOffsets`")

        self._r_cutoff = self._omm_force.getCutoffDistance().value_in_unit_system(unit.md_unit_system) if self._periodic else vacuum_r_cutoff
        self._r_switch = self._omm_force.getSwitchingDistance().value_in_unit_system(unit.md_unit_system) if self._periodic else vacuum_r_switch
        self._delta = self._omm_force.getEwaldErrorTolerance()

        if self._periodic:
            raise NotImplementedError(f"periodic needs yet another test. currently not implemented")
        else:
            self._electrostatic_fn = partial(lifted_vacuum_electrostatics, r_cutoff = self._r_cutoff, r_switch = self._r_switch)
            self._steric_fn = partial(lifted_vacuum_lj, r_cutoff = self._r_cutoff, r_switch = self._r_switch)

        self._vmetric = get_periodic_distance_calculator(self._base_metric_fn, self._r_cutoff)

        # the exception sterics and electrostatics are always vacuum
        self._exception_electrostatic_fn = self._electrostatic_fn
        self._exception_steric_fn = self._steric_fn

        def find_indices(arr, nbr_list_idx): return jnp.where(jnp.isin(nbr_list_idx, arr), True, False) #returnable logical `True` if in exceptions
        self._find_indices = find_indices

        #vmap chargeprod, sigma, epsilon computations
        self._vchargeProd = vmap(vmap(lambda x, y : x*y, in_axes=(None, 0)))
        self._vsigma = vmap(vmap(lambda x, y : 0.5 * (x + y), in_axes = (None, 0)))
        self._vepsilon = vmap(vmap(lambda x, y : jnp.sqrt(x * y), in_axes = (None, 0)))

    def _handle_standards(self):
        """return a dictionary of standard nonbonded force parameters"""
        num_particles = self._omm_force.getNumParticles()
        nonbonded_parameters = {'particle_index': [], 'charge': [], 'sigma': [], 'epsilon': [], 'w': []}
        for given_idx in range(num_particles):
            charge, sigma, epsilon = self._omm_force.getParticleParameters(given_idx)
            idx = self._get_given_to_ordered_index(given_idx)
            if not self._omit([idx]):
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

    def _handle_exceptions(self):
        """
        handle the exceptions; this is a little more involved

        """
        num_particles =  len(self._particle_set)
        num_exceptions = self._omm_force.getNumExceptions()
        nonbonded_exception_parameter_template = [[] for i in range(num_particles)] #empty

        template_nonbonded_exception_parameters = {'chargeProd': [[] for i in range(num_particles)],
                                                   'sigma': [[] for i in range(num_particles)],
                                                   'epsilon': [[] for i in range(num_particles)]
                                                   }
        nonbonded_exception_parameters = {}

        # query the exceptions
        for given_idx in range(num_exceptions):
            p1, p2, chargeProd, sigma, epsilon = self._omm_force.getExceptionParameters(given_idx) #query
            p1, p2 = [self._get_given_to_ordered_index(_p) for _p in [p1, p2]]
            if not self._omit([p1, p2]):
                chargeProd = chargeProd.value_in_unit_system(unit.md_unit_system)
                sigma = sigma.value_in_unit_system(unit.md_unit_system)
                epsilon = epsilon.value_in_unit_system(unit.md_unit_system)

                # order particle indices
                particle1 = p1 if p1 < p2 else p2 #order the parameters
                particle2 = p2 if p1 < p2 else p1

                nonbonded_exception_parameter_template[particle1].append(particle2)
                nonbonded_exception_parameter_template[particle2].append(particle1)

                for key, val in zip(template_nonbonded_exception_parameters.keys(), [chargeProd, sigma, epsilon]):
                    template_nonbonded_exception_parameters[key][particle1].append(val)
                    template_nonbonded_exception_parameters[key][particle2].append(val)


        # rework the nonbonded_exception_parameter_template for padding purposes
        arg_max = jnp.argmax(Array([len(q) for q in nonbonded_exception_parameter_template])) #pull the list index with the largest number of exceptions
        max_exceptions = len(nonbonded_exception_parameter_template[arg_max]) #pull out the max number of exceptions
        padded_nonbonded_exception_parameter_template = Array([q + [num_particles]*(max_exceptions - len(q)) for q in nonbonded_exception_parameter_template])
        exception_neighbor_list = NeighborList(idx = padded_nonbonded_exception_parameter_template,
                                              reference_position = None,
                                              did_buffer_overflow = False,
                                              max_occupancy = 0.,
                                              cell_list_capacity = None,
                                              format = None,
                                              update_fn = None)

        for key in template_nonbonded_exception_parameters.keys(): #add pads
            nonbonded_exception_parameters[key] = Array([q + [0.]*(max_exceptions - len(q)) for q in template_nonbonded_exception_parameters[key]])
        nonbonded_exception_parameters['exception_template'] = padded_nonbonded_exception_parameter_template
        exception_overfill_mask = exception_neighbor_list.idx != num_particles
        return exception_neighbor_list, exception_overfill_mask, nonbonded_exception_parameters

    def _make_jax_force_fn(self):
        def _fn(R,
                neighbor_list,
                parameter_dict,
                exception_neighbor_list,
                exception_overfill_mask,
                **kwargs):
            #separate out standard and exception parameters
            standard_parameters = parameter_dict['standard']
            exception_parameters = parameter_dict['exceptions']

            # compute standard
            charges, sigmas, epsilons, ws = standard_parameters['charge'], standard_parameters['sigma'], standard_parameters['epsilon'], standard_parameters['w']
            augmented_R = jnp.concatenate([R, ws[..., jnp.newaxis]], axis=-1) # lift R from R^3 into R^4
            drs = self._vmetric(augmented_R, neighbor_list) #compute lifted drs
            drs = jnp.where(drs <= DEFAULT_EPSILON, drs + DEFAULT_EPSILON, drs) # pad drs
            electrostatic_energies = jnp.vectorize(self._electrostatic_fn)(drs,
                                                                     self._vchargeProd(charges, charges[neighbor_list.idx]))
            steric_energies = jnp.vectorize(self._steric_fn)(drs,
                                                       self._vsigma(sigmas, sigmas[neighbor_list.idx]),
                                                       self._vepsilon(epsilons, epsilons[neighbor_list.idx]))

            energies = electrostatic_energies + steric_energies #the diagonals should be nan...they are to be replaced...
            overfill_mask = get_mask(neighbor_list) # True on the particle indices that aren't overfilled
            exception_mask = vmap(self._find_indices, in_axes=(0,0))(exception_neighbor_list.idx, neighbor_list.idx) # True if in exception
            not_exception_mask = jnp.invert(exception_mask) # False if in exception
            overfill_mask_and_not_exceptions = jnp.logical_and(overfill_mask, not_exception_mask)
            energies = jnp.where(overfill_mask_and_not_exceptions, energies, 0.)

            #compute exceptions
            exception_drs = self._vmetric(augmented_R, exception_neighbor_list)
            exception_drs = jnp.where(exception_drs <= DEFAULT_EPSILON, exception_drs + DEFAULT_EPSILON, exception_drs) # pad exception drs
            exception_elecrostatics = jnp.vectorize(self._exception_electrostatic_fn)(exception_drs,
                                                                                exception_parameters['chargeProd'])
            exception_sterics = jnp.vectorize(self._exception_steric_fn)(exception_drs,
                                                         exception_parameters['sigma'],
                                                         exception_parameters['epsilon'])
            exception_energies = exception_elecrostatics + exception_sterics
            exception_energies = jnp.where(exception_overfill_mask, exception_energies, 0.)
            total_energy = 0.5 * ( high_precision_sum(energies) + high_precision_sum(exception_energies) )
            return total_energy
        return _fn

    def _make_parameter_dict(self):
        standard_nonbonded_parameters = self._handle_standards()
        exception_neighbor_list, exception_overfill_mask, exception_nonbonded_parameters = self._handle_exceptions()
        nonbonded_parameters = {'standard': standard_nonbonded_parameters,
                                'exceptions': exception_nonbonded_parameters,
                                'exception_neighbor_list' : exception_neighbor_list,
                                'exception_overfill_mask' : exception_overfill_mask}
        return nonbonded_parameters

    @property
    def parameters_and_fn(self):
        parameter_dict = self._make_parameter_dict()
        fn = self._make_jax_force_fn()
        partial_fn = partial(fn,
                             exception_neighbor_list = parameter_dict['exception_neighbor_list'],
                             exception_overfill_mask = parameter_dict['exception_overfill_mask'])
        del parameter_dict['exception_neighbor_list']
        del parameter_dict['exception_overfill_mask']
        return parameter_dict, partial_fn

def fail_on_constraints(system, particle_indices):
    num_constraints = system.getNumConstraints()
    if num_constraints > 0:
        for constraint_idx in range(num_constraints):
            p1, p2, distance = system.getConstraintParameters(constraint_idx)
            if p1 in particle_indices or p2 in particle_indices:
                raise NotImplementedError(f"we do currently support constraints")

def make_canonical_energy_fn(system : openmm.System,
                             displacement_fn: space.DisplacementFn,
                             nonbonded_kwargs_dict : Optional[dict] = {'vacuum_r_cutoff' : DEFAULT_VACUUM_R_CUTOFF,
                                                                       'vacuum_r_switch' : DEFAULT_VACUUM_R_SWITCH},
                             particle_indices : Optional[Sequence] = [],
                             allow_constraints : Optional[bool] = False)-> Tuple[ArrayTree, EnergyFn]:
    """write a canonical energy function containing HarmonicBonds, HarmonicAngles, PeriodicTorsions and a NonbondedForce"""
    forces = system.getForces()
    force_dict = {force.__class__.__name__ : force for force in forces}
    particle_indices = particle_indices if len(particle_indices) != 0 else list(range(system.getNumParticles()))
    if not allow_constraints: fail_on_constraints(system, particle_indices) # check for no constraints
    out_params, out_fns = {}, {}

    for force_name, force in force_dict.items():
        if force_name == 'HarmonicBondForce':
            force_converter = HarmonicBondForceConverter(omm_force = force,
                                                         displacement_fn = displacement_fn,
                                                         particle_indices = particle_indices)
        elif force_name == 'HarmonicAngleForce':
            force_converter = HarmonicAngleForceConverter(omm_force = force,
                                                         displacement_fn = displacement_fn,
                                                         particle_indices = particle_indices)
        elif force_name == 'PeriodicTorsionForce':
            force_converter = PeriodicTorsionForceConverter(omm_force = force,
                                                         displacement_fn = displacement_fn,
                                                         particle_indices = particle_indices)
        elif force_name == 'NonbondedForce':
            force_converter = NonbondedForceConverter(omm_force = force,
                                                         displacement_fn = displacement_fn,
                                                         particle_indices = particle_indices,
                                                         **nonbonded_kwargs_dict)
        elif force_name == 'CustomCentroidBondForce':
            particle_masses = Array([system.getParticleMass(_i).value_in_unit_system(unit.md_unit_system)
                                     for _i in range(system.getNumParticles())])
            force_converter = HarmonicCOMRestraintForce(omm_force = force,
                                                        displacement_fn = displacement_fn,
                                                        particle_indices = particle_indices,
                                                        particle_masses = particle_masses)
        else:
            raise NotImplementedError(f"{force_name} is not an implemented force object")

        # now pull the parameters and the functions.
        parameter_dict, energy_fn = force_converter.parameters_and_fn
        out_fns[force_name] = energy_fn
        out_params[force_name] = parameter_dict
        del force_converter


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
