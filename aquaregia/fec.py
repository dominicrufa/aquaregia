"""free energy calculation modules"""
from typing import Sequence, Callable, Dict, Tuple, Optional, NamedTuple, Any
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import tqdm

from jax.config import config
config.update("jax_enable_x64", True)
from aquaregia.utils import Array, ArrayTree
from aquaregia.openmm import DEFAULT_VACUUM_R_CUTOFF
from jax_md.partition import NeighborList
from jax.flatten_util import ravel_pytree


def bool_vector_element_fn(_bool : bool,
                           _vector_element : float,
                           _lambda : float,
                           true_fn : Optional[Callable[[float, float], float]] = lambda _lam, _val: _lam*_val,
                           false_fn : Optional[Callable[[float, float], float]] = lambda _lam, _val: _val):
    """a simple vmappable function for performing operation with logic on protocols"""
    return jax.lax.cond(_bool, true_fn, false_fn, _lambda, _vector_element)

class FECProtocol(object):
    """
    generic class to produce protocol parameter dictionaries

    includes the following properties:
        parameter_dict : a vectorized (along leading axis) version of the `coupled_parameter_dict` that is vmappable
                         to a canonical `energy_function`
        vmap_axes_dict : a vectorized version of the parameter dict that specifies which axes to vmap
    """
    def __init__(self,
                 coupled_parameter_dict : dict,
                 lambdas : Optional[Array] = jnp.linspace(1., 0., 10)
                ):
        """
        Args:
            coupled_parameter_dict : canonical energy fn dict
            lambdas : an array of lambdas (typically from 1 to 0 (i.e. interacting to noninteracting))
        """
        flat_params, back_fn = ravel_pytree(coupled_parameter_dict)

        self._back_fn = back_fn
        self._flat_params = flat_params
        self._parameter_dict_template = back_fn(flat_params) # tricky way of making a copy
        self._lambdas = lambdas

    def _get_vmap_axes_dict(self):
        raise NotImplementedError(f"vmap_axes_dict is not implemented")

    def _get_parameter_dict(self):
        raise NotImplementedError(f"parameter dict is not implemented")

    @property
    def vmap_axes_dict(self):
        return self._get_vmap_axes_dict()

    @property
    def parameter_dict(self):
        return self._get_parameter_dict()

class AbsoluteFECProtocol(FECProtocol):
    """
    an alchemical protocol that will
        1. lift a prespecified index set into a 4th dimension (from lambda 1->0) to r_cutoff with a specified w_lift_fn
        2. optionally annihilate sterics
        3. optionally annihilate electrostatics
        4. do (3,4) with a scale function argument

    TODO : support lifting between ValenceForces as well as Nonbonded Exceptions (we currently don't do this.)
    """
    def __init__(self,
                 coupled_parameter_dict : dict,
                 alchemical_particle_indices : Sequence,
                 lambdas : Optional[Array] = jnp.linspace(1., 0., 10),
                 r_cutoff : Optional[float] = DEFAULT_VACUUM_R_CUTOFF,
                 annihilate_sterics : Optional[bool] = False,
                 annihilate_electrostatics : Optional[bool] = True,
                ):

        super().__init__(coupled_parameter_dict = coupled_parameter_dict, lambdas = lambdas)
        self._r_cutoff = r_cutoff
        self._alchemical_particle_indices = alchemical_particle_indices
        self._alchemical_mask = Array([idx in alchemical_particle_indices for idx in range(coupled_parameter_dict['NonbondedForce']['standard']['charge'].shape[0])])
        self._annihilate_sterics = annihilate_sterics
        self._annihilate_electrostatics = annihilate_electrostatics

        # the linear w lift fn: in both cases, the argument order is [mask, param_vec, lambdas]
        self._w_lift_fn = jax.vmap(jax.vmap(partial(bool_vector_element_fn,
                                  true_fn = lambda _lam, _val: (1. - _lam) * _val, # here, _val is `r_cutoff`
                                  false_fn = lambda _lam, _val: 0.), in_axes=(0, 0, None)), in_axes=(None, None, 0))

        self._linear_scale_fn = jax.vmap(jax.vmap(partial(bool_vector_element_fn,
                                  true_fn = lambda _lam, _val: _lam * _val, # here, _val is the parameter matrix
                                  false_fn = lambda _lam, _val: _val), in_axes=(0, 0, None)), in_axes=(None, None, 0))

        if self._annihilate_electrostatics:
            self._electrostatics_scale_fn = self._linear_scale_fn

        if annihilate_sterics:
            self._sterics_scale_fn = self._linear_scale_fn

    def _annihilate_electrostatics_parameterizer(self, parameter_dict):
        template_charges = self._parameter_dict_template['NonbondedForce']['standard']['charge']
        template_chargeProds = self._parameter_dict_template['NonbondedForce']['exceptions']['chargeProd']
        new_charges = self._electrostatics_scale_fn(self._alchemical_mask, template_charges, self._lambdas)
        new_chargeProds = self._electrostatics_scale_fn(self._alchemical_mask, template_chargeProds, self._lambdas)
        parameter_dict['NonbondedForce']['standard']['charge'] = new_charges
        parameter_dict['NonbondedForce']['exceptions']['chargeProd'] = new_chargeProds
        return parameter_dict

    def _annihilate_electrostatics_vmapper(self, axes_dict):
        axes_dict['NonbondedForce']['standard']['charge'] = 0
        axes_dict['NonbondedForce']['exceptions']['chargeProd'] = 0
        return axes_dict

    def _annihilate_sterics_parameterizer(self, parameter_dict):
        template_epsilons = self._parameter_dict_template['NonbondedForce']['standard']['epsilon']
        template_exception_epsilons = self._parameter_dict_template['NonbondedForce']['exceptions']['epsilon']
        new_epsilons = self._sterics_scale_fn(self._alchemical_mask, template_epsilons, self._lambdas)
        new_exception_epsilons = self._sterics_scale_fn(self._alchemical_mask, template_exception_epsilons, self._lambdas)
        parameter_dict['NonbondedForce']['standard']['epsilon'] = new_epsilons
        parameter_dict['NonbondedForce']['exceptions']['epsilon'] = new_exception_epsilons
        return parameter_dict

    def _annihilate_sterics_vmapper(self, axes_dict):
        axes_dict['NonbondedForce']['standard']['epsilon'] = 0
        axes_dict['NonbondedForce']['exceptions']['epsilon'] = 0
        return axes_dict


    def _get_parameter_dict(self):
        parameter_dict = self._back_fn(self._flat_params)

        # ws (always lift)
        ws = self._parameter_dict_template['NonbondedForce']['standard']['w']
        new_ws = self._w_lift_fn(self._alchemical_mask, jnp.ones_like(ws) * self._r_cutoff, self._lambdas)
        parameter_dict['NonbondedForce']['standard']['w'] = new_ws

        # electrostatics
        if self._annihilate_electrostatics:
            parameter_dict = self._annihilate_electrostatics_parameterizer(parameter_dict)
        if self._annihilate_sterics:
            parameter_dict = self._annihilate_sterics_parameterizer(parameter_dict)

        return parameter_dict


    def _get_vmap_axes_dict(self):
        if 'NonbondedForce' not in self._parameter_dict_template.keys():
            raise KeyError(f"not `NonbondedForce` supplied")

        #make the vmap param dict
        vmap_param_dict = {key: None for key in self._parameter_dict_template.keys()}
        vmap_param_dict['NonbondedForce'] = {key: None for key in self._parameter_dict_template['NonbondedForce'].keys()}
        vmap_param_dict['NonbondedForce']['standard'] = {key: None for key in self._parameter_dict_template['NonbondedForce']['standard'].keys()}

        # ws (always lift)
        vmap_param_dict['NonbondedForce']['standard']['w'] = 0

        if self._annihilate_electrostatics or self._annihilate_sterics: # accommodate exception keys
            vmap_param_dict['NonbondedForce']['exceptions'] = {key: None for key in self._parameter_dict_template['NonbondedForce']['exceptions'].keys()}

        if self._annihilate_electrostatics:
            vmap_param_dict = self._annihilate_electrostatics_vmapper(vmap_param_dict)
        if self._annihilate_sterics:
            vmap_param_dict = self._annihilate_sterics_vmapper(vmap_param_dict)
        return vmap_param_dict
