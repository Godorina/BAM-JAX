# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exporting potential models to MLIR.

Extracted from chemtrain (https://github.com/tummfm/chemtrain) and adapted
to remove jax_md/jax_md_mod dependencies. Compatible with JAX >= 0.9.x.
"""

import abc
import functools

import jax
from jax import numpy as jnp, export, lax

from typing import Dict, Any, List, Tuple, Callable

from . import graphs, util
from . import model_pb2 as model_proto


class Exporter(metaclass=abc.ABCMeta):
    """Exports a potential model to an MLIR module.

    Subclass this and define energy_fn() to export a model.

    Attributes:
        graph_type: Neighborhood representation type (default: SimpleSparseNeighborList).
        nbr_order: [newton_order, non_newton_order] for neighbor requirements.
        r_cutoff: Cutoff radius for the potential.
        unit_style: LAMMPS unit style string.
        has_aux: Whether energy_fn returns (energies, aux_dict).
    """

    graph_type = graphs.SimpleSparseNeighborList
    nbr_order: List[int] = [1, 1]
    r_cutoff: float
    unit_style: str = "real"
    has_aux: bool = False

    _symbols: List[str] = None
    _constraints: List[str] = None
    _init_fns: List[Callable] = None
    _proto: model_proto.Model = None

    @abc.abstractmethod
    def energy_fn(self, position, species, graph):
        """Compute per-atom energies.

        Args:
            position: (N, 3) atom positions including ghost atoms.
            species: (N,) atom species indices.
            graph: Graph representation from graph_type.

        Returns:
            (N,) per-atom energy contributions.
        """
        pass

    @staticmethod
    @util.define_symbols("n_atoms")
    def _define_position_shapes(n_atoms, **kwargs):
        return (
            jax.ShapeDtypeStruct((n_atoms, 3), jnp.float32),  # position
            jax.ShapeDtypeStruct((n_atoms,), jnp.int32),       # species
            jax.ShapeDtypeStruct((), jnp.int32),               # n_local
            jax.ShapeDtypeStruct((), jnp.int32),               # n_ghost
            jax.ShapeDtypeStruct((), jnp.bool_),               # newton flag
        )

    def _add_shapes(self, init_fn):
        init_fn(self._symbols, self._constraints, self._init_fns)

    def _create_shapes(self):
        all_symbols = ",".join(self._symbols)
        symbols = {
            key: symb for key, symb in zip(
                self._symbols,
                export.symbolic_shape(all_symbols, constraints=self._constraints),
            )
        }
        shapes = []
        for init_fn in self._init_fns:
            shapes.extend(init_fn(**symbols))

        self._symbols, self._constraints, self._init_fns = [], [], []
        return shapes

    def _energy_fn(self, position, species, n_local, n_ghost, newton, *graph_args):
        """Internal energy/force computation with force via jax.grad."""
        valid_mask = jnp.arange(position.shape[0]) < (n_local + n_ghost)
        local_mask = jnp.arange(position.shape[0]) < n_local

        graph, build_statistics = self.graph_type.create_from_args(
            self.r_cutoff, self.nbr_order, position, species,
            local_mask, valid_mask, newton, *graph_args)
        graph = lax.stop_gradient(graph)

        @functools.partial(jax.grad, has_aux=True)
        def force_and_aux(pos):
            out = self.energy_fn(pos, species, graph)
            if self.has_aux:
                per_atom_energies, aux = out
            else:
                per_atom_energies = out
                aux = {}

            assert per_atom_energies.shape == local_mask.shape, (
                f"Per particle energies have shape {per_atom_energies.shape}, "
                f"but should have shape {local_mask.shape}."
            )

            # high_precision_sum: use float64 intermediate
            total_energy = util.high_precision_sum(
                jnp.where(valid_mask, per_atom_energies, jnp.float32(0.0)))
            local_energy = util.high_precision_sum(
                jnp.where(local_mask, per_atom_energies, jnp.float32(0.0)))

            force_energy = jnp.where(newton, local_energy, total_energy)
            force_energy = jnp.negative(force_energy)

            return force_energy, (per_atom_energies, aux)

        force, (energy, aux) = force_and_aux(position)
        predictions = dict(U=energy, F=force, **aux)

        for key, value in predictions.items():
            assert value.shape[0] == local_mask.size, (
                f"Wrong shape for prediction {value}. All model outputs "
                f"must be per-atom quantities."
            )

        return predictions, build_statistics

    def export(self) -> None:
        """Export the potential model to an MLIR module."""
        self._symbols: List[str] = []
        self._constraints: List[str] = []
        self._init_fns: List[Callable] = []

        proto = model_proto.Model()

        proto.neighbor_list.cutoff = self.r_cutoff
        proto.unit_style = self.unit_style

        assert len(self.nbr_order) == 2, (
            "nbr_order must contain order for newton and non-newton settings."
        )
        proto.neighbor_list.nbr_order.extend(self.nbr_order)
        self.graph_type.set_properties(proto)

        self._add_shapes(self._define_position_shapes)
        self._add_shapes(self.graph_type.create_symbolic_input_format)

        export_fn = jax.jit(self._energy_fn)
        shapes = self._create_shapes()

        exp: export.Exported = export.export(export_fn, platforms=["cuda"])(*shapes)

        predictions, statistics = exp.out_tree.unflatten(exp.out_avals)

        proto.neighbor_list.statistics_keys.extend(statistics.keys())
        proto.quantities.extend(predictions.keys())

        # Re-serialize targeting StableHLO v1.7.8 (chemtrain-deploy's version).
        # Default JAX serialization targets v1.13.4 which is incompatible.
        from jax._src.lib import xla_client
        mlir_text = exp.mlir_module()
        proto.mlir_module = xla_client._xla.mlir.serialize_portable_artifact(
            mlir_text, "1.7.8")

        self._proto = proto

    def __str__(self):
        assert self._proto is not None, (
            "Model has not been exported yet. Please call `export()` first."
        )
        return str(self._proto)

    def save(self, file: str) -> None:
        """Save exported protobuf to file.

        Args:
            file: Output path (e.g., 'model-lammps.ptb').
        """
        assert self._proto is not None, (
            "Model has not been exported yet. Please call `export()` first."
        )
        with open(file, "wb") as f:
            f.write(self._proto.SerializeToString())
