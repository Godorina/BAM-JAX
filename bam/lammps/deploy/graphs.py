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

"""Graph representations for exporting potential models.

Extracted from chemtrain (https://github.com/tummfm/chemtrain) and adapted
to remove jax_md/jax_md_mod dependencies. Only SimpleSparseNeighborList is
retained since that's all BAMExporter needs.
"""

import abc
import functools
import typing
from typing import NamedTuple, Tuple

import jax
from jax import export, numpy as jnp, lax

from . import util
from . import model_pb2 as model_proto


ListStatistics = typing.Dict


class NeighborList(metaclass=abc.ABCMeta):
    """Abstract base class for neighbor list graphs."""

    @staticmethod
    @abc.abstractmethod
    def set_properties(proto: model_proto.Model):
        pass

    @staticmethod
    @util.define_symbols("")
    @abc.abstractmethod
    def create_symbolic_input_format(*args, **kwargs):
        pass

    @staticmethod
    def create_from_args(r_cutoff, num_mpl, position, species,
                         ghost_mask, valid_mask, newton, *args,
                         half=True):
        pass


class NeighborListStatistics(typing.TypedDict, total=True):
    """Statistics for SimpleSparseNeighborList."""
    max_neighbors: typing.Required[int]
    overlong: typing.Optional[int]


class SimpleSparseNeighborList(NamedTuple):
    """Simple sparse neighbor list using precomputed sender/receiver indices.

    This is the primary graph representation for BAM LAMMPS deployment.
    LAMMPS provides precomputed neighbor lists; this class acts as an interface
    between LAMMPS neighbor data and the exported model.

    It filters edges beyond the model cutoff and prunes edges between ghost
    atoms that are irrelevant for correct force computation.

    Attributes:
        senders: Sender indices of edges.
        receivers: Receiver indices of edges.
        max_edges: Buffer for edge validity tracking.
    """

    senders: jax.Array
    receivers: jax.Array
    max_edges: jax.Array

    @staticmethod
    def set_properties(proto: model_proto.Model):
        proto.neighbor_list.type = model_proto.Model.SIMPLE_SPARSE
        proto.neighbor_list.half_list = True

    @staticmethod
    @util.define_symbols(
        "max_buffers, max_edges",
        ["max_edges <= 2 * max_buffers"]
    )
    def create_symbolic_input_format(max_buffers, max_edges, **kwargs):
        senders = jax.ShapeDtypeStruct((max_buffers,), jnp.int32)
        receivers = jax.ShapeDtypeStruct((max_buffers,), jnp.int32)
        buffer = jax.ShapeDtypeStruct((max_edges,), jnp.bool_)
        return senders, receivers, buffer

    @staticmethod
    def create_from_args(r_cutoff, nbr_order, position, species,
                         local_mask, valid_mask, newton,
                         *args) -> Tuple["SimpleSparseNeighborList",
                                         "NeighborListStatistics"]:
        invalid_idx = species.size

        senders, receivers, m = args
        max_edges = m.size

        # Remove edges beyond cutoff
        dists = jnp.linalg.norm(position[senders] - position[receivers], axis=-1)
        invalid = dists > r_cutoff

        vs = jnp.where(invalid, invalid_idx, senders)
        vr = jnp.where(invalid, invalid_idx, receivers)

        graph = SimpleSparseNeighborList(vs, vr, m)
        graph, max_neighbors = lax.cond(
            newton,
            functools.partial(prune_neighbor_list, max_edges=max_edges,
                              nbr_order=nbr_order[0], half_list=False),
            functools.partial(prune_neighbor_list, max_edges=max_edges,
                              nbr_order=nbr_order[1], half_list=True),
            graph, local_mask
        )

        statistics = NeighborListStatistics(
            max_neighbors=max_neighbors,
            overlong=jnp.sum(~invalid)
        )

        return graph, statistics


def prune_neighbor_list(list, local, max_edges, nbr_order: int,
                        half_list: bool = False):
    """Prune neighbor list to edges relevant for local atom force computation.

    Args:
        list: SimpleSparseNeighborList to prune.
        local: Boolean mask for local atoms.
        max_edges: Maximum number of edges after pruning.
        nbr_order: Maximum neighbor order for force computation.
        half_list: If True, input is a half list (i->j implies j->i).

    Returns:
        Pruned neighbor list and count of valid edges.
    """
    if half_list:
        senders = jnp.concat([list.senders, list.receivers], axis=0)
        receivers = jnp.concat([list.receivers, list.senders], axis=0)
    else:
        invalid_fill = jnp.full_like(list.senders, local.size)
        senders = jnp.concat([list.senders, invalid_fill], axis=0)
        receivers = jnp.concat([list.receivers, invalid_fill], axis=0)
    list = SimpleSparseNeighborList(senders, receivers, list.max_edges)

    def _update(reachable, _):
        reachable |= jax.ops.segment_max(
            reachable[list.senders], list.receivers, reachable.size)
        return reachable, _

    reachable, _ = lax.scan(_update, local, jnp.arange(nbr_order))
    mask = reachable[list.senders] & reachable[list.receivers]
    mask &= (list.senders < local.size) & (list.receivers < local.size)

    senders = jnp.where(mask, list.senders, local.size)
    receivers = jnp.where(mask, list.receivers, local.size)
    n_valid = jnp.sum(mask)

    mask, select = lax.top_k(mask, k=max_edges)
    senders = senders[select]
    receivers = receivers[select]

    return SimpleSparseNeighborList(senders, receivers, mask), n_valid
