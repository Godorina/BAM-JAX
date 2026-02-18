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

"""Utilities for deploying models.

Extracted from chemtrain (https://github.com/tummfm/chemtrain) and adapted
to remove chemtrain/jax_md_mod dependencies.
"""

from typing import List, Any, Callable

import jax
import jax.numpy as jnp


def edge_to_atom_energies(n_atoms, per_edge_energies, senders):
    """Assigns energies of edges to per-atom energies.

    Args:
        n_atoms: Number of atoms
        per_edge_energies: Energies of each edge
        senders: Sender atoms of the edge

    Returns:
        Per-atom energies. Edge energies are assigned to the senders.
    """
    per_atom_energies = jax.ops.segment_sum(
        per_edge_energies, senders, num_segments=n_atoms
    )
    return per_atom_energies


def high_precision_sum(x, axis=None):
    """Sum with float64 intermediate for numerical stability."""
    return jnp.sum(x, axis=axis, dtype=jnp.float64).astype(x.dtype)


def define_symbols(symbols: str, constraints: List[str] = None):
    """Delays the definition of symbols until all symbols and constraints are known.

    Args:
        symbols: Comma-separated string of symbol names to define.
        constraints: List of constraint strings for symbolic shapes.
    """

    def decorator(f):
        symb = [s.strip() for s in symbols.split(',') if s.strip()]

        def apply_fn(**defined_symbols: Any):
            args = [defined_symbols.pop(s) for s in symb]
            return f(*args, **defined_symbols)

        def wrapped(s: List[Any], c: List, apply_fns: List[Callable]):
            assert set(s).isdisjoint(set(symb)), "Symbols already defined"
            s.extend(symb)
            apply_fns.append(apply_fn)
            if constraints is not None:
                c.extend(constraints)

        return wrapped
    return decorator
