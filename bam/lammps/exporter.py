"""BAMExporter: Export RACE model for LAMMPS via chemtrain-deploy.

This module provides the BAMExporter class that wraps a trained RACE model
for export to MLIR format compatible with chemtrain-deploy's LAMMPS pair_style.

The exporter uses periodic=False mode (direct displacement), which was verified
to be compatible with periodic=True trained models when ghost atom positions
include cell-shift offsets (see test_periodic_compat.py).

Usage (MLIR export):
    exporter = BAMExporter(ckpt_path='ckpt_best.pkl', config=config)
    exporter.export()
    exporter.save('model-lammps.ptb')

Usage (standalone validation without MLIR export):
    exporter = BAMExporter(ckpt_path='ckpt_best.pkl', config=config)
    energies = exporter.compute_node_energies(positions, species, senders, receivers)
"""

import pickle
from typing import Optional

import jax
import jax.numpy as jnp
import jraph
from flax import nnx

from bam.models.race_nnx import RACE
from bam.data.atom_energies import ATOM_ENERGIES


class BAMExporter:
    """Export BAM-JAX RACE model for LAMMPS deployment.

    This class wraps a trained RACE model and provides:
    1. A standalone energy_fn for chemtrain-deploy Exporter integration
    2. A direct compute_node_energies method for validation without chemtrain

    The model is loaded with periodic=False, which computes displacements as:
        r = positions[senders] - positions[receivers]
    This is correct for LAMMPS because ghost atoms already have real-space
    coordinates (no Sij cell-shift needed).

    Attributes:
        r_cutoff: Cutoff radius for interactions (Angstrom).
        unit_style: LAMMPS unit style ("metal" = eV, Angstrom, ps).
        nbr_order: Neighbor list order for newton/non-newton settings.
        has_aux: Whether energy_fn returns auxiliary data.
    """

    # chemtrain-deploy compatible attributes
    unit_style = "metal"     # LAMMPS metal units: eV, Angstrom, ps
    nbr_order = [1, 1]       # single-hop neighbors sufficient per layer
    has_aux = False

    def __init__(
        self,
        ckpt_path: str,
        config: dict,
        cutoff: Optional[float] = None,
    ):
        """Initialize BAMExporter.

        Args:
            ckpt_path: Path to trained checkpoint (ckpt_best.pkl).
            config: Model configuration dictionary.
            cutoff: Override cutoff radius. If None, uses config['cutoff'].
        """
        self.r_cutoff = cutoff or config.get('cutoff', 6.0)
        self.config = config
        self._load_model(ckpt_path, config)

    def _load_model(self, ckpt_path: str, config: dict):
        """Load trained model and extract parameters."""
        # Load checkpoint
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)

        # Get atom energies
        atom_energies = ckpt.get('atom_energies', None)
        if atom_energies is None:
            atom_energies = config.get('atom_energies', ATOM_ENERGIES)

        self._atom_energies = atom_energies

        # Extract params (prefer EMA)
        if 'ema_params' in ckpt:
            params = ckpt['ema_params']
            print("Using EMA parameters")
        else:
            params = ckpt['params']
            print("Using regular parameters")

        # Convert numpy arrays to jax arrays (required for jax.export tracing)
        params = jax.tree.map(jnp.asarray, params)

        # Create model with periodic=False for LAMMPS deployment
        model = RACE(
            n_species=len(atom_energies),
            lmax=config.get('lmax', 3),
            hidden_irreps=config.get('hidden_irreps', '64x0e + 64x1o + 64x2e'),
            n_layers=config.get('n_layers', 5),
            radial_basis_size=config.get('radial_basis_size', 8),
            radial_mlp_size=config.get('radial_mlp_size', 64),
            radial_mlp_layers=config.get('radial_mlp_layers', 3),
            radial_polynomial_p=config.get('radial_polynomial_p', 2.0),
            mlp_init_scale=config.get('mlp_init_scale', 4.0),
            cutoff=self.r_cutoff,
            shift=0.0,
            scale=1.0,
            avg_n_neighbors=config.get('avg_n_neighbors', 25.0),
            atom_energies=atom_energies,
            periodic=False,   # LAMMPS: ghost atoms have real-space coords
            l_train=False,    # Add atom_energies as prior
            rngs=nnx.Rngs(0),
        )
        self._graphdef, _ = nnx.split(model, nnx.Param)
        self._params = params

        # Restore model for direct use
        self._model = nnx.merge(self._graphdef, params)

        n_params = sum(x.size for x in jax.tree.leaves(params))
        print(f"Model loaded: {n_params:,} parameters, cutoff={self.r_cutoff} A")

    def _build_graph(
        self,
        positions: jax.Array,
        species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> jraph.GraphsTuple:
        """Build a jraph GraphsTuple from flat arrays (no Sij, no cell).

        This converts the chemtrain-deploy graph format (senders/receivers)
        into the jraph format expected by RACE.node_energies().
        """
        n_atoms = positions.shape[0]
        n_edges = senders.shape[0]

        return jraph.GraphsTuple(
            n_node=jnp.array([n_atoms]),
            n_edge=jnp.array([n_edges]),
            nodes={
                'species': species,
                'positions': positions,
                'forces': jnp.zeros_like(positions),
            },
            edges={},  # No Sij needed (periodic=False)
            senders=senders,
            receivers=receivers,
            globals={
                'energy': jnp.zeros(1),
                'cell': jnp.zeros((1, 3, 3)),
                'volume': jnp.zeros(1),
                'stress': jnp.zeros((1, 6)),
                'cutoff': jnp.array([self.r_cutoff]),
            },
        )

    def compute_node_energies(
        self,
        positions: jax.Array,
        species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> jax.Array:
        """Compute per-atom energies directly (for validation).

        Args:
            positions: (N, 3) atom positions (local + ghost + padding).
            species: (N,) atom species indices.
            senders: (E,) edge sender indices.
            receivers: (E,) edge receiver indices.

        Returns:
            (N,) per-atom energies including isolated atom energy prior.
        """
        data = self._build_graph(positions, species, senders, receivers)

        # node_energies returns (N, 1) without atom_energies prior
        node_energies = self._model.node_energies(positions, None, None, data)

        # Add isolated atom energies (l_train=False path in __call__)
        node_energies = node_energies + jax.lax.stop_gradient(
            self._model.atom_energies[...][species, None]
        )

        return node_energies.squeeze(-1)  # (N, 1) -> (N,)

    def energy_fn(self, position, species, graph):
        """chemtrain-deploy Exporter interface.

        This method is called by chemtrain-deploy's _energy_fn wrapper,
        which handles force computation via jax.grad and newton's 3rd law.

        Args:
            position: (N, 3) atom positions (local + ghost + padding).
            species: (N,) atom species indices (int32).
            graph: SimpleSparseNeighborList with .senders, .receivers, .max_edges.

        Returns:
            (N,) per-atom energies.
        """
        return self.compute_node_energies(
            position, species, graph.senders, graph.receivers
        )

    def export(self):
        """Export model to MLIR via local deploy module (extracted from chemtrain)."""
        from bam.lammps.deploy.exporter import Exporter as ct_exporter_cls
        from bam.lammps.deploy import graphs

        # Create a chemtrain Exporter subclass dynamically with our energy_fn
        bam_exporter = self

        class _BAMChemtrainExporter(ct_exporter_cls):
            graph_type = graphs.SimpleSparseNeighborList
            r_cutoff = bam_exporter.r_cutoff
            unit_style = bam_exporter.unit_style
            nbr_order = bam_exporter.nbr_order
            has_aux = bam_exporter.has_aux

            def energy_fn(self_ct, position, species, graph):
                return bam_exporter.energy_fn(position, species, graph)

        self._ct_exporter = _BAMChemtrainExporter()
        self._ct_exporter.export()

    def save(self, filepath: str):
        """Save exported model to protobuf file.

        Args:
            filepath: Output path (e.g., 'model-lammps.ptb').
        """
        if not hasattr(self, '_ct_exporter'):
            raise RuntimeError(
                "Model has not been exported yet. Call export() first."
            )
        self._ct_exporter.save(filepath)
        print(f"Saved: {filepath}")

    def save_standalone(self, filepath: str):
        """Save model as standalone pickle (for validation without chemtrain).

        Args:
            filepath: Output path (e.g., 'model-standalone.pkl').
        """
        data = {
            'graphdef': self._graphdef,
            'params': self._params,
            'config': self.config,
            'cutoff': self.r_cutoff,
            'atom_energies': self._atom_energies,
            'unit_style': self.unit_style,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved standalone: {filepath}")
