"""Multihead RACE ASE Calculator for inference.

For multi-head fine-tuned checkpoints. Selects a single head for inference.

Usage:
    from bam.inference.calculator_multihead import RACEMultiheadCalculator

    calc = RACEMultiheadCalculator(
        model_path='ckpt_best.pkl', cutoff=6.0,
        num_heads=2, head_idx=0,   # head 0 for inference
    )
    atoms.calc = calc
    energy = atoms.get_potential_energy()
"""

import json
import pickle
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import jraph
from flax import nnx

from ase.calculators.calculator import Calculator, all_changes

from bam.models.race_multihead_nnx import RACEMultihead
from bam.data.data_nnx import atoms_to_graph, nearest_bucket


class RACEMultiheadCalculator(Calculator):
    """ASE Calculator wrapper for multihead RACE model."""

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, model_path='ckpt_best.pkl', cutoff=6.0,
                 num_heads=2, head_idx=0, **kwargs):
        node_boundaries = kwargs.pop('node_boundaries', None)
        edge_boundaries = kwargs.pop('edge_boundaries', None)
        fixed_pad_node = kwargs.pop('fixed_pad_node', None)
        fixed_pad_edge = kwargs.pop('fixed_pad_edge', None)
        super().__init__(**kwargs)

        self.cutoff = cutoff
        self.num_heads = num_heads
        self.head_idx = head_idx
        self.node_boundaries = np.asarray(node_boundaries) if node_boundaries is not None else None
        self.edge_boundaries = np.asarray(edge_boundaries) if edge_boundaries is not None else None
        self.fixed_pad_node = fixed_pad_node
        self.fixed_pad_edge = fixed_pad_edge
        model_dir = Path(model_path).parent

        # Load checkpoint
        print(f"Loading multihead model from {model_path}...")
        with open(model_path, 'rb') as f:
            ckpt = pickle.load(f)

        # Get model config
        self.config = ckpt.get('config', {})
        if not self.config:
            config_candidates = [
                model_dir / 'input.json',
                model_dir / 'input_omat24.json',
                model_dir / 'input_nequip.json',
                model_dir / 'input_multihead.json',
                model_dir / 'config.json',
            ]
            config_path = next((c for c in config_candidates if c.exists()), None)
            if config_path is None:
                raise FileNotFoundError(f"No config found in checkpoint or {model_dir}")

            print(f"Loading config from {config_path}...")
            with open(config_path, 'r') as f:
                self.config = json.load(f)

        self.atom_energies = ckpt.get('atom_energies', None)

        # Extract params (prefer ema_params if available)
        if 'ema_params' in ckpt:
            self.params = ckpt['ema_params']
            print("Using EMA parameters")
        else:
            self.params = ckpt['params']
            print("Using regular parameters")

        self._setup_model()
        self._predict_fn = jax.jit(self._predict)
        print(f"Multihead model loaded: {num_heads} heads, using head_idx={head_idx}")

    def warmup(self, n_atoms=8, pad_node=None, pad_edge=None):
        """Run a dummy inference to trigger JIT compilation."""
        import ase
        pad_node = pad_node or self.fixed_pad_node or nearest_bucket(n_atoms + 1, 64)
        pad_edge = pad_edge or self.fixed_pad_edge or nearest_bucket(n_atoms * 40 + 1, 512)

        dummy = ase.Atoms('Si' * n_atoms,
                          positions=np.random.randn(n_atoms, 3) * 2.0,
                          cell=np.eye(3) * 10.0, pbc=True)
        graph = atoms_to_graph(dummy, cutoff=self.cutoff, self_interaction=False)
        graph = jraph.pad_with_graphs(graph, n_node=pad_node, n_edge=pad_edge, n_graph=2)

        # Inject head index
        n_graphs = graph.n_node.shape[0]
        new_globals = dict(graph.globals)
        new_globals["head"] = jnp.full((n_graphs,), self.head_idx, dtype=jnp.int32)
        graph = graph._replace(globals=new_globals)

        print(f"JIT warmup: pad=({pad_node}, {pad_edge}) ...", end=" ", flush=True)
        import time
        t0 = time.time()
        self._predict_fn(self.params, graph)
        print(f"done in {time.time() - t0:.1f}s")

    def _detect_n_species(self):
        """Detect n_species from checkpoint weight shapes."""
        import jax
        for path, leaf in jax.tree_util.tree_leaves_with_path(self.params):
            path_str = '/'.join(
                str(getattr(p, 'key', getattr(p, 'idx', p))) for p in path
            )
            if 'species_embedding' in path_str:
                return leaf.shape[0]
        return None

    def _setup_model(self):
        """Setup multihead model architecture from config."""
        config = self.config

        # Detect n_species from weights (most reliable)
        n_species = self._detect_n_species()
        if n_species is not None:
            print(f"Detected n_species={n_species} from checkpoint weights")
        else:
            n_species = 89
            print(f"Could not detect n_species, using default {n_species}")

        if self.atom_energies is None:
            self.atom_energies = np.zeros(n_species)

        if len(self.atom_energies) != n_species:
            print(f"Padding atom_energies from {len(self.atom_energies)} to {n_species}")
            padded = np.zeros(n_species)
            padded[:min(len(self.atom_energies), n_species)] = self.atom_energies[:n_species]
            self.atom_energies = padded

        model = RACEMultihead(
            n_species=n_species,
            num_heads=self.num_heads,
            lmax=config.get('lmax', 3),
            hidden_irreps=config.get('hidden_irreps', "128x0e + 128x1o + 128x2e"),
            n_layers=config.get('n_layers', 5),
            radial_basis_size=config.get('radial_basis_size', 8),
            radial_mlp_size=config.get('radial_mlp_size', 128),
            radial_mlp_layers=config.get('radial_mlp_layers', 3),
            radial_polynomial_p=config.get('radial_polynomial_p', 5.0),
            mlp_init_scale=config.get('mlp_init_scale', 4.0),
            cutoff=self.cutoff,
            shift=0.0,
            scale=1.0,
            avg_n_neighbors=config.get('avg_n_neighbors', 25.0),
            atom_energies=self.atom_energies,
            l_train=False,
            periodic=True,
            rngs=nnx.Rngs(0),
        )
        self.graphdef, _ = nnx.split(model, nnx.Param)

    def _predict(self, params, graph):
        model = nnx.merge(self.graphdef, params)
        return model(graph)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        graph = atoms_to_graph(atoms, cutoff=self.cutoff, self_interaction=False)
        n_atoms = len(atoms)
        n_edges = len(graph.senders)

        # Determine padding
        if self.fixed_pad_node is not None:
            pad_node = self.fixed_pad_node
        elif self.node_boundaries is not None:
            idx = np.searchsorted(self.node_boundaries, n_atoms + 1)
            pad_node = int(self.node_boundaries[min(idx, len(self.node_boundaries) - 1)])
        else:
            pad_node = nearest_bucket(n_atoms + 1, 64)

        if self.fixed_pad_edge is not None:
            pad_edge = self.fixed_pad_edge
        elif self.edge_boundaries is not None:
            idx = np.searchsorted(self.edge_boundaries, n_edges + 1)
            pad_edge = int(self.edge_boundaries[min(idx, len(self.edge_boundaries) - 1)])
        else:
            pad_edge = nearest_bucket(n_edges + 1, 512)

        # Safety: fall back to dynamic if structure exceeds fixed padding
        if n_atoms + 1 > pad_node:
            pad_node = nearest_bucket(n_atoms + 1, 64)
        if n_edges + 1 > pad_edge:
            pad_edge = nearest_bucket(n_edges + 1, 512)

        graph = jraph.pad_with_graphs(
            graph,
            n_node=pad_node,
            n_edge=pad_edge,
            n_graph=2,
        )

        # Inject head index into graph globals
        n_graphs = graph.n_node.shape[0]
        new_globals = dict(graph.globals)
        new_globals["head"] = jnp.full((n_graphs,), self.head_idx, dtype=jnp.int32)
        graph = graph._replace(globals=new_globals)

        energy, forces, stress = self._predict_fn(self.params, graph)

        self.results['energy'] = float(energy[0])
        self.results['forces'] = np.array(forces[:n_atoms])
        self.results['stress'] = np.array(stress[0])
