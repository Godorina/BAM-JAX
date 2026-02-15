"""RACE ASE Calculator for inference.

Usage:
    from bam.inference.calculator import RACECalculator
    calc = RACECalculator(model_path='ckpt_best.pkl', cutoff=6.0)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
"""

import json
import pickle
from pathlib import Path

import numpy as np
import jax
import jraph
from flax import nnx

from ase.calculators.calculator import Calculator, all_changes

from bam.models.race_nnx import RACE
from bam.data.data_nnx import atoms_to_graph, nearest_bucket


class RACECalculator(Calculator):
    """ASE Calculator wrapper for RACE model."""

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, model_path='ckpt_best.pkl', cutoff=6.0, **kwargs):
        node_boundaries = kwargs.pop('node_boundaries', None)
        edge_boundaries = kwargs.pop('edge_boundaries', None)
        super().__init__(**kwargs)

        self.cutoff = cutoff
        self.node_boundaries = np.asarray(node_boundaries) if node_boundaries is not None else None
        self.edge_boundaries = np.asarray(edge_boundaries) if edge_boundaries is not None else None
        model_dir = Path(model_path).parent

        # Load checkpoint
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            ckpt = pickle.load(f)

        # Get model config: first check checkpoint, then look for input.json in same directory
        self.config = ckpt.get('config', {})
        if not self.config:
            config_candidates = [
                model_dir / 'input.json',
                model_dir / 'input_omat24.json',
                model_dir / 'input_nequip.json',
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
        print("Model loaded successfully")

    def _setup_model(self):
        """Setup model architecture from config."""
        config = self.config

        if self.atom_energies is None:
            try:
                from bam.data.atom_energies import ATOM_ENERGIES
                self.atom_energies = ATOM_ENERGIES
            except ImportError:
                self.atom_energies = np.zeros(120)

        model = RACE(
            n_species=len(self.atom_energies),
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

        if self.node_boundaries is not None:
            idx = np.searchsorted(self.node_boundaries, n_atoms + 1)
            pad_node = int(self.node_boundaries[min(idx, len(self.node_boundaries) - 1)])
        else:
            pad_node = nearest_bucket(n_atoms + 1, 64)

        n_edges = len(graph.senders)
        if self.edge_boundaries is not None:
            idx = np.searchsorted(self.edge_boundaries, n_edges + 1)
            pad_edge = int(self.edge_boundaries[min(idx, len(self.edge_boundaries) - 1)])
        else:
            pad_edge = nearest_bucket(n_edges + 1, 512)

        graph = jraph.pad_with_graphs(
            graph,
            n_node=pad_node,
            n_edge=pad_edge,
            n_graph=2,
        )

        energy, forces, stress = self._predict_fn(self.params, graph)

        self.results['energy'] = float(energy[0])
        self.results['forces'] = np.array(forces[:n_atoms])
        self.results['stress'] = np.array(stress[0])
