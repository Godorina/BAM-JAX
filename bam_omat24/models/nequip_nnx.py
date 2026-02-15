"""NequIP model implementation using Flax NNX.

This module implements the NequIP (Neural Equivariant Interatomic Potential) model
using Flax NNX framework with e3nn_jax for equivariant operations.
"""

#import json
import math
from typing import Callable, Optional, Sequence

import e3nn_jax as e3nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jraph

from bam_omat24.models.linear_nnx import Linear as e3nn_nnx_Linear
#import sys

def bessel_basis(x: jax.Array, num_basis: int, r_max: jax.Array) -> jax.Array:
    """Compute Bessel radial basis functions.

    Args:
        x: Distance array of shape (n_edges,)
        num_basis: Number of basis functions
        r_max: Cutoff radius

    Returns:
        Bessel basis values of shape (n_edges, num_basis)
    """
    r_max = r_max[:,None]
    x = x[:,None]
    prefactor = 2.0 / r_max
    bessel_weights = jnp.linspace(1.0, num_basis, num_basis) * jnp.pi

    return prefactor * jnp.where(
        x == 0.0,
        bessel_weights / r_max,
        jnp.sin(bessel_weights * x / r_max) / x,
    )


def polynomial_cutoff(x: jax.Array, r_max: jax.Array, p: float) -> jax.Array:
    """Compute polynomial cutoff function.

    Args:
        x: Distance array
        r_max: Cutoff radius
        p: Polynomial order

    Returns:
        Cutoff values (smoothly goes to zero at r_max)
    """
    factor = 1.0 / r_max
    x = x * factor
    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * jnp.power(x, p))
    out = out + (p * (p + 2.0) * jnp.power(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * jnp.power(x, p + 2.0))
    return out * jnp.where(x < 1.0, 1.0, 0.0)


def separated_layer_norm (
    input: e3nn.IrrepsArray
)->e3nn.IrrepsArray:

    xi_s = jnp.zeros ( (input.shape[0], 1) )
    cnt  = 0
    for (mul, ir), chunk in zip (input.irreps, input.chunks):
        if not ir.is_scalar():
            xi = chunk.reshape(-1,mul*ir.dim)
            xi_s += jnp.sum(xi**2, axis=-1, keepdims=True)
            cnt  += mul*ir.dim
    xi_rms_nonscalar = jnp.sqrt ( xi_s/jnp.maximum(cnt, 1) + 1e-6)

    new_chunks = []
    for (mul, ir), chunk in zip (input.irreps, input.chunks):
        if ir.is_scalar():
            xi = chunk.reshape(-1,mul)
            xi_centered = xi - jnp.mean(xi, axis=-1, keepdims=True)
            xi_rms_scalar = jnp.sqrt (jnp.mean(xi_centered**2, axis=-1, keepdims=True)+1e-6)
            chunk = (xi_centered/xi_rms_scalar).reshape(-1, mul, ir.dim)
        else:
            xi = chunk.reshape(-1,mul*ir.dim)
            chunk = (xi/xi_rms_nonscalar).reshape(-1, mul, ir.dim)

        new_chunks.append(chunk)

    return e3nn.from_chunks(
        input.irreps,
        new_chunks,
        (input.shape[0],)
    )

class Linear(nnx.Module):
    """Simple linear layer with optional bias."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        use_bias: bool = True,
        init_scale: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        scale = math.sqrt(init_scale / in_size)
        self.weights = nnx.Param(
            jax.random.normal(rngs.params(), (in_size, out_size)) * scale
        )
        self.use_bias = use_bias
        self.bias: nnx.Param[jax.Array] | None
        if use_bias:
            self.bias = nnx.Param(jnp.zeros(out_size))
        else:
            self.bias = None

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.dot(x, self.weights[...])
        if self.use_bias:
            x = x + self.bias[...]
        return x


class MLP(nnx.Module):
    """Multi-layer perceptron with configurable activation."""

    def __init__(
        self,
        sizes: Sequence[int],
        activation: Callable = jax.nn.silu,
        init_scale: float = 1.0,
        use_bias: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.activation = activation
        layers = []

        for i in range(len(sizes) - 1):
            layer = Linear(
                sizes[i],
                sizes[i + 1],
                use_bias=use_bias,
                # don't scale last layer since no activation
                init_scale=init_scale if i < len(sizes) - 2 else 1.0,
                rngs=rngs,
            )
            layers.append(layer)

        self.layers = nnx.List(layers)
        #self.layers = layers

    def __call__(self, x: jax.Array) -> jax.Array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


class NequipConvolution(nnx.Module):
    """NequIP convolution layer with equivariant message passing."""

    def __init__(
        self,
        input_irreps: e3nn.Irreps,
        output_irreps: e3nn.Irreps,
        sh_irreps: e3nn.Irreps,
        n_species: int,
        radial_basis_size: int,
        radial_mlp_size: int,
        radial_mlp_layers: int,
        mlp_init_scale: float,
        avg_n_neighbors: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.output_irreps = output_irreps
        self.avg_n_neighbors = avg_n_neighbors

        tp_irreps = e3nn.tensor_product(
            input_irreps, sh_irreps, filter_ir_out=output_irreps
        )

        self.linear_1 = e3nn_nnx_Linear(
            irreps_in=input_irreps,
            irreps_out=input_irreps,
            rngs=rngs,
        )

        self.radial_mlp = MLP(
            sizes=[radial_basis_size]
            + [radial_mlp_size] * radial_mlp_layers
            + [tp_irreps.num_irreps],
            activation=jax.nn.silu,
            use_bias=False,
            init_scale=mlp_init_scale,
            rngs=rngs,
        )

        # add extra irreps to output to account for gate
        gate_irreps = e3nn.Irreps(
            f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e"
        )
        gated_output_irreps = (output_irreps + gate_irreps).regroup()

        self.linear_2 = e3nn_nnx_Linear(
            irreps_in=tp_irreps,
            irreps_out=gated_output_irreps,
            rngs=rngs,
        )

        # skip connection has per-species weights
        self.skip = e3nn_nnx_Linear(
            irreps_in=input_irreps,
            irreps_out=gated_output_irreps,
            linear_type="indexed",
            num_indexed_weights=n_species,
            force_irreps_out=True,
            rngs=rngs,
        )

    def __call__(
        self,
        features: e3nn.IrrepsArray,
        species: jax.Array,
        sh: e3nn.IrrepsArray,
        radial_basis: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> e3nn.IrrepsArray:
        # follow equiformer_v2
        features_norm = separated_layer_norm (features)
        messages = self.linear_1(features_norm)[senders]
        messages = e3nn.tensor_product(messages, sh, filter_ir_out=self.output_irreps)
        radial_message = jax.vmap(self.radial_mlp)(radial_basis)
        messages = messages * radial_message

        messages_agg = e3nn.scatter_sum(
            messages, dst=receivers, output_size=features.shape[0]
        ) / jnp.sqrt(self.avg_n_neighbors)

        features = self.linear_2(messages_agg) + self.skip(species, features)

        # follow equiformer
        return e3nn.gate(
            features,
            even_act=jax.nn.silu,
            odd_act=jax.nn.sigmoid,
            even_gate_act=jax.nn.silu,
        )


class Nequip(nnx.Module):
    """NequIP model for predicting energies and forces.

    Neural Equivariant Interatomic Potential using E(3)-equivariant
    graph neural networks.

    Args:
        n_species: Number of atom species
        lmax: Maximum angular momentum for spherical harmonics
        cutoff: Radial cutoff distance
        hidden_irreps: Hidden feature dimension
        n_layers: Number of convolution layers
        radial_basis_size: Number of radial basis functions
        radial_mlp_size: Hidden size of radial MLP
        radial_mlp_layers: Number of radial MLP layers
        radial_polynomial_p: Polynomial cutoff order
        mlp_init_scale: Initialization scale for MLPs
        shift: Energy shift
        scale: Energy scale
        avg_n_neighbors: Average number of neighbors for normalization
        atom_energies: Isolated atom energies for each species
        rngs: Flax NNX random number generator
    """

    def __init__(
        self,
        n_species: int,
        lmax: int = 3,
        cutoff: float = 5.0,
        hidden_irreps: str = "64x0e + 64x1o + 64x2e",
        n_layers: int = 5,
        radial_basis_size: int = 8,
        radial_mlp_size: int = 64,
        radial_mlp_layers: int = 3,
        radial_polynomial_p: float = 2.0,
        mlp_init_scale: float = 4.0,
        shift: float = 0.0,
        scale: float = 1.0,
        avg_n_neighbors: float = 25.0,
        atom_energies: Optional[Sequence[float]] = None,
        l_train: bool = True,
        periodic: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        # Store static configuration
        self.lmax = lmax
        self.cutoff = cutoff
        self.periodic = periodic
        self.n_species = n_species
        self.radial_basis_size = radial_basis_size
        self.radial_polynomial_p = radial_polynomial_p
        self.l_train = l_train
        # Store trainable scale parameters
        self.shift = nnx.Param(jnp.array(shift))
        self.scale = nnx.Param(jnp.array(scale))
        self.avg_n_neighbors = nnx.Param(jnp.array(avg_n_neighbors))

        if atom_energies is not None:
            self.atom_energies = nnx.Param(jnp.array(atom_energies))
        else:
            self.atom_energies = nnx.Param(jnp.zeros(n_species, dtype=jnp.float32))

        hidden_irreps = e3nn.Irreps(hidden_irreps)
        input_irreps = hidden_irreps.filter(keep="0e")
        sh_irreps = e3nn.s2_irreps(lmax)
        

        self.species_embedding = e3nn_nnx_Linear(
            irreps_in=e3nn.Irreps(f"{n_species}x0e"),
            irreps_out=input_irreps,
            rngs=rngs,
        )

        layers = []
        for i in range(n_layers):
            layer = NequipConvolution(
                input_irreps=input_irreps if i == 0 else hidden_irreps,
                output_irreps=hidden_irreps,
                sh_irreps=sh_irreps,
                n_species=n_species,
                radial_basis_size=radial_basis_size,
                radial_mlp_size=radial_mlp_size,
                radial_mlp_layers=radial_mlp_layers,
                mlp_init_scale=mlp_init_scale,
                avg_n_neighbors=avg_n_neighbors,
                rngs=rngs,
            )
            layers.append(layer)

        self.layers = nnx.List(layers)
        #self.layers = layers

        hidden_irreps_2 = (hidden_size//2) * e3nn.s2_irreps(lmax)
        self.readout1 = e3nn_nnx_Linear(
            irreps_in=hidden_irreps,
            irreps_out=hidden_irreps_2,
            biases=True, # willow
            rngs=rngs,
        )
        self.readout = e3nn_nnx_Linear(
            irreps_in=hidden_irreps_2,
            irreps_out=e3nn.Irreps("0e"),
            biases=True, # willow
            rngs=rngs,
        )

    def node_energies(self,
                      positions: jax.Array,
                      cell: jax.Array,
                      cutoff: jax.Array,
                      data: jraph.GraphsTuple) -> jax.Array:
        """Compute per-node energies.

        Args:
            positions: Atomic positions of shape (n_atoms, 3)
            cell: (n_graph, 3, 3) for periodic, None for molecular
            cutoff: (n_graph,) for periodic, None for molecular
            data: Graph data structure

        Returns:
            Per-node energies of shape (n_atoms, 1)
        """
        # input features are one-hot encoded species
        features = e3nn.IrrepsArray(
            e3nn.Irreps(f"{self.n_species}x0e"),
            jax.nn.one_hot(data.nodes["species"], self.n_species),
        )
        features = self.species_embedding (features)

        if self.periodic:
            # --- Periodic (condensed phase) ---
            # Transform cell-shift vectors Sij into Cartesian displacements
            Sij = data.edges['Sij']
            num_edges = Sij.shape[0]
            Sij = jnp.einsum(
                'ei,eij->ej',
                Sij,
                jnp.repeat(
                    cell, # [n_graph, 3, 3]
                    data.n_edge, #[n_graph]
                    axis=0,
                    total_repeat_length=num_edges,
                ) # [n_edges, 3, 3]
            ) # [n_edges, 3]
            cutoff_per_edge = jnp.repeat(
                cutoff, # [n_graph,]
                data.n_edge, # [n_graph]
                axis=0,
                total_repeat_length=num_edges
            )
            r = positions[data.senders] - (positions[data.receivers] + Sij)
        else:
            # --- Molecular (non-periodic) ---
            # Direct displacement, no cell shifts
            r = positions[data.senders] - positions[data.receivers]
            num_edges = data.senders.shape[0]
            cutoff_per_edge = jnp.full((num_edges,), self.cutoff)

        cutoff_per_edge = jnp.where(cutoff_per_edge > 1.0, cutoff_per_edge, 1.0)
        cutoff_per_edge = cutoff_per_edge.squeeze() # [n_edges,]

        # safe norm (avoids nan for r = 0)
        square_r_norm = jnp.sum(r**2, axis=-1)
        r_norm = jnp.where(square_r_norm == 0.0, 0.0, jnp.sqrt(square_r_norm))

        radial_basis = (
            bessel_basis(r_norm, self.radial_basis_size, cutoff_per_edge)
            * polynomial_cutoff(
                r_norm,
                cutoff_per_edge,
                self.radial_polynomial_p,
            )[:, None]
        )

        # compute spherical harmonics of edge displacements
        sh = e3nn.spherical_harmonics(
            e3nn.s2_irreps(self.lmax),
            r,
            normalize=True,
            normalization="component",
        )

        for layer in self.layers:
            features = layer(
                features,
                data.nodes["species"],
                sh,
                radial_basis,
                data.senders,
                data.receivers,
            )

        features = self.readout1(features)
        node_energies = self.readout(features)

        # scale and shift energies
        node_energies = node_energies * (
            self.scale[...]
        ) + (self.shift[...])

        return node_energies.array

    def __call__(self, data: jraph.GraphsTuple) -> tuple[jax.Array, jax.Array]:
        """Compute energies and forces for a batch of graphs.

        Args:
            data: Graph data structure with nodes containing 'positions' and 'species'.
                  For periodic systems, globals must contain 'cell', 'cutoff', 'volume'.
                  For molecular systems, these are not required.

        Returns:
            Tuple of (graph_energies, forces, stress) where:
                - graph_energies: Total energy per graph, shape (n_graphs,)
                - forces: Forces on each atom, shape (n_atoms, 3)
                - stress: Stress tensor per graph, shape (n_graphs, 6)
                          (Voigt: xx, yy, zz, yz, xz, xy)
                          Zero for molecular systems.
        """
        n_graphs = data.n_node.shape[0]

        if self.periodic:
            # --- Periodic: differentiate w.r.t. positions AND cell ---
            def total_energy_fn(positions: jax.Array,
                                cell: jax.Array,
                                cutoff: jax.Array) -> tuple[jax.Array, jax.Array]:
                node_energies = self.node_energies(positions, cell, cutoff, data)
                return jnp.sum(node_energies), node_energies

            grad_fn = jax.grad(total_energy_fn, argnums=(0,1), has_aux=True)
            (grad, cellgrad), node_energies = \
                grad_fn(data.nodes["positions"],
                        data.globals["cell"],
                        data.globals['cutoff'])
        else:
            # --- Molecular: differentiate w.r.t. positions only ---
            def total_energy_fn(positions: jax.Array) -> tuple[jax.Array, jax.Array]:
                node_energies = self.node_energies(positions, None, None, data)
                return jnp.sum(node_energies), node_energies

            grad_fn = jax.grad(total_energy_fn, has_aux=True)
            grad, node_energies = grad_fn(data.nodes["positions"])

        if not self.l_train:
            # add isolated atom energies to each node as prior
            # to predict energies of testing dataset. [in which no energy/forces are provided]
            node_energies = node_energies + jax.lax.stop_gradient(
                self.atom_energies[...][data.nodes["species"], None]
            )

        # Get node mask for padded graphs
        # jraph.get_node_padding_mask treats the last graph as padding,
        # so for single graphs or non-padded batches, we need special handling
        if n_graphs > 1:
            # For padded batches, use jraph's mask (last graph is padding)
            node_mask = jraph.get_node_padding_mask(data)[:, None]
            grad = jnp.where(node_mask, grad, 0.0)
        # For single graphs, no masking needed (all nodes are valid)

        # Handle any NaN forces from numerical issues
        grad = jnp.where(jnp.isnan(grad), 0.0, grad)

        # compute total energies across each subgraph
        graph_energies = jraph.segment_sum(
            node_energies,
            node_graph_idx(data),
            num_segments=n_graphs,
            indices_are_sorted=True,
        )

        if self.periodic:
            # --- Stress for periodic systems ---
            cellgrad = jnp.where(jnp.isnan(cellgrad), 0.0, cellgrad)

            volume = data.globals["volume"]
            volume = jnp.where(volume > 0.0, volume, 1.0)
            stress_cell = (
                jnp.transpose(cellgrad, (0, 2, 1)) @ data.globals["cell"]
            )

            stress_grad = jnp.einsum("iu,iv->iuv", grad, data.nodes["positions"])
            stress_grad = jraph.segment_sum(
                stress_grad,
                node_graph_idx(data),
                num_segments=n_graphs,
                indices_are_sorted=True,
            )
            virial = (stress_cell + stress_grad).reshape(-1, 9)[:, [0, 4, 8, 5, 2, 1]]
            stress = virial / volume[:, None]
        else:
            # --- No stress for molecular systems ---
            stress = jnp.zeros((n_graphs, 6))

        return graph_energies[:, 0], -grad, stress


def node_graph_idx(data: jraph.GraphsTuple) -> jnp.ndarray:
    """Returns the index of the graph for each node."""
    n_graph = data.n_node.shape[0]
    sum_n_node = jax.tree_util.tree_leaves(data.nodes)[0].shape[0]
    graph_idx = jnp.arange(n_graph)
    node_gr_idx = jnp.repeat(
        graph_idx, data.n_node, axis=0, total_repeat_length=sum_n_node
    )
    return node_gr_idx




if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pickle
    #from ase.io import read

    print("=" * 70)
    print("Nequip-Style Equivariant Potential")
    print("=" * 70)

    """
    uniq_element = {z: i for i, z in enumerate([1, 6, 7, 8])}
    #enr_avg_per_element = {i: -654.09 for i in range(4)}
    enr_avg_per_element = {
        1: -13.587222780835477,
        6: -1029.4889999855063,
        7: -1484.9814568572233,
        8: -2041.9816003861047,
    }

    # Load data
    atoms = read('dataset_3BPA/train_300K.xyz', index=0)
    graphset = get_graphset(
        [atoms],
        cutoff=5.0,
        nbatch=1,
        uniq_element=uniq_element,
        enr_avg_per_element=enr_avg_per_element
    )
    """
    #data = graphset[0]
    with open('omat24_val_0.pkl', 'rb') as f:
        graphset = pickle.load (f)
    data = graphset[0]

    lines = open('atom_energies.dat', 'r').readlines()
    atom_energies = []
    for line in lines:
        key = line.split()
        atom_energies.append (float(key[1]))

    # Create model
    rngs = nnx.Rngs(42)
    model = Nequip(
        n_species=len(atom_energies),
        lmax=3,
        hidden_size=128,
        n_layers=3,
        radial_basis_size=8,
        radial_mlp_size=64,
        radial_mlp_layers=3,
        mlp_init_scale=4.0,
        shift = 0.0,
        scale = 1.0,
        avg_n_neighbors=25,
        atom_energies=atom_energies,
        rngs=rngs,
    )


    # Count parameters
    graphdef, state = nnx.split(model)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state))
    print(f"\nNumber of parameters: {n_params:,}")


    # Run forward pass
    energies, forces, stress = model(data)
    print(f"Energies shape: {energies.shape}")
    print(f"Forces shape: {forces.shape}")
    print(f"Total energy: {energies[0]:.6f}")
    print(f"Force magnitude: {np.linalg.norm(forces, axis=-1).mean():.6f}")
    print(f"stress", stress)
    print(f"Target energies", data.globals['energy'])
    print ('target stress', data.globals['stress'], 'eV/A^3')

