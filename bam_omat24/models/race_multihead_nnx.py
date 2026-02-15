"""Multihead RACE model implementation using Flax NNX.

Extends the RACE model with per-head readouts for fine-tuning on
multiple datasets simultaneously, preventing catastrophic forgetting.

Architecture (MACE/BAM-torch style readout expansion):
- Each per-layer readout outputs num_heads scalars instead of 1.
- Per-layer outputs (conv readout + x_readout + f_readout) are summed across layers.
- Per-head scale/shift applied, then head_idx selects one energy per atom.
- Backbone (convolutions, embeddings, radial MLPs) is shared across heads.
"""

import pickle
from typing import Optional, Sequence

import e3nn_jax as e3nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jraph

from bam_omat24.models.nequip_nnx import (
    bessel_basis,
    polynomial_cutoff,
    separated_layer_norm,
    node_graph_idx,
)
from bam_omat24.models.linear_nnx import Linear as e3nn_nnx_Linear
from bam_omat24.models.race_nnx import RACEXReadout, RACEConvolution


class RACEConvolutionMultihead(RACEConvolution):
    """RACEConvolution with multihead readout.

    Overrides the final readout to output num_heads scalars instead of 1.
    All other layers (linear_1, linear_2, skip, radial_mlp, readout1) are unchanged.
    The __call__ method is inherited and works without modification.
    """

    def __init__(
        self,
        input_irreps: e3nn.Irreps,
        output_irreps: e3nn.Irreps,
        x_irreps: e3nn.Irreps,
        sh_irreps: e3nn.Irreps,
        n_species: int,
        radial_basis_size: int,
        radial_mlp_size: int,
        radial_mlp_layers: int,
        mlp_init_scale: float,
        avg_n_neighbors: float,
        num_heads: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            input_irreps=input_irreps,
            output_irreps=output_irreps,
            x_irreps=x_irreps,
            sh_irreps=sh_irreps,
            n_species=n_species,
            radial_basis_size=radial_basis_size,
            radial_mlp_size=radial_mlp_size,
            radial_mlp_layers=radial_mlp_layers,
            mlp_init_scale=mlp_init_scale,
            avg_n_neighbors=avg_n_neighbors,
            rngs=rngs,
        )
        # Override final readout: "0e" -> "num_heads x 0e"
        output_irreps_2 = output_irreps // 2
        self.readout = e3nn_nnx_Linear(
            irreps_in=output_irreps_2,
            irreps_out=e3nn.Irreps(f"{num_heads}x0e"),
            biases=True,
            rngs=rngs,
        )


class RACEMultihead(nnx.Module):
    """Multihead RACE model for multi-dataset fine-tuning.

    Uses MACE/BAM-torch style readout expansion: each per-layer readout
    (conv readout, x_readout, f_readout) outputs num_heads scalars.
    The backbone is shared across all heads.

    Args:
        n_species: Number of atom species.
        num_heads: Number of output heads (one per dataset).
        lmax: Maximum angular momentum for spherical harmonics.
        cutoff: Radial cutoff distance.
        hidden_irreps: Hidden feature irreps string.
        n_layers: Number of convolution layers.
        radial_basis_size: Number of radial basis functions.
        radial_mlp_size: Hidden size of radial MLP.
        radial_mlp_layers: Number of radial MLP layers.
        radial_polynomial_p: Polynomial cutoff order.
        mlp_init_scale: Initialization scale for MLPs.
        shift: Energy shift (scalar, broadcast to all heads).
        scale: Energy scale (scalar, broadcast to all heads).
        avg_n_neighbors: Average number of neighbors for normalization.
        atom_energies: Isolated atom energies for each species.
        rngs: Flax NNX random number generator.
    """

    def __init__(
        self,
        n_species: int,
        num_heads: int = 2,
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
        self.lmax = lmax
        self.cutoff = cutoff
        self.periodic = periodic
        self.n_species = n_species
        self.radial_basis_size = radial_basis_size
        self.radial_polynomial_p = radial_polynomial_p
        self.l_train = l_train
        self.num_heads = num_heads
        self.n_layers = n_layers

        # Per-head scale and shift parameters
        self.shifts = nnx.Param(jnp.full((num_heads,), shift))
        self.scales = nnx.Param(jnp.full((num_heads,), scale))
        self.avg_n_neighbors = nnx.Param(jnp.array(avg_n_neighbors))

        if atom_energies is not None:
            self.atom_energies = nnx.Param(jnp.array(atom_energies))
        else:
            self.atom_energies = nnx.Param(jnp.zeros(n_species, dtype=jnp.float32))

        hidden_irreps = e3nn.Irreps(hidden_irreps)
        input_irreps = hidden_irreps.filter(keep="0e")
        x_irreps = e3nn.Irreps("1x0e")
        self.x_linear = e3nn_nnx_Linear(
            irreps_in=input_irreps,
            irreps_out=x_irreps,
            linear_type="indexed",
            num_indexed_weights=n_species,
            force_irreps_out=True,
            rngs=rngs,
        )

        sh_irreps = e3nn.s2_irreps(lmax)

        self.species_embedding = e3nn_nnx_Linear(
            irreps_in=e3nn.Irreps(f"{n_species}x0e"),
            irreps_out=input_irreps,
            rngs=rngs,
        )

        # Multihead readout irreps
        multihead_out = e3nn.Irreps(f"{num_heads}x0e")

        layers = []
        x_readouts = []
        f_readouts = []
        for i in range(n_layers):
            # Convolution with multihead readout
            layer = RACEConvolutionMultihead(
                input_irreps=input_irreps if i == 0 else hidden_irreps,
                output_irreps=hidden_irreps,
                x_irreps=x_irreps,
                sh_irreps=sh_irreps,
                n_species=n_species,
                radial_basis_size=radial_basis_size,
                radial_mlp_size=radial_mlp_size,
                radial_mlp_layers=radial_mlp_layers,
                mlp_init_scale=mlp_init_scale,
                avg_n_neighbors=avg_n_neighbors,
                num_heads=num_heads,
                rngs=rngs,
            )
            layers.append(layer)

            # x_readout with multihead output
            x_readout = RACEXReadout(
                hidden_irreps=e3nn.Irreps("64x0e"),
                output_irreps=multihead_out,
                x_irreps=x_irreps,
                n_species=n_species,
                avg_n_neighbors=avg_n_neighbors,
                rngs=rngs,
            )
            x_readouts.append(x_readout)

            # f_readout with multihead output
            f_readout = RACEXReadout(
                hidden_irreps=e3nn.Irreps("64x0e"),
                output_irreps=multihead_out,
                x_irreps=input_irreps if i == 0 else hidden_irreps,
                n_species=n_species,
                avg_n_neighbors=avg_n_neighbors,
                rngs=rngs,
            )
            f_readouts.append(f_readout)

        self.layers = nnx.List(layers)
        self.x_readouts = nnx.List(x_readouts)
        self.f_readouts = nnx.List(f_readouts)

    def node_energies(
        self,
        positions: jax.Array,
        cell: jax.Array,
        cutoff: jax.Array,
        data: jraph.GraphsTuple,
        head_idx: jax.Array = None,
    ) -> jax.Array:
        """Compute per-node energies with head selection.

        Args:
            positions: Atomic positions (n_atoms, 3).
            cell: Unit cells (n_graph, 3, 3) or None.
            cutoff: Cutoff per graph (n_graph,) or None.
            data: Graph data structure.
            head_idx: Per-node head index (n_atoms,). If None, returns all heads.

        Returns:
            Per-node energies (n_atoms, 1) if head_idx given, else (n_atoms, num_heads).
        """
        # Species embedding (identical to RACE)
        features = e3nn.IrrepsArray(
            e3nn.Irreps(f"{self.n_species}x0e"),
            jax.nn.one_hot(data.nodes["species"], self.n_species),
        )
        features = self.species_embedding(features)
        x_feats = self.x_linear(data.nodes['species'], features)

        if self.periodic:
            Sij = data.edges['Sij']
            num_edges = Sij.shape[0]
            Sij = jnp.einsum(
                'ei,eij->ej',
                Sij,
                jnp.repeat(
                    cell,
                    data.n_edge,
                    axis=0,
                    total_repeat_length=num_edges,
                )
            )
            cutoff_per_edge = jnp.repeat(
                cutoff,
                data.n_edge,
                axis=0,
                total_repeat_length=num_edges,
            )
            r = positions[data.senders] - (positions[data.receivers] + Sij)
        else:
            r = positions[data.senders] - positions[data.receivers]
            num_edges = data.senders.shape[0]
            cutoff_per_edge = jnp.full((num_edges,), self.cutoff)

        cutoff_per_edge = jnp.where(cutoff_per_edge > 1.0, cutoff_per_edge, 1.0)
        cutoff_per_edge = cutoff_per_edge.squeeze()

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

        sh = e3nn.spherical_harmonics(
            e3nn.s2_irreps(self.lmax),
            r,
            normalize=True,
            normalization="component",
        )

        # Per-layer readouts: each outputs (n_atoms, num_heads)
        outputs = []
        for (layer, x_readout, f_readout) in zip(
            self.layers, self.x_readouts, self.f_readouts
        ):
            f_energy = f_readout(data.nodes["species"], features)
            node_energies, features = layer(
                features,
                x_feats,
                data.nodes["species"],
                sh,
                radial_basis,
                data.senders,
                data.receivers,
            )
            x_energy = x_readout(data.nodes["species"], x_feats)
            # Each term: IrrepsArray with shape (n_atoms, num_heads)
            outputs.append(
                node_energies.array + x_energy.array + f_energy.array
            )

        # Sum across layers: (n_atoms, n_layers, num_heads) -> (n_atoms, num_heads)
        node_energies = jnp.sum(jnp.stack(outputs, axis=1), axis=1)

        # Per-head scale and shift
        node_energies = node_energies * self.scales[...] + self.shifts[...]

        # Select per head_idx
        if head_idx is not None:
            n_atoms = node_energies.shape[0]
            selected = node_energies[jnp.arange(n_atoms), head_idx]
            return selected[:, None]  # (n_atoms, 1)
        else:
            return node_energies  # (n_atoms, num_heads)

    def __call__(
        self, data: jraph.GraphsTuple
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compute energies, forces, and stress for a batch of graphs.

        Expects data.globals["head"] of shape (n_graphs,) with per-graph head indices.

        Returns:
            Tuple of (graph_energies, forces, stress).
        """
        n_graphs = data.n_node.shape[0]

        # Expand graph-level head index to node-level
        sum_n_node = jax.tree_util.tree_leaves(data.nodes)[0].shape[0]
        graph_idx = jnp.arange(n_graphs)
        node_graph = jnp.repeat(
            graph_idx, data.n_node, axis=0, total_repeat_length=sum_n_node
        )
        head_idx = data.globals["head"][node_graph]  # (n_atoms,)

        if self.periodic:
            def total_energy_fn(
                positions: jax.Array, cell: jax.Array, cutoff: jax.Array
            ) -> tuple[jax.Array, jax.Array]:
                node_energies = self.node_energies(
                    positions, cell, cutoff, data, head_idx
                )
                return jnp.sum(node_energies), node_energies

            grad_fn = jax.grad(total_energy_fn, argnums=(0, 1), has_aux=True)
            (grad, cellgrad), node_energies = grad_fn(
                data.nodes["positions"],
                data.globals["cell"],
                data.globals["cutoff"],
            )
        else:
            def total_energy_fn(
                positions: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                node_energies = self.node_energies(
                    positions, None, None, data, head_idx
                )
                return jnp.sum(node_energies), node_energies

            grad_fn = jax.grad(total_energy_fn, has_aux=True)
            grad, node_energies = grad_fn(data.nodes["positions"])

        if not self.l_train:
            node_energies = node_energies + jax.lax.stop_gradient(
                self.atom_energies[...][data.nodes["species"], None]
            )

        if n_graphs > 1:
            node_mask = jraph.get_node_padding_mask(data)[:, None]
            grad = jnp.where(node_mask, grad, 0.0)
            node_energies = jnp.where(node_mask, node_energies, 0.0)

        grad = jnp.where(jnp.isnan(grad), 0.0, grad)

        graph_energies = jraph.segment_sum(
            node_energies,
            node_graph_idx(data),
            num_segments=n_graphs,
            indices_are_sorted=True,
        )

        if self.periodic:
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
            virial = (stress_cell + stress_grad).reshape(-1, 9)[
                :, [0, 4, 8, 5, 2, 1]
            ]
            stress = virial / volume[:, None]
        else:
            stress = jnp.zeros((n_graphs, 6))

        return graph_energies[:, 0], -grad, stress


def load_foundation_as_multihead(
    ckpt_path: str,
    num_heads: int,
    config: dict,
    rngs: nnx.Rngs,
) -> tuple:
    """Load a pre-trained RACE checkpoint and create a RACEMultihead model.

    Backbone parameters are copied directly from the foundation.
    Readout parameters (final readout, x_readout3) are repeated num_heads
    times along the output dimension (MACE/BAM-torch convention).
    The foundation's scalar shift/scale are broadcast to per-head shifts/scales.

    Args:
        ckpt_path: Path to foundation checkpoint (ckpt_best.pkl).
        num_heads: Number of output heads.
        config: Model config dict (lmax, hidden_irreps, n_layers, etc.).
        rngs: Random number generators.

    Returns:
        Tuple of (graphdef, params) for the multihead model.
    """
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)

    foundation_params = ckpt.get('ema_params', ckpt['params'])

    model = RACEMultihead(
        n_species=len(config["atom_energies"]),
        num_heads=num_heads,
        lmax=config["lmax"],
        hidden_irreps=config["hidden_irreps"],
        n_layers=config["n_layers"],
        radial_basis_size=config["radial_basis_size"],
        radial_mlp_size=config["radial_mlp_size"],
        radial_mlp_layers=config["radial_mlp_layers"],
        radial_polynomial_p=config["radial_polynomial_p"],
        mlp_init_scale=config["mlp_init_scale"],
        shift=0.0,
        scale=1.0,
        avg_n_neighbors=25.0,
        atom_energies=config["atom_energies"],
        l_train=True,
        periodic=True,
        rngs=rngs,
    )

    graphdef, multihead_params = nnx.split(model, nnx.Param)

    # Flatten both param trees for matching
    foundation_flat = jax.tree_util.tree_leaves_with_path(foundation_params)
    multihead_flat = jax.tree_util.tree_leaves_with_path(multihead_params)

    def path_to_str(path):
        parts = []
        for p in path:
            if hasattr(p, 'key'):
                parts.append(str(p.key))
            elif hasattr(p, 'idx'):
                parts.append(str(p.idx))
            else:
                parts.append(str(p))
        return '/'.join(parts)

    def normalize_readout_path(path_str):
        """Replace multihead output irreps with single-head for matching.

        Weight names contain irreps like 'weight_w_0_0_32x0e_2x0e' for
        num_heads=2, vs foundation's 'weight_w_0_0_32x0e_1x0e'.
        Replace the last occurrence of _{num_heads}x0e with _1x0e.
        """
        target = f'_{num_heads}x0e'
        idx = path_str.rfind(target)
        if idx >= 0:
            return path_str[:idx] + '_1x0e' + path_str[idx + len(target):]
        return path_str

    foundation_lookup = {
        path_to_str(path): value for path, value in foundation_flat
    }

    new_leaves = []
    copied = 0
    repeated = 0
    skipped = 0

    for path, value in multihead_flat:
        key_str = path_to_str(path)

        # Handle per-head shifts/scales from foundation's scalar shift/scale
        if 'shifts' in key_str or 'scales' in key_str:
            if 'shifts' in key_str:
                foundation_key = key_str.replace('shifts', 'shift')
            else:
                foundation_key = key_str.replace('scales', 'scale')

            if foundation_key in foundation_lookup:
                foundation_val = foundation_lookup[foundation_key]
                new_leaves.append(
                    jnp.full((num_heads,), float(foundation_val))
                )
                copied += 1
            else:
                new_leaves.append(value)
                skipped += 1

        elif key_str in foundation_lookup:
            # Direct match (backbone params)
            foundation_val = foundation_lookup[key_str]
            if foundation_val.shape == value.shape:
                new_leaves.append(foundation_val)
                copied += 1
            else:
                print(f"Shape mismatch for {key_str}: "
                      f"foundation {foundation_val.shape} vs multihead {value.shape}")
                new_leaves.append(value)
                skipped += 1

        else:
            # Try normalized path (for readout params with different irreps names)
            normalized_key = normalize_readout_path(key_str)
            if normalized_key in foundation_lookup:
                foundation_val = foundation_lookup[normalized_key]
                # Repeat foundation weights along output dim for multihead
                new_val = jnp.repeat(foundation_val, num_heads, axis=-1)
                if new_val.shape == value.shape:
                    new_leaves.append(new_val)
                    repeated += 1
                else:
                    print(f"Shape mismatch after repeat for {key_str}: "
                          f"got {new_val.shape} vs expected {value.shape}")
                    new_leaves.append(value)
                    skipped += 1
            else:
                print(f"No foundation match for: {key_str}")
                new_leaves.append(value)
                skipped += 1

    multihead_params = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(multihead_params),
        new_leaves,
    )

    print(f"Foundation loading: copied {copied}, "
          f"repeated {repeated} (readout expansion), "
          f"skipped {skipped}")

    return graphdef, multihead_params


if __name__ == "__main__":
    import numpy as np
    from bam_omat24.data.atom_energies import ATOM_ENERGIES

    print("=" * 70)
    print("Multihead RACE Equivariance Test (Readout Expansion)")
    print("=" * 70)

    with open('omat24_val_0.pkl', 'rb') as f:
        graphset = pickle.load(f)
    data = graphset[0]

    # Inject head index into globals
    new_globals = dict(data.globals)
    new_globals["head"] = np.array([0], dtype=np.int32)
    data = data._replace(globals=new_globals)

    rngs = nnx.Rngs(42)
    model = RACEMultihead(
        n_species=len(ATOM_ENERGIES),
        num_heads=2,
        lmax=3,
        hidden_irreps="64x0e + 64x1o + 64x2e",
        n_layers=3,
        radial_basis_size=8,
        radial_mlp_size=64,
        radial_mlp_layers=3,
        mlp_init_scale=4.0,
        shift=0.0,
        scale=1.0,
        avg_n_neighbors=25,
        atom_energies=ATOM_ENERGIES,
        l_train=True,
        periodic=True,
        rngs=rngs,
    )

    # Count parameters
    graphdef, state = nnx.split(model)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state))
    print(f"\nNumber of parameters: {n_params:,}")

    # Forward pass
    energies, forces, stress = model(data)
    print(f"Energies shape: {energies.shape}")
    print(f"Forces shape: {forces.shape}")
    print(f"Total energy: {energies[0]:.6f}")
    print(f"Force magnitude: {np.linalg.norm(forces, axis=-1).mean():.6f}")
    print(f"Stress: {stress}")

    # ---- Equivariance test ----
    print("\n" + "=" * 70)
    print("Equivariance Test: Random Rotation")
    print("=" * 70)

    # Random rotation matrix
    key = jax.random.PRNGKey(123)
    q, _ = jnp.linalg.qr(jax.random.normal(key, (3, 3)))
    # Ensure proper rotation (det=+1)
    q = q * jnp.linalg.det(q)

    # Rotate positions and cell
    rotated_positions = data.nodes["positions"] @ q.T
    rotated_cell = data.globals["cell"] @ q.T

    rotated_nodes = dict(data.nodes)
    rotated_nodes["positions"] = rotated_positions
    rotated_globals = dict(data.globals)
    rotated_globals["cell"] = rotated_cell
    data_rot = data._replace(nodes=rotated_nodes, globals=rotated_globals)

    energies_rot, forces_rot, stress_rot = model(data_rot)

    print(f"Energy difference (should be ~0): "
          f"{jnp.abs(energies[0] - energies_rot[0]):.2e}")
    print(f"Force rotation error (should be ~0): "
          f"{jnp.max(jnp.abs(forces @ q.T - forces_rot)):.2e}")
