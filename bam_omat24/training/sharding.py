"""Shared sharding utilities for multi-GPU training."""

from typing import Any, Tuple
import pickle

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import jraph
import numpy as np


# =============================================================================
# Device Mesh Setup
# =============================================================================

def setup_mesh() -> Tuple[Mesh, int]:
    """Setup device mesh for data parallelism."""
    devices = jax.devices()
    n_devices = len(devices)
    print(f"Setting up mesh with {n_devices} device(s)")
    for i, d in enumerate(devices):
        print(f"  Device {i}: {d}")

    device_mesh = mesh_utils.create_device_mesh((n_devices,))
    mesh = Mesh(device_mesh, axis_names=('dp',))  # dp = data parallel
    return mesh, n_devices


# =============================================================================
# Replication Utilities
# =============================================================================

def replicate(x: Any, mesh: Mesh) -> Any:
    """Replicate array across all devices."""
    return jax.device_put(x, NamedSharding(mesh, P()))


def replicate_pytree(pytree: Any, mesh: Mesh) -> Any:
    """Replicate entire pytree across all devices."""
    return jax.tree.map(lambda x: replicate(x, mesh), pytree)


def unreplicate(x: Any) -> Any:
    """Get value from replicated array."""
    return np.asarray(x)


def unreplicate_pytree(pytree: Any) -> Any:
    """Unreplicate entire pytree."""
    return jax.tree.map(unreplicate, pytree)


# =============================================================================
# Batch Shape Utilities
# =============================================================================

def squeeze_batch(batch: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Squeeze the leading device dimension from a sharded batch.

    When using shard_map with P('dp') on the batch, each device receives
    a slice with shape (1, ...). This function removes that leading dimension.

    (1, n_nodes, ...) -> (n_nodes, ...)
    (1, n_edges, ...) -> (n_edges, ...)
    (1, n_graphs, ...) -> (n_graphs, ...)
    """
    def squeeze_array(x):
        if x is None:
            return None
        if hasattr(x, 'shape') and len(x.shape) > 0 and x.shape[0] == 1:
            return jnp.squeeze(x, axis=0)
        return x

    def squeeze_dict(d):
        if d is None:
            return None
        return {k: squeeze_array(v) for k, v in d.items()}

    return jraph.GraphsTuple(
        n_node=squeeze_array(batch.n_node),
        n_edge=squeeze_array(batch.n_edge),
        nodes=squeeze_dict(batch.nodes),
        edges=squeeze_dict(batch.edges),
        globals=squeeze_dict(batch.globals),
        senders=squeeze_array(batch.senders),
        receivers=squeeze_array(batch.receivers),
    )


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(state_dict: dict, path: str):
    """Save checkpoint."""
    with open(path, 'wb') as f:
        pickle.dump(state_dict, f)


def load_checkpoint(path: str) -> dict:
    """Load checkpoint."""
    with open(path, 'rb') as f:
        return pickle.load(f)
