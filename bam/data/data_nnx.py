import pickle
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Iterator, Optional, Literal
from dataclasses import dataclass

import ase
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import jraph
import matscipy.neighbours
import numpy as np
from flax import nnx
from tqdm import tqdm

from bam.data.atom_energies import ATOMIC_NUMBER_TO_INDEX


def atoms_to_graph(
    atoms: ase.Atoms,
    cutoff: float = 6.0,
    self_interaction: bool = False,
) -> jraph.GraphsTuple:
    """Convert an ASE Atoms object to a jraph.GraphsTuple for model inference.

    Args:
        atoms: ASE Atoms object (periodic or non-periodic).
        cutoff: Cutoff radius for neighbor list computation.
        self_interaction: If True, include self-loops. Default False.

    Returns:
        jraph.GraphsTuple ready for model evaluation.
    """
    src, dst, Sij = matscipy.neighbours.neighbour_list("ijS", atoms, cutoff)

    if not self_interaction:
        mask = src != dst
        src, dst, Sij = src[mask], dst[mask], Sij[mask]

    species = np.array(
        [ATOMIC_NUMBER_TO_INDEX[z] for z in atoms.get_atomic_numbers()],
    ).astype(np.int32)

    cell = np.array([atoms.get_cell()[:]], dtype=np.float32)
    volume = np.array([atoms.get_volume()], dtype=np.float32)

    n_atoms = len(atoms)

    return jraph.GraphsTuple(
        n_node=np.array([n_atoms], dtype=np.int32),
        n_edge=np.array([len(src)], dtype=np.int32),
        nodes={
            "species": species,
            "positions": atoms.positions.astype(np.float32),
            "forces": np.zeros((n_atoms, 3), dtype=np.float32),
        },
        edges={"Sij": Sij.astype(np.float32)},
        senders=dst.astype(np.int32),
        receivers=src.astype(np.int32),
        globals={
            "energy": np.zeros(1, dtype=np.float32),
            "cell": cell,
            "volume": volume,
            "stress": np.zeros((1, 6), dtype=np.float32),
            "cutoff": np.array([cutoff], dtype=np.float32),
        },
    )


def atoms_to_graph_with_targets(
    atoms: ase.Atoms,
    cutoff: float,
    atom_energies: np.ndarray,
    atom_indices: dict,
) -> Optional[jraph.GraphsTuple]:
    """Convert an ASE Atoms object to a jraph.GraphsTuple with training targets.

    Same logic as build_pkl_v2.preprocess_graph: computes neighbor list,
    subtracts isolated atom energies, extracts forces/stress/cell.

    Args:
        atoms: ASE Atoms object with energy, forces, stress in calculator.
        cutoff: Cutoff radius for neighbor list computation.
        atom_energies: Array of per-species reference energies (indexed by species index).
        atom_indices: Dict mapping atomic number -> species index.

    Returns:
        jraph.GraphsTuple with targets, or None if no neighbors found.
    """
    src, dst, Sij = matscipy.neighbours.neighbour_list("ijS", atoms, cutoff)
    if len(src) < 20:
        cutoff = cutoff + 3.0
        src, dst, Sij = matscipy.neighbours.neighbour_list("ijS", atoms, cutoff)
        if len(src) == 0:
            return None

    species = np.array(
        [atom_indices[n] for n in atoms.get_atomic_numbers()],
    ).astype(np.int32)

    # Subtract isolated atom energies
    graph_e0 = 0.0
    if atom_energies is not None:
        node_e0 = atom_energies[species]
        graph_e0 = node_e0.sum()

    energy = np.array([atoms.get_potential_energy() - graph_e0], dtype=np.float32)
    forces = np.array(atoms.get_forces(), dtype=np.float32, copy=True)
    cell = np.array(atoms.get_cell().array, dtype=np.float32)[np.newaxis]  # (1, 3, 3)
    volume = np.array([atoms.get_volume()], dtype=np.float32)
    try:
        stress = np.array(atoms.get_stress(), dtype=np.float32, copy=True)[np.newaxis]  # (1, 6)
    except Exception:
        stress = np.zeros((1, 6), dtype=np.float32)

    return jraph.GraphsTuple(
        n_node=np.array([len(atoms)], dtype=np.int32),
        n_edge=np.array([len(src)], dtype=np.int32),
        nodes={
            "species": species,
            "positions": atoms.positions.astype(np.float32),
            "forces": forces,
        },
        edges={"Sij": Sij.astype(np.float32)},
        senders=dst.astype(np.int32),
        receivers=src.astype(np.int32),
        globals={
            "energy": energy,
            "cell": cell,
            "volume": volume,
            "stress": stress,
            "cutoff": np.array([cutoff], dtype=np.float32),
        },
    )


def create_size_buckets(
    graphs: List[jraph.GraphsTuple],
    n_buckets: int = 8
) -> List[List[int]]:
    """Group graph indices into buckets by size.

    Graphs in the same bucket have similar sizes, reducing padding waste.
    """
    sizes = np.array([int(g.n_node[0]) for g in graphs])
        # graphs = [GraphsTuple_0, GraphsTuple_1, ..., GraphsTuple_99999]
        # g.n_node[0] = 이 분자의 원자 수
        # sizes = [3, 150, 5, 148, 12, 87, 4, 200, ...]
        #         ↑  ↑    ↑   ↑    ↑   ↑
        #         분자별 원자 수 배열 (10만개)

    # Use quantiles for bucket boundaries
    percentiles = np.linspace(0, 100, n_buckets + 1) # = [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
    boundaries = np.percentile(sizes, percentiles)
        # np.percentile(sizes, 0)     # 하위 0% = 최솟값 = 3
        # np.percentile(sizes, 50)    # 하위 50% = 중앙값 = 25
        # np.percentile(sizes, 100)   # 하위 100% = 최댓값 = 200

        # "하위 0%는 몇이냐?"   → 3
        # "하위 12.5%는 몇이냐?" → 5
        # "하위 25%는 몇이냐?"   → 7
        # "하위 37.5%는 몇이냐?" → 10
        # "하위 50%는 몇이냐?"   → 25    (중앙값)
        # "하위 62.5%는 몇이냐?" → 50
        # "하위 75%는 몇이냐?"   → 80
        # "하위 87.5%는 몇이냐?" → 150
        # "하위 100%는 몇이냐?"  → 200

    buckets = [[] for _ in range(n_buckets)]
    for i, size in enumerate(sizes):
        # Find which bucket this graph belongs to
        bucket_idx = min(
            np.searchsorted(boundaries[1:], size, side='right'),
            n_buckets - 1
        )
        buckets[bucket_idx].append(i)

    return buckets


def nearest_bucket(x: int, bucket_size: int = 32) -> int:
    """Round up to nearest bucket boundary.

    More memory-efficient than power-of-2 for intermediate sizes.
    E.g., with bucket_size=32: 65 -> 96, 100 -> 128
    """
    return ((x + bucket_size - 1) // bucket_size) * bucket_size


def default_collate_fn(
    graphs: List[jraph.GraphsTuple],
    batch_size: int,
    node_bucket_size: int = 128,
    edge_bucket_size: int = 4096
) -> jraph.GraphsTuple:

    if len(graphs) == 0:
        raise ValueError("Cannot collate empty list of graphs")

    # Batch graphs
    batched = jraph.batch_np(graphs)

    # Compute current sizes
    n_nodes = int(np.sum(batched.n_node))
    n_edges = int(np.sum(batched.n_edge))

    pad_nodes = nearest_bucket(n_nodes + 1, node_bucket_size)
    pad_edges = nearest_bucket(n_edges + 1, edge_bucket_size)
    pad_graphs = batch_size + 1

    return jraph.pad_with_graphs(batched, pad_nodes, pad_edges, pad_graphs)


# =============================================================================
# Dynamic Batching (Size-Aware)
# =============================================================================

def compute_batch_stats(graphs: List[jraph.GraphsTuple]) -> dict:
    """Compute statistics about graph sizes for optimal batching."""
    n_nodes = [int(g.n_node[0]) for g in graphs]
    n_edges = [int(g.n_edge[0]) for g in graphs]

    return {
        'count': len(graphs),
        'nodes': {
            'min': min(n_nodes),
            'max': max(n_nodes),
            'mean': np.mean(n_nodes),
            'std': np.std(n_nodes),
            'total': sum(n_nodes),
        },
        'edges': {
            'min': min(n_edges),
            'max': max(n_edges),
            'mean': np.mean(n_edges),
            'std': np.std(n_edges),
            'total': sum(n_edges),
        }
    }


# =============================================================================
# Dataset Class
# =============================================================================

class Dataset(nnx.Object):
    """NNX-based dataset that reads xyz files and returns jraph.GraphsTuple.

    Supports distributed loading across multiple GPUs/processes.
    """

    def __init__(
        self,
        file_path: str = None,
        process_id: int = 0,
        n_processes: int = 1,
        cutoff: float = 6.0,
        atom_energies: np.ndarray = None,
        atom_indices: dict = None,
    ):
        """Initialize dataset with optional distributed loading.

        Args:
            file_path: Path to pickle file or trajectory file (.traj, .xyz, etc.).
            process_id: Current process/node ID (0 to n_processes-1).
            n_processes: Total number of processes/nodes.
            cutoff: Cutoff radius (used only for traj/xyz files).
            atom_energies: Per-species reference energies (used only for traj/xyz files).
            atom_indices: Atomic number to species index mapping (used only for traj/xyz files).

        Example with 1,000,000 graphs and 2 processes:
            Process 0: graphs[0:500,000]
            Process 1: graphs[500,000:1,000,000]
        """
        ext = Path(file_path).suffix.lower()
        if ext == '.pkl':
            with open(file_path, "rb") as f:
                all_graphs = pickle.load(f)
        else:
            # traj, xyz, extxyz, etc. — read with ASE and convert to graphs
            import ase.io
            atoms_list = ase.io.read(file_path, index=':')
            if atom_indices is None:
                atom_indices = ATOMIC_NUMBER_TO_INDEX
            all_graphs = []
            for atoms in tqdm(atoms_list, desc=f"Converting {Path(file_path).name}"):
                g = atoms_to_graph_with_targets(
                    atoms, cutoff=cutoff,
                    atom_energies=atom_energies, atom_indices=atom_indices,
                )
                if g is not None:
                    all_graphs.append(g)

        total_graphs = len(all_graphs)

        if n_processes > 1:
            # Calculate this process's portion
            graphs_per_process = total_graphs // n_processes
            remainder = total_graphs % n_processes

            start_idx = process_id * graphs_per_process
            end_idx = start_idx + graphs_per_process

            # Last process gets the remainder
            if process_id == n_processes - 1:
                end_idx += remainder

            self.graphs = all_graphs[start_idx:end_idx]
            self.global_start_idx = start_idx  # Store global offset for this process

            print(f"[Process {process_id}/{n_processes}] Loaded {len(self.graphs)}/{total_graphs} graphs "
                  f"(indices {start_idx}~{end_idx-1})", flush=True)
        else:
            self.graphs = all_graphs
            self.global_start_idx = 0  # No offset for single process

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> jraph.GraphsTuple:
        return self.graphs[idx]

    def get_stats(self) -> dict:
        """Get statistics about the dataset."""
        return compute_batch_stats(self.graphs)


# =============================================================================
# Advanced: Bucketed DataLoader for Minimum Padding
# =============================================================================

class BucketedDataLoader(nnx.Object): # 비슷한 원자 수(노드 수)의 분자끼리 묶어서 배치를 만든 것입니다.
    """DataLoader that groups graphs by size into buckets for minimal padding.

    This loader:
    1. Groups graphs into size buckets
    2. Samples batches from within buckets
    3. Results in much less padding waste than random batching
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 10,
        n_buckets: int = 8,
        shuffle: bool = True,
        drop_last: bool = False,
        rngs: nnx.Rngs | None = None,
        max_nodes_per_batch=256,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_buckets = n_buckets
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = default_collate_fn
        self.max_node_per_batch = max_nodes_per_batch

        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs

        # Create size buckets
        self.buckets = create_size_buckets(dataset.graphs, n_buckets) # 버켓 수에 따라 그래프를 분할
        self.bucket_sizes = [len(b) for b in self.buckets] # 버켓별 그래프 수

        # Calculate total batches
        self.n_batches = sum(
            size // batch_size if drop_last else (size + batch_size - 1) // batch_size
            for size in self.bucket_sizes if size > 0
        )

        self._reset()

    def _reset(self): # 에폭(epoch) 시작할 때 데이터 순서를 초기화하는 함수 (train 때는 순서를 랜덤하게,test 때는 순서대로)
        """Reset for new epoch."""
        self.current_bucket = 0
        self.bucket_idx = 0

        # Shuffle within each bucket
        if self.shuffle:
            key = self.rngs.params()
            self.bucket_orders = []
            for bucket in self.buckets:
                if len(bucket) > 0:
                    key, subkey = jax.random.split(key)
                    perm = jax.random.permutation(subkey, len(bucket))
                    self.bucket_orders.append([bucket[i] for i in np.asarray(perm)])
                else:
                    self.bucket_orders.append([])
        else:
            self.bucket_orders = [list(b) for b in self.buckets]

    def __iter__(self):
        self._reset()
        return self

    def __next__(self) -> Tuple[jraph.GraphsTuple, List[int]]:
        """Returns (batch, global_indices) where global_indices are the actual graph indices."""
        # Find next non-empty bucket with remaining samples
        while self.current_bucket < self.n_buckets:
            bucket = self.bucket_orders[self.current_bucket]
            remaining = len(bucket) - self.bucket_idx

            if remaining <= 0:
                self.current_bucket += 1
                self.bucket_idx = 0
                continue

            if self.drop_last and remaining < self.batch_size:
                self.current_bucket += 1
                self.bucket_idx = 0
                continue

            # Collect batch from current bucket
            n_to_take = min(self.batch_size, remaining)
            graphs = []
            graph_indices = []  # Track actual graph indices
            n_nodes = 0
            curr_idx = self.bucket_idx

            # Get global offset from dataset
            global_offset = getattr(self.dataset, 'global_start_idx', 0)

            for _ in range(n_to_take):
                local_idx = bucket[curr_idx]  # Index within this process's portion
                curr_graph = self.dataset.graphs[local_idx]
                n_nodes += curr_graph.n_node[0]
                if n_nodes >= self.max_node_per_batch:
                    break
                curr_idx += 1
                graphs.append(curr_graph)
                # Store global index (local_idx + global_offset)
                graph_indices.append(global_offset + local_idx)

            self.bucket_idx = curr_idx

            return self.collate_fn(graphs, self.batch_size), graph_indices

        raise StopIteration

    def __len__(self) -> int:
        return self.n_batches


# =============================================================================
# Multi-Device Batch Info Dataclass
# =============================================================================

@dataclass
class MultiDeviceBatchInfo:
    """Information about a multi-device batch."""
    n_devices: int
    valid_device_count: int
    n_graphs_per_device: List[int]
    n_nodes_per_device: List[int]
    n_edges_per_device: List[int]
    padded_n_nodes: int
    padded_n_edges: int
    padded_n_graphs: int
    graph_indices_per_device: List[List[int]] = None  # Actual global graph indices per device

    def __repr__(self):
        return (
            f"MultiDeviceBatchInfo(\n"
            f"  valid_devices={self.valid_device_count}/{self.n_devices}\n"
            f"  graphs_per_device={self.n_graphs_per_device}\n"
            f"  nodes_per_device={self.n_nodes_per_device}\n"
            f"  edges_per_device={self.n_edges_per_device}\n"
            f"  padded_shape=(nodes={self.padded_n_nodes}, edges={self.padded_n_edges}, graphs={self.padded_n_graphs})\n"
            f"  efficiency={100 * sum(self.n_nodes_per_device) / (self.padded_n_nodes * self.n_devices):.1f}%\n"
            f")"
        )



def stack_batches_for_devices(
    batches: List[jraph.GraphsTuple],
    mesh: Mesh,
    graph_indices: List[List[int]] = None,
) -> Tuple[jraph.GraphsTuple, MultiDeviceBatchInfo]:
    """Stack multiple batches and shard across devices.

    Use this after collecting batches from BucketedDataLoader:

        batches = [next(loader) for _ in range(n_devices)]
        sharded_batch, info = stack_batches_for_devices(batches, mesh)

    Args:
        batches: List of GraphsTuple from BucketedDataLoader
        mesh: JAX device mesh for sharding

    Returns:
        Tuple of (sharded_batch, batch_info)
    """
    n_devices = len(batches)

    # Helper to get actual array sizes
    def get_node_size(b):
        if isinstance(b.nodes, dict):
            first_key = next(iter(b.nodes.keys()))
            return b.nodes[first_key].shape[0]
        return b.nodes.shape[0]

    def get_edge_size(b):
        return b.senders.shape[0]

    # Collect stats before padding
    stats = {
        'n_graphs': [],
        'n_nodes': [],
        'n_edges': [],
    }
    for b in batches:
        # Real counts (not padded)
        real_graphs = jraph.get_graph_padding_mask(b).sum()
        real_nodes = jraph.get_node_padding_mask(b).sum()
        real_edges = jraph.get_edge_padding_mask(b).sum()
        stats['n_graphs'].append(real_graphs)
        stats['n_nodes'].append(real_nodes)
        stats['n_edges'].append(real_edges)
    # Find max actual array sizes
    node_sizes = [get_node_size(b) for b in batches]
    edge_sizes = [get_edge_size(b) for b in batches]
    graph_sizes = [b.n_node.shape[0] for b in batches]

    max_nodes = max(node_sizes)
    max_edges = max(edge_sizes)
    max_graphs = max(graph_sizes)

    # Pad all batches to same size
    padded_batches = []
    for b, n_nodes, n_edges, n_graphs in zip(batches, node_sizes, edge_sizes, graph_sizes):
        if n_nodes < max_nodes or n_edges < max_edges or n_graphs < max_graphs:
            ub = jraph.unpad_with_graphs (b)
            b = jraph.pad_with_graphs(ub, max_nodes, max_edges, max_graphs)
        padded_batches.append(b)
    # Stack along device dimension
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *padded_batches)

    # For multi-host: each host only puts data on its LOCAL devices
    # Then combine into a global sharded array
    n_hosts = jax.process_count()
    local_devices = jax.local_devices()
    n_local = len(local_devices)

    if n_hosts > 1:
        # Multi-host: use make_array_from_single_device_arrays
        sharding = NamedSharding(mesh, P('dp'))

        def shard_for_multihost(arr):
            # arr shape: (n_local_devices, ...)
            # Each shard needs shape (1, ...) to match PartitionSpec('dp')
            # Put each slice on its corresponding local device
            local_arrays = [
                jax.device_put(arr[i:i+1], local_devices[i])  # Keep leading dim: (1, ...)
                for i in range(n_local)
            ]
            # Create global array from local device arrays
            # Global shape: (n_total_devices, ...)
            global_shape = (jax.device_count(),) + arr.shape[1:]
            return jax.make_array_from_single_device_arrays(
                global_shape, sharding, local_arrays
            )

        sharded = jax.tree.map(shard_for_multihost, stacked)
    else:
        # Single-host: use simple device_put
        sharding = NamedSharding(mesh, P('dp'))
        sharded = jax.tree.map(lambda x: jax.device_put(x, sharding), stacked)

    info = MultiDeviceBatchInfo(
        n_devices=n_devices,
        valid_device_count=n_devices,  # All valid in this approach
        n_graphs_per_device=stats['n_graphs'],
        n_nodes_per_device=stats['n_nodes'],
        n_edges_per_device=stats['n_edges'],
        padded_n_nodes=max_nodes,
        padded_n_edges=max_edges,
        padded_n_graphs=max_graphs,
        graph_indices_per_device=graph_indices,  # Actual global graph indices
    )

    return sharded, info


def multi_device_iterator(
    loader,  # BucketedDataLoader
    n_devices: int,
    mesh: Mesh,
    drop_incomplete: bool = True,
) -> Iterator[Tuple[jraph.GraphsTuple, MultiDeviceBatchInfo]]:
    """Wrap a BucketedDataLoader to yield multi-device batches.

    Example:
        loader = BucketedDataLoader(dataset, batch_size=20)
        for batch, info in multi_device_iterator(loader, n_devices, mesh):
            loss = train_step(batch)

    Args:
        loader: BucketedDataLoader instance
        n_devices: Number of devices (total or local depending on multi-host)
        mesh: JAX device mesh
        drop_incomplete: If True, skip final batch if < n_devices batches available

    Yields:
        Tuple of (sharded_batch, batch_info)
    """
    # In multi-host mode, each host should only collect batches for its LOCAL devices
    n_hosts = jax.process_count()
    n_local = jax.local_device_count()

    # Number of batches to collect per iteration on this host
    if n_hosts > 1:
        batches_per_host = n_local  # Each host collects for its own devices
    else:
        batches_per_host = n_devices

    batches = []
    all_indices = []  # Collect graph indices for each device

    for batch_data in loader:
        # BucketedDataLoader now returns (batch, indices)
        if isinstance(batch_data, tuple) and len(batch_data) == 2:
            batch, indices = batch_data
        else:
            batch = batch_data
            indices = []

        batches.append(batch)
        all_indices.append(indices)

        if len(batches) == batches_per_host:
            yield stack_batches_for_devices(batches, mesh, all_indices)
            batches = []
            all_indices = []

    # Handle remaining batches
    if batches and not drop_incomplete:
        # Pad with duplicates of last batch
        valid_count = len(batches)
        while len(batches) < batches_per_host:
            batches.append(batches[-1])
            all_indices.append(all_indices[-1] if all_indices else [])

        sharded, info = stack_batches_for_devices(batches, mesh, all_indices)
        # Adjust valid count
        info.valid_device_count = valid_count
        yield sharded, info


class MultiDeviceDataLoader(nnx.Object):
    """Wrapper that converts any single-device loader to multi-device.

    This wraps BucketedDataLoader (or any iterator) and collects
    n_devices batches before stacking and sharding.

    In multi-host mode, each host collects only local device batches,
    then combines them into a global sharded array.

    Example:
        base_loader = BucketedDataLoader(dataset, batch_size=20)
        loader = MultiDeviceDataLoader(base_loader, n_devices, mesh)

        for batch, info in loader:
            loss = train_step(batch)
    """

    def __init__(
        self,
        base_loader,
        n_devices: int,
        mesh: Mesh,
        drop_incomplete: bool = True,
    ):
        self.base_loader = base_loader
        self.n_devices = n_devices
        self.mesh = mesh
        self.drop_incomplete = drop_incomplete

        # In multi-host mode, each host collects batches for local devices only
        n_hosts = jax.process_count()
        n_local = jax.local_device_count()
        self.batches_per_host = n_local if n_hosts > 1 else n_devices

        # Estimate number of multi-device batches
        if hasattr(base_loader, '__len__'):
            self.n_batches = len(base_loader) // self.batches_per_host
        else:
            self.n_batches = None

    def __iter__(self):
        return multi_device_iterator(
            iter(self.base_loader),
            self.n_devices,
            self.mesh,
            self.drop_incomplete,
        )

    def __len__(self):
        if self.n_batches is not None:
            return self.n_batches
        raise TypeError("Base loader doesn't support len()")




def print_bucket_batch_info(bucket_batch_info: dict,
                            title: str = "Bucket Batch Statistics"):
    """Pretty print bucket batch statistics."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"\n{'Unique (node_pad, edge_pad) shapes:':<40} {len(bucket_batch_info)}")
    print(f"\n{'Shape (nodes, edges)':<25} {'Count':<10} {'Percentage':<10}")
    print(f"{'-'*45}")

    total_batches = sum(bucket_batch_info.values())
    sorted_items = sorted(bucket_batch_info.items(), key=lambda x: -x[1])

    for (n_nodes, n_edges), count in sorted_items:
        pct = 100 * count / total_batches
        print(f"({n_nodes:>6}, {n_edges:>6})       {count:<10} {pct:>6.1f}%")

    print(f"{'-'*45}")
    print(f"{'Total batches:':<25} {total_batches}")
    print(f"{'='*60}\n")


# =============================================================================
# Test / Example
# =============================================================================

if __name__ == "__main__":
    from natsort import natsorted
    from bam.training.sharding import setup_mesh

    mesh, n_devices = setup_mesh()

    train_path = Path('/home/willow/Python/BAM-nequip-main/bam')
    train_files_pkl = natsorted(list(train_path.glob('*.pkl')))

    rngs = nnx.Rngs(42)
    batch_size = 20

    # Compare padding strategies
    print("Comparing padding strategies...")
    train_dataset = Dataset(file_path=train_files_pkl[0])

    # Print dataset stats
    stats = train_dataset.get_stats()
    print(f"\nDataset Statistics:")
    print(f"  Total graphs: {stats['count']}")
    print(f"  Nodes: min={stats['nodes']['min']}, max={stats['nodes']['max']}, "
          f"mean={stats['nodes']['mean']:.1f}, std={stats['nodes']['std']:.1f}")
    print(f"  Edges: min={stats['edges']['min']}, max={stats['edges']['max']}, "
          f"mean={stats['edges']['mean']:.1f}, std={stats['edges']['std']:.1f}")

    # Test MultiDeviceBucketedDataLoader
    print("\n\nTesting BucketedDataLoader...")
    base_loader = BucketedDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        n_buckets=8,
        shuffle=True,
        drop_last=True,
        rngs=rngs,
        max_nodes_per_batch=256,
    )

    loader = MultiDeviceDataLoader( base_loader,
                                    n_devices=n_devices,
                                    mesh=mesh)

    bucket_batch_info = {}

    total_real_nodes = 0
    total_padded_nodes = 0

    print("\nIterating through batches...")
    for batch_idx, (batch, info) in enumerate(tqdm(loader)):

        # Track padding statistics
        key = (info.padded_n_nodes, info.padded_n_edges)
        bucket_batch_info[key] = bucket_batch_info.get(key, 0) + 1

        # Calculate padding efficiency
        real_nodes = sum(info.n_nodes_per_device)
        padded_nodes = info.padded_n_nodes * n_devices
        total_real_nodes += real_nodes
        total_padded_nodes += padded_nodes

        # Print detailed info for first few batches
        if batch_idx < 5:
            print(f"\nBatch {batch_idx}:")
            print(info)
            # Handle both dict and array node features
            if isinstance(batch.nodes, dict):
                first_key = next(iter(batch.nodes.keys()))
                node_shape = batch.nodes[first_key].shape
                print(f"  Batch shape: n_node={batch.n_node.shape}, nodes['{first_key}']={node_shape}")
            else:
                print(f"  Batch shape: n_node={batch.n_node.shape}, nodes={batch.nodes.shape}")
            print(f"  Padding efficiency: {100 * real_nodes / padded_nodes:.1f}%")

    # Print summary statistics
    print_bucket_batch_info(bucket_batch_info)


    # Print overall padding efficiency
    print(f"\nOverall Padding Efficiency:")
    print(f"  Total real nodes: {total_real_nodes}")
    print(f"  Total padded nodes: {total_padded_nodes}")
    print(f"  Efficiency: {100 * total_real_nodes / total_padded_nodes:.1f}%")
