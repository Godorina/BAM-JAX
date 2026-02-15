import pickle
from pathlib import Path
import sys
import ase
#import ase.io
import jraph
import matscipy.neighbours
import numpy as np
from tqdm import tqdm
from fairchem.core.datasets import AseDBDataset
import gc

def preprocess_graph(
    atoms: ase.Atoms,
    atom_energies: np.ndarray,
    atom_indices: dict[int, int],
    default_cutoff: float,
    targets: bool,
) -> jraph.GraphsTuple:
    """Preprocess ase.Atoms object into a jraph.GraphsTuple, with optional targets"""

    cutoff = default_cutoff
    src, dst, Sij = matscipy.neighbours.neighbour_list("ijS", atoms, cutoff)
    if len(src) < 20:
        cutoff = default_cutoff + 3.0
        src, dst, Sij = matscipy.neighbours.neighbour_list("ijS", atoms, cutoff)
        print (len(src), atoms.get_atomic_numbers())
        if len(src) == 0:
            return None

    forces = atoms.get_forces().astype(np.float32) if targets else None
    species = np.array(
                [atom_indices[n] for n in atoms.get_atomic_numbers()],
            ).astype(np.int32)

    graph_e0 = 0.0
    if atom_energies is not None:
        node_e0 = atom_energies[species]
        graph_e0 = node_e0.sum()

    energy = (
        np.array([atoms.get_potential_energy()-graph_e0]) if targets else None
    )
    cell = (
        np.array([atoms.get_cell()])
    )
    volume = (
        np.array([atoms.get_volume()])
    )
    stress = (
        np.array([atoms.get_stress()])
    )

    globals_cutoff = (
        np.array([cutoff])
    )

    return jraph.GraphsTuple(
        n_node=np.array([len(atoms)]).astype(np.int32),
        n_edge=np.array([len(src)]).astype(np.int32),
        nodes={
            "species": species,
            "positions": atoms.positions.astype(np.float32),
            "forces": forces,
        },
        edges={
            'Sij':Sij
        },
        senders=dst.astype(np.int32),
        receivers=src.astype(np.int32),
        globals={
            "energy": energy,
            "cell": cell,
            "volume": volume,
            "stress": stress,
            "cutoff": globals_cutoff
        },
    )

def atomic_numbers_to_indices(atomic_numbers: list[int]) -> dict[int, int]:
    """Convert list of atomic numbers to dictionary of atomic number to index."""
    return {n: i for i, n in enumerate(sorted(atomic_numbers))}


def save_checkpoint(checkpoint_path: str, processed_indices: set, current_idx: int):
    """Save checkpoint for resuming."""
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({'processed_indices': processed_indices, 'current_idx': current_idx}, f)
    print(f"  Checkpoint saved: {len(processed_indices)} indices processed")


def load_checkpoint(checkpoint_path: str):
    """Load checkpoint if exists."""
    if Path(checkpoint_path).exists():
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        print(f"  Resuming from checkpoint: {len(data['processed_indices'])} indices already processed")
        return data['processed_indices'], data['current_idx']
    return set(), 0


if __name__ == "__main__":
    from ase.io import write
    lines = open('/home/gpuuser/prog/BAM-nequip-main/input/atom_energies.dat', 'r').readlines()
    atom_energies = []
    for line in lines:
        key = line.split()
        atom_energies.append (float(key[1]))
    atom_energies = np.array(atom_energies)

    config_kwargs = {}
    dir_train = "/dataset/usr004/hgpark/omat24/train/"
    sample_list = [
        "aimd-from-PBE-1000-npt",
        "aimd-from-PBE-1000-nvt",
        "aimd-from-PBE-3000-npt",
        "aimd-from-PBE-3000-nvt",
        "rattled-1000",
        "rattled-1000-subsampled",
        "rattled-300",
        "rattled-300-subsampled",
        "rattled-500",
        "rattled-500-subsampled",
        "rattled-relax"
    ]
    trainset_paths = [
        dir_train + sample for sample in sample_list
    ]
    print (trainset_paths, len(trainset_paths))
    trainset = AseDBDataset(config=dict(src=trainset_paths,**config_kwargs))

    atomic_numbers = list(np.arange(83)+1) + [89, 90, 91, 92, 93, 94]
    atomic_indices = atomic_numbers_to_indices(atomic_numbers)
    ndata = len(trainset)
    rng = np.random.RandomState(seed=42)
    perm = rng.permutation(ndata)

    # === 설정 ===
    # 전체 데이터를 여러 개의 작은 pkl 파일로 나눔
    graphs_per_file = 50000      # 각 pkl 파일당 그래프 수 (메모리 효율)
    save_interval = 5000         # 몇 개마다 중간 저장할지
    cutoff = 6.0
    output_dir = Path("./train_pkl")
    output_dir.mkdir(exist_ok=True)
    checkpoint_path = "build_pkl_checkpoint.pkl"

    print(f"Total data: {ndata}")
    print(f"Graphs per file: {graphs_per_file}")
    print(f"Expected files: {(ndata + graphs_per_file - 1) // graphs_per_file}")

    # 체크포인트에서 재개
    processed_indices, start_file_idx = load_checkpoint(checkpoint_path)

    ipkl = start_file_idx
    graphs = []
    skipped = 0

    pbar = tqdm(range(ndata), desc="Processing graphs", initial=len(processed_indices))

    for i in pbar:
        # 이미 처리된 인덱스는 건너뛰기
        if i in processed_indices:
            continue

        try:
            graph = preprocess_graph(
                trainset.get_atoms(perm[i]),
                atom_energies,
                atomic_indices,
                cutoff,
                True
            )

            if graph is not None:
                graphs.append(graph)
            else:
                skipped += 1

            processed_indices.add(i)

        except Exception as e:
            print(f"\nError at index {i}: {e}")
            skipped += 1
            processed_indices.add(i)
            continue

        # 진행상황 업데이트
        pbar.set_postfix({
            "graphs": len(graphs),
            "skipped": skipped,
            "file": ipkl
        })

        # graphs_per_file 개 모이면 저장
        if len(graphs) >= graphs_per_file:
            pkl_path = output_dir / f'omat24_train_{ipkl:04d}.pkl'
            print(f"\nSaving {len(graphs)} graphs to {pkl_path}")
            with open(pkl_path, "wb") as f:
                pickle.dump(graphs, f)

            # 메모리 정리
            graphs = []
            gc.collect()

            ipkl += 1

            # 체크포인트 저장
            save_checkpoint(checkpoint_path, processed_indices, ipkl)

        # 중간 체크포인트 저장 (파일 저장 사이에도)
        elif len(processed_indices) % save_interval == 0:
            save_checkpoint(checkpoint_path, processed_indices, ipkl)

    # 남은 그래프 저장
    if len(graphs) > 0:
        pkl_path = output_dir / f'omat24_train_{ipkl:04d}.pkl'
        print(f"\nSaving remaining {len(graphs)} graphs to {pkl_path}")
        with open(pkl_path, "wb") as f:
            pickle.dump(graphs, f)
        ipkl += 1

    # 최종 체크포인트 저장
    save_checkpoint(checkpoint_path, processed_indices, ipkl)

    print(f"\n{'='*50}")
    print(f"Completed!")
    print(f"Total processed: {len(processed_indices)}")
    print(f"Total skipped: {skipped}")
    print(f"Total files created: {ipkl}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*50}")
