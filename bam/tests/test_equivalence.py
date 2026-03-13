"""
Equivalence test: bam vs bam_omat24
====================================
같은 가중치 + 같은 데이터 → 같은 에너지/힘/스트레스가 나오는지 검증.

사용법:
    cd /home/hgpark/BAM-JAX-master-phg_260227
    python -m bam.tests.test_equivalence

테스트 전략:
    1) 같은 seed로 두 모델을 초기화
    2) bam_omat24의 state를 bam에 복사 (구조 호환 확인)
    3) 같은 데이터에 대해 forward pass 비교
    4) node_energies (중간값) 비교
    5) 최종 energy, forces, stress 비교
"""

import sys
import os
import pickle
import numpy as np

# JAX setup
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import jax
import jax.numpy as jnp
from flax import nnx

# Import both codebases
sys.path.insert(0, "/home/hgpark/prog/bam_jax_260310/BAM_V2-main")
sys.path.insert(0, "/home/hgpark/BAM-JAX-master-phg_260227")


def print_header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def print_result(name, val, threshold=1e-5):
    status = "PASS" if val < threshold else "FAIL"
    print(f"  [{status}] {name}: {val:.2e} (threshold: {threshold:.0e})")
    return val < threshold


def compare_pytrees(tree_a, tree_b, prefix=""):
    """Compare two pytrees leaf by leaf, print mismatches."""
    leaves_a = jax.tree_util.tree_leaves_with_path(tree_a)
    leaves_b = jax.tree_util.tree_leaves_with_path(tree_b)

    if len(leaves_a) != len(leaves_b):
        print(f"  WARNING: leaf count mismatch: {len(leaves_a)} vs {len(leaves_b)}")

    n_match = 0
    n_mismatch = 0
    n_shape_mismatch = 0

    for (path_a, leaf_a), (path_b, leaf_b) in zip(leaves_a, leaves_b):
        path_str = "/".join(str(k) for k in path_a)
        if leaf_a.shape != leaf_b.shape:
            print(f"  SHAPE MISMATCH {path_str}: {leaf_a.shape} vs {leaf_b.shape}")
            n_shape_mismatch += 1
        elif not jnp.allclose(leaf_a, leaf_b, atol=1e-7):
            diff = jnp.abs(leaf_a - leaf_b).max()
            print(f"  VALUE MISMATCH {path_str}: max_diff={diff:.2e}")
            n_mismatch += 1
        else:
            n_match += 1

    print(f"  Summary: {n_match} match, {n_mismatch} value mismatch, "
          f"{n_shape_mismatch} shape mismatch")
    return n_mismatch == 0 and n_shape_mismatch == 0


def test_equivalence():
    # =====================================================================
    # Configuration (같은 파라미터로 두 모델 초기화)
    # hidden_irreps를 "64x0e + 64x1o + 64x2e"로 설정해야
    # bam의 하드코딩된 "64x0e" readout과 호환됨
    # =====================================================================
    config = dict(
        n_species=89,  # 원소 수 (둘 다 동일해야 함)
        lmax=3,
        cutoff=5.0,
        hidden_irreps="64x0e + 64x1o + 64x2e",
        n_layers=3,
        radial_basis_size=8,
        radial_mlp_size=64,
        radial_mlp_layers=3,
        radial_polynomial_p=2.0,
        mlp_init_scale=4.0,
        shift=0.0,
        scale=1.0,
        avg_n_neighbors=25.0,
        l_train=True,
        periodic=True,
    )
    seed = 42

    # =====================================================================
    # 0. 데이터 로드
    # =====================================================================
    print_header("Loading test data")
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "examples", "training",
        "data", "valid_pkl", "valid_0000.pkl"
    )
    if not os.path.exists(data_path):
        # fallback: absolute path
        data_path = "/home/hgpark/BAM-JAX-master-phg_260227/bam/examples/training/data/valid_pkl/valid_0000.pkl"

    if not os.path.exists(data_path):
        print("  ERROR: No test data found. Run examples/training/run.sh first.")
        print("  Trying to generate minimal synthetic data instead...")
        data = create_synthetic_data(config["n_species"])
    else:
        with open(data_path, "rb") as f:
            graphset = pickle.load(f)
        data = graphset[0]
        print(f"  Loaded graph: n_node={data.n_node}, n_edge={data.n_edge}")

    # =====================================================================
    # 1. 모델 초기화
    # =====================================================================
    print_header("Initializing models with same seed")

    # --- bam_omat24 model ---
    from bam_omat24.race_nnx import RACE as RACE_omat24

    model_omat24 = RACE_omat24(
        n_species=config["n_species"],
        lmax=config["lmax"],
        cutoff=config["cutoff"],
        hidden_irreps=config["hidden_irreps"],
        n_layers=config["n_layers"],
        radial_basis_size=config["radial_basis_size"],
        radial_mlp_size=config["radial_mlp_size"],
        radial_mlp_layers=config["radial_mlp_layers"],
        radial_polynomial_p=config["radial_polynomial_p"],
        mlp_init_scale=config["mlp_init_scale"],
        shift=config["shift"],
        scale=config["scale"],
        avg_n_neighbors=config["avg_n_neighbors"],
        l_train=config["l_train"],
        periodic=config["periodic"],
        rngs=nnx.Rngs(seed),
    )

    # --- bam model ---
    from bam.models.race_nnx import RACE as RACE_bam

    model_bam = RACE_bam(
        n_species=config["n_species"],
        lmax=config["lmax"],
        cutoff=config["cutoff"],
        hidden_irreps=config["hidden_irreps"],
        n_layers=config["n_layers"],
        radial_basis_size=config["radial_basis_size"],
        radial_mlp_size=config["radial_mlp_size"],
        radial_mlp_layers=config["radial_mlp_layers"],
        radial_polynomial_p=config["radial_polynomial_p"],
        mlp_init_scale=config["mlp_init_scale"],
        shift=config["shift"],
        scale=config["scale"],
        avg_n_neighbors=config["avg_n_neighbors"],
        l_train=config["l_train"],
        periodic=config["periodic"],
        use_checkpoint=False,
        rngs=nnx.Rngs(seed),
    )

    # Parameter count
    _, state_omat24 = nnx.split(model_omat24, nnx.Param)
    _, state_bam = nnx.split(model_bam, nnx.Param)
    n_omat24 = sum(x.size for x in jax.tree_util.tree_leaves(state_omat24))
    n_bam = sum(x.size for x in jax.tree_util.tree_leaves(state_bam))
    print(f"  bam_omat24 params: {n_omat24:,}")
    print(f"  bam params:        {n_bam:,}")
    if n_omat24 != n_bam:
        print(f"  WARNING: Parameter count mismatch! ({n_omat24} vs {n_bam})")
        print(f"  This means the models have different architectures.")
        print(f"  Check readout hidden_irreps: bam uses hardcoded '64x0e'")
    else:
        print(f"  Parameter counts match!")

    # =====================================================================
    # 2. 가중치 비교 (같은 seed → 같은 가중치인지)
    # =====================================================================
    print_header("Test 1: Same seed initialization check")
    same_init = compare_pytrees(state_omat24, state_bam)
    if same_init:
        print("  Same seed produces identical weights.")
    else:
        print("  Same seed produces DIFFERENT weights.")
        print("  Will copy omat24 weights to bam for forward pass test.")
        # 가중치 복사 시도
        try:
            nnx.update(model_bam, state_omat24)
            print("  Weight copy successful (nnx.update).")
        except Exception as e:
            print(f"  Weight copy failed: {e}")
            print("  Attempting manual leaf-by-leaf copy...")
            try:
                manual_copy_weights(model_omat24, model_bam)
                print("  Manual weight copy successful.")
            except Exception as e2:
                print(f"  Manual copy also failed: {e2}")
                print("  Cannot proceed with forward pass comparison.")
                return

    # =====================================================================
    # 3. Forward pass 비교 (node_energies 중간값)
    # =====================================================================
    print_header("Test 2: node_energies() comparison")

    positions = data.nodes["positions"]
    cell = data.globals.get("cell", None)
    cutoff_val = data.globals.get("cutoff", None)

    # bam_omat24: returns dict
    result_omat24 = model_omat24.node_energies(positions, cell, cutoff_val, data)
    ne_omat24 = result_omat24["node_energies"]

    # bam: returns plain array
    ne_bam = model_bam.node_energies(positions, cell, cutoff_val, data)

    max_diff_ne = float(jnp.abs(ne_omat24 - ne_bam).max())
    mean_diff_ne = float(jnp.abs(ne_omat24 - ne_bam).mean())
    print(f"  node_energies shape: omat24={ne_omat24.shape}, bam={ne_bam.shape}")
    print(f"  omat24 range: [{float(ne_omat24.min()):.6f}, {float(ne_omat24.max()):.6f}]")
    print(f"  bam    range: [{float(ne_bam.min()):.6f}, {float(ne_bam.max()):.6f}]")
    print_result("node_energies max_diff", max_diff_ne)
    print_result("node_energies mean_diff", mean_diff_ne)

    # =====================================================================
    # 4. Full forward pass (energy, forces, stress)
    # =====================================================================
    print_header("Test 3: __call__() full forward pass")

    e_omat24, f_omat24, s_omat24 = model_omat24(data)
    e_bam, f_bam, s_bam = model_bam(data)

    print(f"\n  --- Energy ---")
    print(f"  omat24: {e_omat24}")
    print(f"  bam:    {e_bam}")
    diff_e = float(jnp.abs(e_omat24 - e_bam).max())
    print_result("energy max_diff", diff_e)

    print(f"\n  --- Forces ---")
    print(f"  omat24 shape: {f_omat24.shape}, norm: {float(jnp.linalg.norm(f_omat24)):.6f}")
    print(f"  bam    shape: {f_bam.shape}, norm: {float(jnp.linalg.norm(f_bam)):.6f}")
    diff_f = float(jnp.abs(f_omat24 - f_bam).max())
    mean_diff_f = float(jnp.abs(f_omat24 - f_bam).mean())
    print_result("forces max_diff", diff_f)
    print_result("forces mean_diff", mean_diff_f)

    print(f"\n  --- Stress ---")
    print(f"  omat24: {s_omat24}")
    print(f"  bam:    {s_bam}")
    diff_s = float(jnp.abs(s_omat24 - s_bam).max())
    print_result("stress max_diff", diff_s)

    # =====================================================================
    # 5. Summary
    # =====================================================================
    print_header("SUMMARY")
    all_pass = True
    all_pass &= print_result("param count match", abs(n_omat24 - n_bam), threshold=0.5)
    all_pass &= print_result("node_energies", max_diff_ne)
    all_pass &= print_result("energy", diff_e)
    all_pass &= print_result("forces", diff_f)
    all_pass &= print_result("stress", diff_s)

    if all_pass:
        print("\n  ALL TESTS PASSED - Models are equivalent!")
    else:
        print("\n  SOME TESTS FAILED - Models differ!")
        print("  Check the details above to identify where they diverge.")


def manual_copy_weights(src_model, dst_model):
    """Manually copy weights leaf-by-leaf, matching by path name."""
    gd_src, state_src = nnx.split(src_model, nnx.Param)
    gd_dst, state_dst = nnx.split(dst_model, nnx.Param)

    src_leaves = dict(jax.tree_util.tree_leaves_with_path(state_src))
    dst_leaves = dict(jax.tree_util.tree_leaves_with_path(state_dst))

    new_leaves = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(state_dst):
        path_str = "/".join(str(k) for k in path)
        # Try to find matching path in source
        matched = False
        for src_path, src_leaf in jax.tree_util.tree_leaves_with_path(state_src):
            src_path_str = "/".join(str(k) for k in src_path)
            if src_path_str == path_str:
                if src_leaf.shape == leaf.shape:
                    new_leaves[path] = src_leaf
                    matched = True
                else:
                    print(f"    Shape mismatch at {path_str}: "
                          f"{src_leaf.shape} vs {leaf.shape}, skipping")
                break
        if not matched:
            new_leaves[path] = leaf  # keep original

    # Rebuild state with copied leaves
    new_state = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(state_dst),
        [new_leaves.get(p, l) for p, l in jax.tree_util.tree_leaves_with_path(state_dst)]
    )
    nnx.update(dst_model, new_state)


def create_synthetic_data(n_species):
    """Create minimal synthetic graph for testing when no data file available."""
    import jraph

    n_atoms = 8
    n_edges = 24

    nodes = {
        "positions": jnp.array(np.random.randn(n_atoms, 3).astype(np.float32)),
        "species": jnp.array([0, 1, 0, 1, 2, 2, 0, 1], dtype=jnp.int32),
        "forces": jnp.zeros((n_atoms, 3)),
    }

    # Random senders/receivers
    senders = jnp.array([0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5, 6,6,6, 7,7,7])
    receivers = jnp.array([1,2,3, 0,2,4, 0,1,5, 0,4,6, 1,3,7, 2,6,7, 3,5,7, 4,5,6])

    edges = {
        "Sij": jnp.zeros((n_edges, 3)),
    }

    cell = jnp.eye(3)[None] * 10.0  # [1, 3, 3]
    cutoff = jnp.array([5.0])
    volume = jnp.array([1000.0])

    globals_ = {
        "energy": jnp.array([0.0]),
        "stress": jnp.zeros((1, 6)),
        "cell": cell,
        "cutoff": cutoff,
        "volume": volume,
    }

    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([n_atoms]),
        n_edge=jnp.array([n_edges]),
        globals=globals_,
    )


if __name__ == "__main__":
    test_equivalence()
