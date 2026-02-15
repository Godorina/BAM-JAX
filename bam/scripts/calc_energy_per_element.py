import numpy as np
from scipy.optimize import minimize
from ase.io import read, iread
from tqdm import tqdm
import json


def get_enr_avg_per_element(traj, element):
    tgt_enr = np.array([atoms.get_potential_energy()
                        for atoms in tqdm(traj, desc="Reading energies")])

    uniq_element = {int(e): i for i, e in enumerate(element)}
    element_counts = {i: np.array([(atoms.numbers == e).sum()
                                   for atoms in tqdm(traj, desc=f"Counting element (index {i})")])
                      for e, i in uniq_element.items()}
    c0 = np.array([element_counts[i] for i in element_counts.keys()])
    m0 = tgt_enr.sum() / c0.sum()
    w0 = np.array([m0 for _ in element])

    def loss_fn(weight, count):
        # weight:  (nspec)
        # count:  (nspec, ndata)

        def objective_mean(w0, c0):
            # w0: weight (nspec)
            # c0: count  (nspec, ndata)
            return np.einsum('i,ij->j', w0, c0)

        prd_enr = objective_mean(weight, count)
        diff = (tgt_enr - prd_enr)
        return (diff * diff).mean()

    results = minimize(loss_fn, x0=w0, args=(c0,), method='BFGS')
    w0 = results.x

    enr_avg_per_element = {}
    for i, e in enumerate(element):
        enr_avg_per_element[i] = w0[i]

    return enr_avg_per_element, uniq_element


if __name__ == "__main__":
    # Trajectory 파일 경로
    traj_path = "/home/hgpark/Desktop/bam_jax/BAM-nequip-main/catalyst/substrate_HER/data/total_substrate_HER_data.traj"

    # Trajectory 읽기
    print("Loading trajectory...")
    traj = list(tqdm(iread(traj_path, index=":"), desc="Reading structures"))
    print(f"Loaded {len(traj)} structures from trajectory")

    # 모든 구조에서 고유한 원소 번호 추출
    all_elements = set()
    for atoms in traj:
        all_elements.update(atoms.numbers)
    element = sorted(list(all_elements))
    print(f"Found elements (atomic numbers): {element}")

    # 원소 기호로 변환하여 출력
    from ase.data import chemical_symbols
    element_symbols = [chemical_symbols[z] for z in element]
    print(f"Found elements (symbols): {element_symbols}")

    # 원소별 평균 에너지 계산
    enr_avg_per_element, uniq_element = get_enr_avg_per_element(traj, element)

    # 결과를 리스트로 정리 (Python 기본 타입으로 변환)
    atomic_numbers = [int(e) for e in element]
    atom_energies = [round(float(enr_avg_per_element[i]), 3) for i in range(len(element))]

    # 딕셔너리 형태로 저장
    result = {
        "atomic_numbers": atomic_numbers,
        "atom_energies": atom_energies
    }

    print("\n" + "=" * 50)
    print("Result:")
    print("=" * 50)
    print(f'"atomic_numbers": {atomic_numbers}')
    print(f'"atom_energies": {atom_energies}')

    print("\n" + "=" * 50)
    print("Energy per element (eV/atom):")
    print("=" * 50)
    for idx, atomic_num in enumerate(element):
        symbol = chemical_symbols[atomic_num]
        energy = enr_avg_per_element[idx]
        print(f"  {symbol} (Z={atomic_num}): {energy:.6f} eV")

    # JSON 파일로 저장
    output_path = "/home/hgpark/Desktop/bam_jax/BAM-nequip-main/catalyst/substrate_HER/data/atom_energies_260211.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {output_path}")
