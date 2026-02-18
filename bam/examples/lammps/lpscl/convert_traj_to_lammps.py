#!/usr/bin/env python3
"""Convert valid_200.traj (frame 0) to LAMMPS data file for LPSCl.

Atom types = atomic numbers (chemtrain_deploy requirement: species = type - 1).
  Li -> type 3, P -> type 15, S -> type 16, Cl -> type 17
Declares 17 atom types (H through Cl) so types match atomic numbers exactly.
"""

import numpy as np
from ase.io import read

# ── Configuration ────────────────────────────────────────────────
TRAJ_FILE = "/home/hgpark/main_git_251204/MACE_V4_251218_LoRA/Data_sampling/target/valid_data/valid_200.traj"
OUTPUT_FILE = "lpscl_structure.data"
FRAME_INDEX = 0

# Standard atomic masses for types 1-17 (H through Cl)
MASSES = {
    1:   1.0080,   # H
    2:   4.0026,   # He
    3:   6.9400,   # Li
    4:   9.0122,   # Be
    5:  10.8100,   # B
    6:  12.0110,   # C
    7:  14.0070,   # N
    8:  15.9990,   # O
    9:  18.9984,   # F
    10:  20.1797,  # Ne
    11:  22.9898,  # Na
    12:  24.3050,  # Mg
    13:  26.9815,  # Al
    14:  28.0855,  # Si
    15:  30.9738,  # P
    16:  32.0600,  # S
    17:  35.4530,  # Cl
}

ELEMENT_NAMES = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl",
}


def cell_to_lammps(cell):
    """Convert ASE cell (3x3 matrix) to LAMMPS box parameters.

    LAMMPS triclinic box:
        xlo xhi  (with xy xz yz tilt factors)
        a = (xhi-xlo, 0, 0)
        b = (xy, yhi-ylo, 0)
        c = (xz, yz, zhi-zlo)
    """
    a, b, c = cell[0], cell[1], cell[2]

    # a vector along x
    ax = np.linalg.norm(a)

    # b vector in xy plane
    bx = np.dot(b, a) / ax
    by = np.sqrt(np.dot(b, b) - bx**2)

    # c vector
    cx = np.dot(c, a) / ax
    cy = (np.dot(c, b) - bx * cx) / by
    cz = np.sqrt(np.dot(c, c) - cx**2 - cy**2)

    xlo, xhi = 0.0, ax
    ylo, yhi = 0.0, by
    zlo, zhi = 0.0, cz
    xy, xz, yz = bx, cx, cy

    return xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz


def atoms_to_lammps_coords(atoms, cell_params):
    """Convert ASE fractional coords to LAMMPS Cartesian coords.

    LAMMPS triclinic: r = s1*a_lammps + s2*b_lammps + s3*c_lammps
    where a_lammps=(Lx,0,0), b_lammps=(xy,Ly,0), c_lammps=(xz,yz,Lz)
    """
    xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz = cell_params
    Lx = xhi - xlo
    Ly = yhi - ylo
    Lz = zhi - zlo

    # Get fractional coordinates
    frac = atoms.get_scaled_positions()

    # Convert to LAMMPS Cartesian
    x = frac[:, 0] * Lx + frac[:, 1] * xy + frac[:, 2] * xz + xlo
    y = frac[:, 1] * Ly + frac[:, 2] * yz + ylo
    z = frac[:, 2] * Lz + zlo

    return np.column_stack([x, y, z])


def main():
    # Read trajectory
    atoms = read(TRAJ_FILE, index=FRAME_INDEX)
    n_atoms = len(atoms)
    atomic_numbers = atoms.get_atomic_numbers()
    n_types = 17  # H through Cl

    print(f"Read frame {FRAME_INDEX}: {n_atoms} atoms")
    unique, counts = np.unique(atomic_numbers, return_counts=True)
    for z, cnt in zip(unique, counts):
        print(f"  {ELEMENT_NAMES.get(z, '?')} (Z={z}): {cnt}")

    # Cell parameters
    cell = atoms.get_cell()
    cell_params = cell_to_lammps(cell)
    xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz = cell_params

    print(f"Cell: xlo={xlo:.4f} xhi={xhi:.4f}")
    print(f"      ylo={ylo:.4f} yhi={yhi:.4f}")
    print(f"      zlo={zlo:.4f} zhi={zhi:.4f}")
    print(f"      xy={xy:.4f} xz={xz:.4f} yz={yz:.4f}")

    # Atom positions in LAMMPS coordinates
    coords = atoms_to_lammps_coords(atoms, cell_params)

    # ── Write LAMMPS data file ──────────────────────────────────
    with open(OUTPUT_FILE, "w") as f:
        f.write("LPSCl Li200S168P32Cl24 (valid_200.traj frame 0)\n\n")
        f.write(f"{n_atoms} atoms\n")
        f.write(f"{n_types} atom types\n\n")

        f.write(f"{xlo:.10f}  {xhi:.10f}  xlo xhi\n")
        f.write(f"{ylo:.10f}  {yhi:.10f}  ylo yhi\n")
        f.write(f"{zlo:.10f}  {zhi:.10f}  zlo zhi\n")
        f.write(f"{xy:.10f}  {xz:.10f}  {yz:.10f}  xy xz yz\n\n")

        f.write("Masses\n\n")
        for t in range(1, n_types + 1):
            used = "*" if t in atomic_numbers else ""
            f.write(f"  {t:>2}  {MASSES[t]:>10.4f}  # {ELEMENT_NAMES[t]} {used}\n")

        f.write("\nAtoms # atomic\n\n")
        for i in range(n_atoms):
            atom_type = int(atomic_numbers[i])
            x, y, z = coords[i]
            f.write(f"  {i+1:>5}  {atom_type:>2}  {x:>20.15f}  {y:>20.15f}  {z:>20.15f}\n")

    print(f"\nWritten {OUTPUT_FILE}: {n_atoms} atoms, {n_types} types")


if __name__ == "__main__":
    main()
