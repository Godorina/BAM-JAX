"""LAMMPS dump + log → ASE trajectory 변환.

Usage:
    python lammpsout_to_traj.py
    python lammpsout_to_traj.py --dump dump.lammpstrj --log log.lammps --output md.traj
"""

import argparse
import numpy as np
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator


def parse_energies(log_path):
    """Parse potential energies from LAMMPS log file.

    Reads the thermo output section (after the 'Step ... PotEng ...' header)
    and extracts the PotEng column.

    Args:
        log_path: Path to LAMMPS log file.

    Returns:
        List of potential energies (float), one per thermo output step.
    """
    energies = []
    in_thermo = False
    pe_col = None

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Step"):
                cols = line.split()
                # Find the PotEng column index
                if "PotEng" in cols:
                    pe_col = cols.index("PotEng")
                    in_thermo = True
                    continue
                elif "Pe" in cols:
                    pe_col = cols.index("Pe")
                    in_thermo = True
                    continue
            if in_thermo:
                # End of thermo block: non-numeric first field
                parts = line.split()
                if not parts:
                    continue
                try:
                    int(parts[0])  # Step is always an integer
                except ValueError:
                    in_thermo = False
                    pe_col = None
                    continue
                energies.append(float(parts[pe_col]))

    return energies


def convert(dump_path, log_path, output_path):
    """Convert LAMMPS dump + log to ASE trajectory.

    Args:
        dump_path: Path to LAMMPS dump file (dump.lammpstrj).
        log_path: Path to LAMMPS log file (log.lammps).
        output_path: Output ASE trajectory file path.
    """
    # Read all frames from dump file
    frames = read(dump_path, index=":", format="lammps-dump-text")
    if not isinstance(frames, list):
        frames = [frames]

    # Parse energies from log
    energies = parse_energies(log_path)

    if len(energies) != len(frames):
        print(f"Warning: {len(energies)} energies from log vs {len(frames)} frames from dump. "
              f"Using min({len(energies)}, {len(frames)}) frames.")

    n_frames = min(len(energies), len(frames))

    # Write trajectory
    with Trajectory(output_path, "w") as traj:
        for i in range(n_frames):
            atoms = frames[i]
            energy = energies[i]
            forces = atoms.arrays.get("forces", np.zeros((len(atoms), 3)))
            stress = np.zeros(6)  # Not available in thermo output

            calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
            atoms.calc = calc
            traj.write(atoms)

    print(f"Wrote {n_frames} frames to {output_path}")
    if n_frames > 0:
        print(f"  Energy range: [{energies[0]:.4f}, {energies[n_frames-1]:.4f}] eV")


def main():
    parser = argparse.ArgumentParser(description="Convert LAMMPS dump + log to ASE trajectory")
    parser.add_argument("--dump", default="dump.lammpstrj", help="LAMMPS dump file (default: dump.lammpstrj)")
    parser.add_argument("--log", default="log.lammps", help="LAMMPS log file (default: log.lammps)")
    parser.add_argument("--output", "-o", default="md.traj", help="Output trajectory file (default: md.traj)")
    args = parser.parse_args()

    convert(args.dump, args.log, args.output)


if __name__ == "__main__":
    main()
