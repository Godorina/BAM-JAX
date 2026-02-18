"""Export trained RACE model for LAMMPS deployment.

Converts a trained checkpoint (ckpt_best.pkl) to either:
- MLIR protobuf (.ptb) via chemtrain-deploy (for LAMMPS pair_style chemtrain)
- Standalone pickle (.pkl) for validation without chemtrain-deploy

Usage:
    # Export to MLIR (uses local deploy module, no JAX version constraint):
    python -m bam.lammps.export_model \
        --ckpt ckpt_best.pkl \
        --config input.json \
        --output model-lammps.ptb

    # Export standalone pickle (for validation):
    python -m bam.lammps.export_model \
        --ckpt ckpt_best.pkl \
        --config input.json \
        --output model-standalone.pkl \
        --standalone
"""

import argparse
import json
import sys

from bam.lammps.exporter import BAMExporter


def main():
    parser = argparse.ArgumentParser(
        description='Export BAM-JAX RACE model for LAMMPS deployment'
    )
    parser.add_argument(
        '--ckpt', required=True,
        help='Path to trained checkpoint (ckpt_best.pkl)'
    )
    parser.add_argument(
        '--config', required=True,
        help='Path to model config JSON file'
    )
    parser.add_argument(
        '--output', default='model-lammps.ptb',
        help='Output file path (default: model-lammps.ptb)'
    )
    parser.add_argument(
        '--cutoff', type=float, default=None,
        help='Override cutoff radius (default: from config)'
    )
    parser.add_argument(
        '--standalone', action='store_true',
        help='Save as standalone pickle instead of MLIR protobuf'
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    # Load atom energies if not in config
    if 'atom_energies' not in config:
        from bam.data.atom_energies import ATOM_ENERGIES
        config['atom_energies'] = ATOM_ENERGIES.tolist()

    # Create exporter
    exporter = BAMExporter(
        ckpt_path=args.ckpt,
        config=config,
        cutoff=args.cutoff,
    )

    if args.standalone:
        # Save as standalone pickle (no chemtrain dependency)
        exporter.save_standalone(args.output)
    else:
        # Export to MLIR via chemtrain-deploy
        exporter.export()
        exporter.save(args.output)

    print(f"\nExport complete:")
    print(f"  Output: {args.output}")
    print(f"  Cutoff: {exporter.r_cutoff} A")
    print(f"  Unit style: {exporter.unit_style}")
    print(f"  Format: {'standalone pickle' if args.standalone else 'MLIR protobuf'}")


if __name__ == '__main__':
    main()
