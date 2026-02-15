"""Atom reference energies for machine learning potentials.

This module provides reference atomic energies indexed by atomic number indices,
along with utility functions for converting between atomic numbers and indices.
"""

import numpy as np


# Supported atomic numbers: 1-83 + actinides (89-94)
ATOMIC_NUMBERS: list[int] = list(np.arange(83) + 1) + [89, 90, 91, 92, 93, 94]


def atomic_numbers_to_indices(atomic_numbers: list[int]) -> dict[int, int]:
    """Convert list of atomic numbers to dictionary of atomic number to index.

    Args:
        atomic_numbers: List of atomic numbers to convert.

    Returns:
        Dictionary mapping atomic number to its index in the sorted list.
    """
    return {n: i for i, n in enumerate(sorted(atomic_numbers))}


# Mapping from atomic number to index
ATOMIC_NUMBER_TO_INDEX: dict[int, int] = atomic_numbers_to_indices(ATOMIC_NUMBERS)


# Reference atom energies indexed by index (not atomic number)
# To get energy for atomic number Z: ATOM_ENERGIES[ATOMIC_NUMBER_TO_INDEX[Z]]
ATOM_ENERGIES: np.ndarray = np.array([
    -3.667434143111067,
    -1.3389411521004957,
    -3.484135074374991,
    -4.7375733060698835,
    -7.724998581026458,
    -8.40493018309282,
    -7.359106991540641,
    -7.282559671986365,
    -4.895409014806584,
    -6.028859701624247,
    -2.762263125940574,
    -2.8136983653913563,
    -4.846934115287749,
    -7.696091863255697,
    -6.966080477719467,
    -4.671534887887628,
    -2.8111708516752074,
    -5.930012309429277,
    -2.620770860873613,
    -5.393286618553959,
    -7.885002234479723,
    -10.26933418802065,
    -8.66817278405163,
    -9.234620852031703,
    -8.306586950746253,
    -7.049408457014328,
    -5.577375408174762,
    -5.171629657091005,
    -3.253256908016715,
    -1.2909205626044318,
    -3.5271864259617245,
    -4.708383115733539,
    -3.9768273474182645,
    -3.88520201432572,
    -2.5177576519017673,
    6.738148650062646,
    -2.5662093090760325,
    -4.940529924986091,
    -10.150288403873372,
    -11.847117318569865,
    -12.139500032701292,
    -8.793594729721455,
    -8.788258171957457,
    -7.781667999439459,
    -6.848819809812409,
    -4.889813168432439,
    -2.0646919306369713,
    -0.6398246594832406,
    -2.7889114875588437,
    -3.8186354251894117,
    -3.5877140389889806,
    -2.8806433106169584,
    -1.6353329525856473,
    9.835435023043303,
    -2.766756626141838,
    -4.993088830386547,
    -8.935542464003472,
    -8.735872898798528,
    -8.01967577507233,
    -8.25281328614391,
    -7.592759816586044,
    -8.171141034747308,
    -13.59366897134993,
    -18.51795974504965,
    -7.647761248559496,
    -8.12332961409678,
    -7.6079234877279704,
    -6.850184952646372,
    -7.827295982170317,
    -3.5857170346217986,
    -7.456029208305462,
    -12.79525649426845,
    -14.10791957556341,
    -9.357687145958234,
    -11.388168140725373,
    -9.62472764941359,
    -7.323624357849622,
    -5.303987177841373,
    -2.379529355577072,
    0.24903573110237912,
    -2.325373833262773,
    -3.7321842627051143,
    -3.4406403694519634,
    -4.992778057851803,
    -11.023915580858981,
    -12.28052518740774,
    -13.85778414324286,
    -14.92242119954008,
    -15.280369699261607,
])


def get_atom_energy(atomic_number: int) -> float:
    """Get reference energy for a given atomic number.

    Args:
        atomic_number: Atomic number (Z) of the element.

    Returns:
        Reference energy for the atom.

    Raises:
        KeyError: If atomic number is not in the supported list.
    """
    index = ATOMIC_NUMBER_TO_INDEX[atomic_number]
    return ATOM_ENERGIES[index]


def get_total_reference_energy(atomic_numbers: list[int] | np.ndarray) -> float:
    """Calculate total reference energy for a list of atoms.

    Args:
        atomic_numbers: List or array of atomic numbers.

    Returns:
        Sum of reference energies for all atoms.
    """
    return sum(get_atom_energy(z) for z in atomic_numbers)