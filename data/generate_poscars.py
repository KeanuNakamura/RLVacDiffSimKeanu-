from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
from ase import io
import random
from ase.build import bulk
import os
from ase import Atoms
from ase.formula import Formula
from typing import List, Tuple
import numpy as np
from tqdm import tqdm


def get_sro_from_atoms(atoms):
    species = list(sorted(set(atoms.get_atomic_numbers().tolist())))
    n_species = len(species)
    SRO = np.zeros((n_species, n_species))
    # n_atoms = len(atoms)
    dist_all = atoms.get_all_distances(mic=True)
    dist_list = dist_all.flatten()
    dist_list = dist_list[dist_list > 0.1]
    #print(sorted(dist_list,reverse=True))
    NN_dist = np.min(dist_list)
    r_cut = (1 + np.sqrt(2)) / 2 * NN_dist

    pairs = (dist_all > 0.1) * (dist_all < r_cut)
    total_pairs = np.sum(pairs) / 2
    atomic_numbers = atoms.get_atomic_numbers()
    specie_list = [atomic_numbers == species[n] for n in range(n_species)]
    n_specie_list = [np.sum(specie_list[n]) for n in range(n_species)]
    n_specie_list = np.array(n_specie_list) / np.sum(n_specie_list)

    for n1 in range(n_species):
        for n2 in range(n1 + 1):
            number_of_pairs = (
                np.sum(pairs * specie_list[n1][None, :] * specie_list[n2][:, None])
                / 2
            )
            SRO[n1, n2] = (
                1
                - (number_of_pairs / total_pairs)
                / n_specie_list[n1]
                / n_specie_list[n2]
            )

            if n1 != n2:
                SRO[n2, n1] = SRO[n1, n2]

    return SRO

def get_2nn_sro_from_atoms(atoms):
    species = list(sorted(set(atoms.get_atomic_numbers().tolist())))
    n_species = len(species)
    SRO = np.zeros((n_species, n_species))
    # n_atoms = len(atoms)
    dist_all = atoms.get_all_distances(mic=True)
    dist_list = dist_all.flatten()
    dist_list = dist_list[dist_list > 0.1]
    # >3.1 and <4.1
    NN_dist = np.min(dist_list)
    r_cut = (1 + np.sqrt(2)) / 2 * NN_dist

    pairs = (dist_all > 3.1) * (dist_all<4.1) 
    total_pairs = np.sum(pairs) / 2
    atomic_numbers = atoms.get_atomic_numbers()
    specie_list = [atomic_numbers == species[n] for n in range(n_species)]
    n_specie_list = [np.sum(specie_list[n]) for n in range(n_species)]
    n_specie_list = np.array(n_specie_list) / np.sum(n_specie_list)

    for n1 in range(n_species):
        for n2 in range(n1 + 1):
            number_of_pairs = (
                np.sum(pairs * specie_list[n1][None, :] * specie_list[n2][:, None])
                / 2
            )
            SRO[n1, n2] = (
                1
                - (number_of_pairs / total_pairs)
                / n_specie_list[n1]
                / n_specie_list[n2]
            )

            if n1 != n2:
                SRO[n2, n1] = SRO[n1, n2]

    return SRO

def sort_atoms(atoms):
    new_atoms = Atoms()
    new_atoms.cell = atoms.cell
    new_atoms.constraints = atoms.constraints

    w = atoms.symbols
    formula = Formula(f"{w}")
    count = list(formula.count())
    for _, symbol in enumerate(count):
        for i in range(len(atoms)):
            if atoms[i].symbol == symbol:
                new_atoms.append(atoms[i])
    return new_atoms

def make_supercell(atoms: Atoms, xyz_scaling:Tuple[int]):
    atoms = AseAtomsAdaptor.get_structure(atoms)
    atoms.make_supercell(list(xyz_scaling))
    atoms = AseAtomsAdaptor.get_atoms(atoms)
    atoms = sort_atoms(atoms)
    return atoms

# Step 1: Build FCC primitive cell
def generate_rss(n_vacancies=1, n_cr=167, n_co=166, n_ni=166, a0=3.528, supercell_size=5):
    prim = bulk('Ni', crystalstructure='fcc', a=a0, cubic=True)
    # prim = AseAtomsAdaptor.get_structure(prim)
    # Step 2: Build 5×5×5 FCC supercell → 500 atoms
    supercell = make_supercell(prim, (supercell_size, supercell_size, supercell_size))
    # supercell = AseAtomsAdaptor.get_atoms(supercell)
    total_atoms = len(supercell)
    assert total_atoms == n_cr + n_co + n_ni + n_vacancies, f"Expected {total_atoms}"
    # Step 3: Randomly remove one atom (vacancy)
    if n_vacancies > 0:
        vacancy_indices = random.sample(range(total_atoms), n_vacancies)
        for idx in sorted(vacancy_indices, reverse=True):
            del supercell[idx]

    # Step 4: Prepare atoms 
    elements = ['Cr'] * n_cr + ['Co'] * n_co + ['Ni'] * n_ni 
    random.shuffle(elements)
    # Step 5: Shuffle and assign to atoms
    random.shuffle(elements)
    supercell.set_chemical_symbols(elements)
    supercell = sort_atoms(supercell)
    return supercell

if __name__ == '__main__':
    MAX_TRY = 1000000
    directory = "/home/keanun65/projects/RLVacDiffSim/data/poscar_files"
    n_vacancies = 1
    n_cr = 85
    n_co = 85
    n_ni = 85
    n_poscars = 50
    supercell_size = 4

    count = 0
    while count < n_poscars:
        for num in range(MAX_TRY):
            supercell = generate_rss(n_vacancies=n_vacancies, n_cr=n_cr, n_co=n_co, n_ni=n_ni, a0=3.528, supercell_size=supercell_size)
            sro_matrix = get_sro_from_atoms(supercell)
            sro = np.linalg.norm(np.triu(sro_matrix))
            if sro < 0.03:
                print(num, count, sro)
                filename = os.path.join(directory, f'POSCAR_{count}')
                io.write(filename, supercell, format='vasp', vasp5=True)
                count += 1
                break  # ✅ important to break out of for-loop when count is incremented
