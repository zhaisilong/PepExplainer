import copy
import shutil
from typing import Optional, List, Dict, Tuple, Union, Literal
import random
import dgl
import numpy as np
import pandas as pd
import torch
from IPython.core.display import display
from dgl import load_graphs
from rdkit import Chem
from rdkit.Chem import rdchem, Draw, AllChem
from sklearn import metrics
from tqdm.auto import tqdm
from pathlib import Path
from copy import deepcopy
import dill
from typing import Union, Any
from loguru import logger
from . import thrombin
from itertools import chain

import yaml

# 移除默认的日志处理器（即输出到控制台的处理器）
logger.remove()
logger.add("data.log", level="INFO", rotation="1 week")

amino_acid_smiles = {
    "A": "C[C@H](N)C(O)=O",
    "R": "O=C(O)[C@H](CCCNC(N)=N)N",
    "N": "O=C(O)[C@H](CC(N)=O)N",
    "D": "O=C(O)[C@H](CC(O)=O)N",
    "C": "O=C(O)[C@H](CS)N",
    "Q": "O=C(O)[C@H](CCC(N)=O)N",
    "E": "O=C(O)[C@H](CCC(O)=O)N",
    "G": "O=C(O)CN",
    "H": "O=C(O)[C@H](CC1=CNC=N1)N",
    "I": "CC[C@H](C)[C@H](N)C(O)=O",
    "L": "CC(C)C[C@H](N)C(O)=O",
    "K": "O=C([C@@H](N)CCCCN)O",
    "M": "O=C(O)[C@@H](N)CCSC",
    "F": "O=C(O)[C@@H](N)CC1=CC=CC=C1",
    "P": "O=C([C@@H]1CCCN1)O",
    "S": "O=C(O)[C@H](CO)N",
    "T": "O=C(O)[C@H]([C@H](O)C)N",
    "W": "O=C(O)[C@@H](N)CC1=CNC2=CC=CC=C12",
    "Y": "O=C(O)[C@H](CC1=CC=C(O)C=C1)N",
    "V": "CC(C)[C@H](N)C(O)=O",
}

amino_acid_symbol = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

from transformers import BertTokenizer


def set_random_seed(seed: Optional[int] = None):
    """Set random seed. 大衍之数 55
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    if seed is None:
        seed = 55
    random.seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    # dgl.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return [x == s for s in allowable_set]


def atom_features(atom, use_chirality=True):
    results = (
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "B",
                "C",
                "N",
                "O",
                "F",
                "Si",
                "P",
                "S",
                "Cl",
                "As",
                "Se",
                "Br",
                "Te",
                "I",
                "At",
                "other",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
        + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
        + one_of_k_encoding_unk(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
                "other",
            ],
        )
        + [atom.GetIsAromatic()]
    )
    results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = (
                results
                + one_of_k_encoding_unk(atom.GetProp("_CIPCode"), ["R", "S"])
                + [atom.HasProp("_ChiralityPossible")]
            )
        except:
            results = results + [False, False] + [atom.HasProp("_ChiralityPossible")]

    return np.array(results)


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def etype_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats_1 = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
    ]
    for i, m in enumerate(bond_feats_1):
        if m == True:
            a = i

    bond_feats_2 = bond.GetIsConjugated()
    if bond_feats_2 == True:
        b = 1
    else:
        b = 0

    bond_feats_3 = bond.IsInRing
    if bond_feats_3 == True:
        c = 1
    else:
        c = 0

    index = a * 1 + b * 4 + c * 8
    if use_chirality:
        bond_feats_4 = one_of_k_encoding_unk(
            str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
        )
        for i, m in enumerate(bond_feats_4):
            if m == True:
                d = i
        index = index + d * 16
    return index


def get_atom_by_map_num(mol: rdchem.Mol, map_num) -> Optional[rdchem.Mol]:
    """通过原子映射编号获取原子"""
    for _atom in mol.GetAtoms():
        if _atom.GetAtomMapNum() == map_num:
            return _atom
    return None


def get_substructure_bond(substructure: Dict, mol: rdchem.Mol) -> List[List[int]]:
    substructure_bond = []
    keys = [k for k in substructure.keys()]
    values = [min(v) for v in substructure.values()]
    for i in range(len(keys) - 1):
        if mol.GetBondBetweenAtoms(keys[i], keys[i + 1]):
            substructure_bond.append([keys[i], keys[i + 1]])
    return substructure_bond


def get_atom_Idx_by_map_num(mol: rdchem.Mol, map_num):
    _atom = get_atom_by_map_num(mol, map_num)
    if _atom is None:
        return None
    else:
        return _atom.GetIdx()


def set_atom_map_nums(mol, map_nums: List) -> None:
    """设置原子编号"""
    atoms = mol.GetAtoms()
    assert len(atoms) == len(map_nums), "AllChem objects must have the same length"
    for atom, map_num in zip(atoms, map_nums):
        mol.GetAtomWithIdx(atom.GetIdx()).SetAtomMapNum(map_num)


def init_atom_map_nums(mol) -> None:
    """初始化原子映射编号为原子编号"""
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp(
            "molAtomMapNumber", str(mol.GetAtomWithIdx(idx).GetIdx())
        )


def show_atom_map_nums(mol, size=(300, 300), prompt: Optional[str] = None) -> None:
    _mol = copy.deepcopy(mol)
    atoms = _mol.GetAtoms()
    for atom in atoms:
        atom.SetProp("molAtomMapNumber", str(atom.GetAtomMapNum()))
    if prompt:
        print(prompt)
    display(Draw.MolToImage(_mol, size=size))


def show_atom_idx(mol, size: Tuple = (300, 300), prompt: Optional[str] = None) -> None:
    _mol = copy.deepcopy(mol)
    atoms = _mol.GetAtoms()
    for atom in atoms:
        atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
    if prompt:
        print(prompt)
    display(Draw.MolToImage(_mol, size=size))


def find_terminal_C(mol) -> Optional[int]:
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "C" and atom.GetTotalDegree() == 3:
            neighbors = [neighbor.GetSymbol() for neighbor in atom.GetNeighbors()]
            if neighbors.count("O") == 2:
                return atom.GetIdx()
    return None


def find_terminal_N(mol) -> Optional[int]:
    terminal_C = find_terminal_C(mol)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "N" and (
            atom.GetTotalNumHs() == 1 or atom.GetTotalNumHs() == 2
        ):
            # Find the shortest path between the two atoms
            atom_idx = atom.GetIdx()
            path = Chem.rdmolops.GetShortestPath(mol, atom_idx, terminal_C)
            # The number of neighbors is the number of intermediate atoms in the path
            if len(path) == 3:
                return atom_idx
    return None


def find_carboxyl_O(mol) -> Optional[int]:
    terminal_C = find_terminal_C(mol)
    for atom in mol.GetAtomWithIdx(terminal_C).GetNeighbors():
        if (
            atom.GetSymbol() == "O" and atom.GetTotalNumHs() == 1
        ):  # the oxygen has a hydrogen
            return atom.GetIdx()
    return None


class AminoAcid:
    def __init__(self, symbol: str):
        if symbol not in amino_acid_smiles.keys():
            raise ValueError(f"Invalid amino acid symbol: {symbol}")
        self.symbol = symbol
        self.smiles = amino_acid_smiles[symbol]
        self.mol = AllChem.MolFromSmiles(self.smiles)

    def __len__(self):
        return self.num_atoms

    def set_maps(self, start: int = 0) -> None:
        """Set maps and make sure beginning at C and ending at N"""
        delta = start
        C_map_number = 0 + start  # Always the first atom
        N_map_number = self.num_atoms + start - 1  # The last atom
        O_map_number = 1 + start  # The second atom

        map_list = []
        map_iterator = iter(range(2, self.num_atoms - 1))  # n = total - 3
        for i in range(self.num_atoms):
            if i == self.terminal_N:
                map_list.append(N_map_number)
            elif i == self.terminal_C:
                map_list.append(C_map_number)
            elif i == self.carboxyl_O:  # always next to terminal C
                map_list.append(O_map_number)
            else:
                try:
                    j = next(map_iterator)
                    map_list.append(j + delta)
                except Exception as e:
                    print(f"{e} failed to set maps")
                    break
        set_atom_map_nums(self.mol, map_list)

    def __repr__(self):
        return self.symbol

    def __str__(self):
        return repr(self)

    @property
    def terminal_C(self) -> Optional[int]:
        return find_terminal_C(self.mol)

    @property
    def terminal_N(self) -> Optional[int]:
        return find_terminal_N(self.mol)

    @property
    def carboxyl_O(self) -> Optional[int]:
        """Return the Oxygen of Car"""
        return find_carboxyl_O(self.mol)

    @property
    def atom_idx_pairs(self) -> List[Tuple[Union[str, int]]]:
        atom_idx_list = []
        for atom in self.mol.GetAtoms():
            atom_idx_list.append((atom.GetSymbol(), atom.GetIdx()))
        return atom_idx_list

    @property
    def num_atoms(self):
        return self.mol.GetNumAtoms()


def get_mapped_substructure(amino_acids: List[AminoAcid]) -> Dict:
    """This function sets the maps of amino acids in peptides

    * set unique atom map numbers for amino acids in peptide
    """
    substructure = dict()
    start = 0
    for i, amino_acid in enumerate(amino_acids):
        if i == 0:
            amino_acid.set_maps()
            substructure[start] = [
                i for i in range(amino_acid.num_atoms) if i != 1
            ]  # remove oxygen
            start += amino_acid.num_atoms
        else:
            amino_acid.set_maps(start=start)
            substructure[start + amino_acid.num_atoms - 1] = [
                i + start for i in range(amino_acid.num_atoms) if i != 1
            ]  # remove oxygen, the N is the beginning of extended amino acids
            start += amino_acid.num_atoms
    return substructure


def link(
    peptide: rdchem.Mol,
    amino_acid: AminoAcid,
    mapped_terminal_C: int,
    mapped_terminal_N: int,
):
    peptide_mol = Chem.RWMol(peptide)
    amino_acid_mol = Chem.RWMol(amino_acid.mol)

    # Get the oxygen (-OH) to be removed
    oxygen = None
    for atom in get_atom_by_map_num(peptide_mol, mapped_terminal_C).GetNeighbors():
        if atom.GetSymbol() == "O" and atom.GetTotalNumHs() == 1:
            oxygen = atom.GetIdx()
    assert oxygen is not None, "failed to find the oxygen"

    # 从羧基中删除氢氧原子
    peptide_mol.RemoveAtom(oxygen)
    # 在羧基碳和氨基氮之间添加肽键
    combined = Chem.RWMol(
        Chem.CombineMols(peptide_mol.GetMol(), amino_acid_mol.GetMol())
    )
    terminal_C = get_atom_Idx_by_map_num(combined, mapped_terminal_C)
    terminal_N = get_atom_Idx_by_map_num(combined, mapped_terminal_N)
    combined.AddBond(
        terminal_C, terminal_N, order=Chem.rdchem.BondType.SINGLE
    )  # 组合的标数会位移
    return combined.GetMol()


def get_maps2idx_dict(mol: rdchem.Mol) -> Dict:
    return {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}


class Peptide:
    def __init__(
        self,
        sequence: str,
        methods: Literal["linear", "cyclic", "thioether", "thrombin"] = "linear",
        force_field: bool = False,
        num_confs: int = 10,
    ):
        if len(sequence) < 1:
            raise ValueError("Peptide sequence must have a length larger than zero.")

        # methods for post processing
        if methods not in ["linear", "cyclic", "thioether", "thrombin"]:
            raise ValueError(
                "Invalid method. Allowed values: 'linear', 'cyclic', 'thioether', 'thrombin'."
            )
        self.sequence = sequence
        self.methods = methods

        # Force Field
        self.force_field = force_field
        self.num_confs = num_confs

        # Initialize structure as None, calculate it when needed
        self.mol = None
        self.get_structure()

    def __repr__(self):
        return self.sequence

    def __str__(self):
        return repr(self)

    def __len__(self):
        if self.methods == "thrombin":
            return len(self.sequence.rsplit("-A0").split("-"))
        else:
            return len(self.sequence)

    @property
    def num_atoms(self) -> int:
        assert self.mol is not None
        return self.mol.GetNumAtoms()

    @property
    def smiles(self):
        # 这个保留立体信息以及正交很关键
        return Chem.MolToSmiles(self.mol, isomericSmiles=True, canonical=True)

    def get_s_idx(self, atoms):
        """start from the tail to obtain the exact S"""
        _atoms = [atom for atom in atoms]
        for atom in _atoms[-11:]:
            symbol = atom.GetSymbol()
            if symbol == "S":
                return atom.GetIdx()

    def single_conf_gen_MMFF(tgt_mol, num_confs=1000, seed=42):
        mol = deepcopy(tgt_mol)
        mol = Chem.AddHs(mol)
        allconformers = Chem.EmbedMultipleConfs(
            mol, numConfs=num_confs, randomSeed=seed, clearConfs=True
        )

        sz = len(allconformers)
        for i in range(sz):
            try:
                Chem.MMFFOptimizeMolecule(mol, confId=i)
            except:
                continue
        mol = Chem.RemoveHs(mol)
        return mol

    def find_terminals_for_thioether(
        self, mol, amino_acid_idx=None
    ) -> Tuple[Optional[int], Optional[int]]:
        # Handle the case when amino_acid_idx is None
        atoms = [x for x in mol.GetAtoms() if x.GetIdx() in amino_acid_idx]

        terminal_C = None
        terminal_N = None

        # Finding terminal C
        for atom in atoms:
            if atom.GetSymbol() == "C" and atom.GetTotalDegree() == 3:
                neighbors_idx = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
                oxygen_count = sum(
                    1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == "O"
                )
                outside_amino_acid = not set(neighbors_idx).issubset(amino_acid_idx)
                if outside_amino_acid and oxygen_count == 1:
                    terminal_C = atom.GetIdx()
                    break

        # Finding terminal N
        if terminal_C:
            for atom in atoms:
                if atom.GetSymbol() == "N" and atom.GetTotalNumHs() == 1:
                    path = Chem.rdmolops.GetShortestPath(mol, atom.GetIdx(), terminal_C)
                    if len(path) == 3:
                        terminal_N = atom.GetIdx()
                        break

        return terminal_C, terminal_N

    def find_terminals_for_linear(
        self,
        mol,
        amino_acid_idx=None,
        is_N_terminal=False,
        is_C_terminal=False,
        if_P=False,
    ) -> Tuple[Optional[int], Optional[int]]:
        # Handle the case when amino_acid_idx is None
        assert not (
            is_N_terminal and is_C_terminal
        ), "is_N_terminal and is_C_terminal cannot both be True at the same time."
        if if_P:
            P_N = 1
        else:
            P_N = 0
        atoms = [x for x in mol.GetAtoms() if x.GetIdx() in amino_acid_idx]

        terminal_C = None
        terminal_N = None

        if is_N_terminal:
            # Finding terminal C
            for atom in atoms:
                if atom.GetSymbol() == "C" and atom.GetTotalDegree() == 3:
                    neighbors_idx = [
                        neighbor.GetIdx() for neighbor in atom.GetNeighbors()
                    ]
                    oxygen_count = sum(
                        1
                        for neighbor in atom.GetNeighbors()
                        if neighbor.GetSymbol() == "O"
                    )
                    outside_amino_acid = not set(neighbors_idx).issubset(amino_acid_idx)
                    if outside_amino_acid and oxygen_count == 1:
                        terminal_C = atom.GetIdx()
                        break
            # Finding terminal N
            if terminal_C:
                for atom in atoms:
                    if atom.GetSymbol() == "N" and atom.GetTotalNumHs() == (2 - P_N):
                        path = Chem.rdmolops.GetShortestPath(
                            mol, atom.GetIdx(), terminal_C
                        )
                        if len(path) == 3:
                            terminal_N = atom.GetIdx()
                            break
            return terminal_C, terminal_N

        if is_C_terminal:
            # Finding terminal C
            for atom in atoms:
                if atom.GetSymbol() == "C" and atom.GetTotalDegree() == 3:
                    neighbors_idx = [
                        neighbor.GetIdx() for neighbor in atom.GetNeighbors()
                    ]
                    oxygen_count = sum(
                        1
                        for neighbor in atom.GetNeighbors()
                        if neighbor.GetSymbol() == "O"
                    )
                    outside_amino_acid = set(neighbors_idx).issubset(amino_acid_idx)
                    if outside_amino_acid and oxygen_count == 2:
                        terminal_C = atom.GetIdx()
                        break
            # Finding terminal N
            for atom in atoms:
                if terminal_C:
                    if atom.GetSymbol() == "N" and atom.GetTotalNumHs() == (1 - P_N):
                        path = Chem.rdmolops.GetShortestPath(
                            mol, atom.GetIdx(), terminal_C
                        )
                        if len(path) == 3:
                            terminal_N = atom.GetIdx()
                            break
            return terminal_C, terminal_N

        # Finding terminal C
        for atom in atoms:
            if atom.GetSymbol() == "C" and atom.GetTotalDegree() == 3:
                neighbors_idx = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
                oxygen_count = sum(
                    1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == "O"
                )
                outside_amino_acid = not set(neighbors_idx).issubset(amino_acid_idx)
                if outside_amino_acid and oxygen_count == 1:
                    terminal_C = atom.GetIdx()
                    break
        # Finding terminal N
        if terminal_C:
            for atom in atoms:
                if atom.GetSymbol() == "N" and atom.GetTotalNumHs() == (1 - P_N):
                    path = Chem.rdmolops.GetShortestPath(mol, atom.GetIdx(), terminal_C)
                    if len(path) == 3:
                        terminal_N = atom.GetIdx()
                        break
        return terminal_C, terminal_N

    def get_structure(self) -> List[List[int]]:
        """获取子结构"""
        self.mol = self.get_mol()
        assert self.mol is not None

        if self.methods == "linear":
            amino_acids = []
            structure = []
            substructre = []
            added_ResidueNumber = set()

            for i, atom in enumerate(self.mol.GetAtoms()):
                info = atom.GetMonomerInfo()
                residue_number = info.GetResidueNumber()
                name = info.GetResidueName()

                if residue_number not in added_ResidueNumber:
                    amino_acids.append(name)
                    added_ResidueNumber.add(residue_number)
                    if residue_number == 1:
                        pass
                    else:
                        structure.append(substructre)
                        substructre = []

                substructre.append(atom.GetIdx())
            structure.append(substructre)  # 添加最后一个

            begins = []
            ends = []
            # Single pass loop to gather all the necessary information
            for i, substructure in enumerate(structure):
                if i == 0:
                    terminal_C, terminal_N = self.find_terminals_for_linear(
                        self.mol,
                        substructure,
                        is_N_terminal=True,
                        if_P=(amino_acids[i] == "PRO"),
                    )
                    begins.append(terminal_C)

                elif i == len(structure) - 1:
                    terminal_C, terminal_N = self.find_terminals_for_linear(
                        self.mol,
                        substructure,
                        is_C_terminal=True,
                        if_P=(amino_acids[i] == "PRO"),
                    )
                    ends.append(terminal_N)

                else:
                    terminal_C, terminal_N = self.find_terminals_for_linear(
                        self.mol, substructure, if_P=(amino_acids[i] == "PRO")
                    )
                    begins.append(terminal_C)
                    ends.append(terminal_N)

            structure_bond = [[a, b] for a, b in zip(begins, ends)]
            self.structure = deepcopy(structure)
            self.amino_acids = deepcopy(amino_acids)
            self.structure_bond = deepcopy(structure_bond)

        elif self.methods == "thrombin":
            self.structure = deepcopy(self.reaction.structure)
            self.structure_bond = deepcopy(self.reaction.structure_bond)
            self.amino_acids = deepcopy(self.reaction.amino_acids)

        elif self.methods == "cyclic":
            pass

        elif self.methods == "thioether":
            last_residue_number = -1  # start with -1
            last_name = "acY"  # N terminal modification

            structure = []
            substructre = []
            name = [last_name]

            for i, atom in enumerate(self.mol.GetAtoms()):
                info = atom.GetMonomerInfo()
                this_residue_number = info.GetResidueNumber()
                this_name = info.GetResidueName()

                if this_residue_number != last_residue_number:
                    last_residue_number = this_residue_number
                    last_name = this_name

                    structure.append(substructre)
                    name.append(this_name)

                    substructre = []

                substructre.append(atom.GetIdx())
            structure.append([i])  # add NH2 to the tail G
            mol_seq = Chem.MolToSequence(self.mol)

            assert (
                len(mol_seq) == len(name) - 2 == len(structure) - 2
            )  # 一个是 ACE, 另一个是 NH2

            begins = [self.begin_index]
            ends = [self.end_index]

            # Single pass loop to gather all the necessary information
            for i, substructure in enumerate(structure):
                terminal_info = self.find_terminals_for_thioether(
                    self.mol, substructure
                )
                if 0 < i < len(structure) - 2:
                    begins.append(terminal_info[0])
                if 1 < i < len(structure) - 1:
                    ends.append(terminal_info[1])

            # Use list comprehension to create structure_bonds
            self.structure_bond = [[a, b] for a, b in zip(begins, ends)]

            # emerge the modification function
            # return structure, name
            structure[0:2] = [structure[0] + structure[1]]
            structure[-2:] = [structure[-2] + structure[-1]]

            self.structure = deepcopy(structure)
            self.amino_acids = deepcopy(name[1:-1])

    def get_mol(self) -> rdchem.Mol:
        if self.methods == "linear":
            mol = Chem.MolFromSequence(self.sequence)
            self.begin_index = 0  # 线肽开始的位置是 0
            self.end_index = len(mol.GetAtoms()) - 1  # 结束的位置是 长度-1
        elif self.methods == "thrombin":
            self.reaction = Reaction(self.sequence)
            self.reaction()
            mol = self.reaction.product
        elif self.methods == "cyclic":
            pass
        elif self.methods == "thioether":
            HELM_str = (
                "PEPTIDE1{[ac].Y." + ".".join(self.sequence) + ".C.G.[am]}$$$$V2.0"
            )
            mol = Chem.MolFromHELM(HELM_str)
            atoms = mol.GetAtoms()
            begin_index = 2
            end_index = self.get_s_idx(atoms)
            mw = Chem.RWMol(mol)
            mw.AddBond(begin_index, end_index, Chem.BondType.SINGLE)
            mol = mw.GetMol()
            Chem.SanitizeMol(mol)

            # 定义成环的两个位置
            self.begin_index = begin_index
            self.end_index = end_index

        # Generates Conformations
        if self.force_field and self.num_confs:
            mol = self.single_conf_gen_MMFF(mol, num_confs=self.num_confs)

        return mol


def construct_RGCN_mol_graph_from_mol(mol, smask):
    g = dgl.DGLGraph()

    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    atoms_feature_all = []
    smask_list = []
    for i, atom in enumerate(mol.GetAtoms()):
        atom_feature = atom_features(atom)
        atoms_feature_all.append(atom_feature)
        if i in smask:
            smask_list.append(0)
        else:
            smask_list.append(1)
    g.ndata["node"] = torch.tensor(np.array(atoms_feature_all))
    g.ndata["smask"] = torch.tensor(smask_list).float()
    # Add edges
    src_list = []
    dst_list = []
    etype_feature_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        etype_feature = etype_features(bond)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
        etype_feature_all.append(etype_feature)
        etype_feature_all.append(etype_feature)

    g.add_edges(src_list, dst_list)
    g.edata["edge"] = torch.tensor(etype_feature_all)
    return g


def build_mol_graph_data(
    dataset_peptide,
    label_name,
    peptide_name,
    methods: Literal["linear", "cyclic", "thioether", "thrombin"] = "linear",
    max_workers: int = None,
):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_peptide[label_name]
    split_index = dataset_peptide["group"]
    peptideList = dataset_peptide[peptide_name]
    molecule_number = len(peptideList)

    with tqdm(enumerate(peptideList), total=molecule_number) as pbar:
        for i, _peptide in pbar:
            peptide = Peptide(_peptide, methods=methods)
            try:
                g_rgcn = construct_RGCN_mol_graph_from_mol(peptide.mol, smask=[])
                molecule = [
                    peptide.sequence,
                    peptide.smiles,
                    g_rgcn,
                    labels.loc[i],
                    split_index.loc[i],
                ]
                dataset_gnn.append(molecule)
                pbar.set_description(
                    "{}/{} molecule is transformed to mol graph! {} is transformed failed!".format(
                        i + 1, molecule_number, len(failed_molecule)
                    )
                )
            except:
                pbar.set_description(
                    "{} is transformed to mol graph failed!".format(peptide.smiles)
                )
                molecule_number = molecule_number - 1
                failed_molecule.append(peptide.smiles)
    print(
        "{}({}) is transformed to mol graph failed!".format(
            failed_molecule, len(failed_molecule)
        )
    )
    return dataset_gnn


# def process_molecule(i, _peptide, label, split_ind, methods):
#     peptide = Peptide(_peptide, methods=methods)
#     aa_mask = []
#     aa_name = []
#     molecules = []

#     for substructure, amino_acid in zip(peptide.structure, peptide.amino_acids):
#         aa_mask.append(substructure)
#         aa_name.append(amino_acid)
#     for j, aa_mask_j in enumerate(aa_mask):
#         try:
#             g_rgcn = construct_RGCN_mol_graph_from_mol(peptide.mol, smask=aa_mask_j)
#             molecule = [
#                 peptide.sequence,
#                 peptide.smiles,
#                 g_rgcn,
#                 label,
#                 split_ind,
#                 aa_mask_j,
#                 aa_name[j],
#             ]
#             molecules.append(molecule)
#         except Exception as e:
#             return (peptide.smiles, str(e))
#     return molecules

# def build_mol_graph_data_for_peptide(
#     dataset_peptide: pd.DataFrame,
#     label_name: str,
#     peptide_name: str,
#     methods: str = "linear",
#     max_workers: int = None,
# ):
#     """build mol graph for peptide (such as function group)

#     引入多线程

#     Args:
#         dataset_peptide (pd.DataFrame): _description_
#         label_name (str): _description_
#         peptide_name (str): _description_
#         methods (str, optional): _description_. Defaults to "linear".

#     Returns:
#         _type_: _description_
#     """

#     dataset_gnn = []
#     failed_molecule = []
#     labels = dataset_peptide[label_name]
#     split_index = dataset_peptide["group"]
#     peptideList = dataset_peptide[peptide_name]
#     molecule_number = len(peptideList)
#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#     # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = []
#         for i, _peptide in enumerate(peptideList):
#             futures.append(executor.submit(process_molecule, i, _peptide, labels.loc[i], split_index.loc[i], methods))

#         for future in tqdm(concurrent.futures.as_completed(futures), total=molecule_number):
#             result = future.result()
#             if isinstance(result, tuple):
#                 failed_molecule.append(result)
#             else:
#                 dataset_gnn.extend(result)

#     print("{}({}) is transformed to mol graph failed!".format(failed_molecule, len(failed_molecule)))
#     return dataset_gnn


def build_mol_graph_data_for_peptide(
    dataset_peptide: pd.DataFrame,
    label_name: str,
    peptide_name: str,
    methods: str = "linear",
    max_workers: int = None,
):
    """build mol graph for peptide (such as function group)

    引入多线程

    Args:
        dataset_peptide (pd.DataFrame): _description_
        label_name (str): _description_
        peptide_name (str): _description_
        methods (str, optional): _description_. Defaults to "linear".

    Returns:
        _type_: _description_
    """

    dataset_gnn = []
    failed_molecule = []
    labels = dataset_peptide[label_name]
    split_index = dataset_peptide["group"]
    peptideList = dataset_peptide[peptide_name]
    molecule_number = len(peptideList)

    with tqdm(enumerate(peptideList), total=molecule_number) as pbar:
        for i, _peptide in pbar:
            peptide = Peptide(_peptide, methods=methods)
            aa_mask = []
            aa_name = []

            for substructure, amino_acid in zip(peptide.structure, peptide.amino_acids):
                aa_mask.append(substructure)
                aa_name.append(amino_acid)
            for j, aa_mask_j in enumerate(aa_mask):
                try:
                    g_rgcn = construct_RGCN_mol_graph_from_mol(
                        peptide.mol, smask=aa_mask_j
                    )
                    molecule = [
                        peptide.sequence,
                        peptide.smiles,
                        g_rgcn,
                        labels.loc[i],
                        split_index.loc[i],
                        aa_mask_j,
                        aa_name[j],
                    ]
                    dataset_gnn.append(molecule)
                    pbar.set_description(
                        "{}/{}, {}/{} molecule is transformed to mol graph! {} is transformed failed!".format(
                            j + 1,
                            len(aa_mask),
                            i + 1,
                            molecule_number,
                            len(failed_molecule),
                        )
                    )
                except Exception as e:
                    pbar.set_description(
                        "{} is transformed to mol graph failed! Since: {}".format(
                            peptide.amino_acids, e
                        )
                    )
                    molecule_number = molecule_number - 1
                    failed_molecule.append(peptide.smiles)
    print(
        "{}({}) is transformed to mol graph failed!".format(
            failed_molecule, len(failed_molecule)
        )
    )
    return dataset_gnn


def make_fake_cv(path):
    for fmat in ["_prediction.csv", "_smask_index.npy"]:
        for seed in range(10):
            src = path + fmat
            tgt = (
                f"{'_'.join(path.split('_')[:2])}_{seed}_{'_'.join(path.split('_')[2:])}"
                + fmat
            )
            print(f"copy file from {src} to {tgt}")
            shutil.copyfile(src, tgt)


def load_graph_from_csv_bin_for_splited(
    bin_path="g_atom.bin",
    group_path="g_group.csv",
    smask_path=None,
    classification=True,
    random_shuffle=True,
    seed=2023,
):
    data = pd.read_csv(group_path)
    sequence = data.sequence.values
    smiles = data.smiles.values
    group = data.group.to_list()

    # load substructure name
    if "sub_name" in data.columns.tolist():
        sub_name = data["sub_name"]
    else:
        sub_name = ["noname" for x in group]

    if random_shuffle:
        random.seed(seed)
        random.shuffle(group)
    homog, detailed_information = load_graphs(bin_path)
    labels = detailed_information["labels"]

    # load smask
    if smask_path is None:
        smask = [-1 for x in range(len(group))]
    else:
        smask = np.load(smask_path, allow_pickle=True)

    # calculate not_use index
    train_index = []
    val_index = []
    test_index = []
    for index, group_index in enumerate(group):
        if group_index == "train":
            train_index.append(index)
        if group_index == "valid":
            val_index.append(index)
        if group_index == "test":
            test_index.append(index)

    task_number = 1
    train_set = []
    val_set = []
    test_set = []

    for i in train_index:
        molecule = [sequence[i], smiles[i], homog[i], labels[i], smask[i], sub_name[i]]
        train_set.append(molecule)

    for i in val_index:
        molecule = [sequence[i], smiles[i], homog[i], labels[i], smask[i], sub_name[i]]
        val_set.append(molecule)

    for i in test_index:
        molecule = [sequence[i], smiles[i], homog[i], labels[i], smask[i], sub_name[i]]
        test_set.append(molecule)

    print(len(train_set), len(val_set), len(test_set), task_number)
    return train_set, val_set, test_set, task_number


def collate_molgraphs(data):
    peptide, smiles, g_rgcn, labels, smask, sub_name = map(list, zip(*data))
    rgcn_bg = dgl.batch(g_rgcn)
    labels = torch.tensor(labels)
    return peptide, smiles, rgcn_bg, labels, smask, sub_name


def pos_weight(train_set):
    """get the weight of positives in specified dataset"""
    sequence, smiles, g_rgcn, labels, smask, sub_name = map(list, zip(*train_set))
    labels = np.array(labels)
    task_pos_weight_list = []
    num_pos = 0
    num_neg = 0
    for i in labels:
        if i == 1:
            num_pos = num_pos + 1
        if i == 0:
            num_neg = num_neg + 1
    weight = num_neg / (num_pos + 0.00000001)
    task_pos_weight_list.append(weight)
    task_pos_weight = torch.tensor(task_pos_weight_list)
    return task_pos_weight


class EarlyStopping(object):
    """Early stop performing
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
    patience : int
        Number of epochs to wait before early stop
        if the metric stops getting improved
    taskname : str or None
        Filename for storing the model checkpoint
    """

    def __init__(
        self,
        pretrained_model="Null_early_stop.pth",
        mode="higher",
        patience=10,
        filename=None,
        task_name="None",
        sub_type="mol",
        seed=None,
        former_task_name="None",
    ):
        if filename is None:
            task_name = task_name
            if seed is None:
                filename = f"./model/{task_name}_{sub_type}_early_stop.pth"
            else:
                filename = f"./model/{task_name}_{sub_type}_{seed}_early_stop.pth"
        if seed is not None:
            former_filename = (
                f"./model/{former_task_name}_{sub_type}_{seed}_early_stop.pth"
            )
        else:
            former_filename = f"./model/{former_task_name}_{sub_type}_early_stop.pth"

        assert mode in ["higher", "lower"]
        self.mode = mode
        if self.mode == "higher":
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.former_filename = former_filename
        self.best_score = None
        self.early_stop = False
        self.pretrained_model = "./model/" + pretrained_model

    def _check_higher(self, score, prev_best_score):
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        return score < prev_best_score

    def step(self, score, model, dist=False):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, dist=dist)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model, dist=dist)
            self.counter = 0
        else:
            self.counter += 1
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def nosave_step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._check(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model, dist=False):
        """Saves model when the metric on the validation set gets improved."""
        if dist:
            torch.save({"model_state_dict": model.module.state_dict()}, self.filename)
        else:
            torch.save({"model_state_dict": model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        """Load model saved with early stopping."""
        print(f"load model from {self.filename}")
        state_dict = torch.load(self.filename)
        model.load_state_dict(state_dict["model_state_dict"])

    def load_former_model(self, model):
        """Load model saved with early stopping."""
        model.load_state_dict(torch.load(self.former_filename)["model_state_dict"])
        # model.load_state_dict(torch.load(self.former_filename, map_location=torch.device('cpu'))['model_state_dict'])

    def load_pretrained_model(self, model):
        """Load pretrained model."""
        model.load_state_dict(torch.load(self.pretrained_model)["model_state_dict"])


def run_a_train_epoch(
    args, model, data_loader, loss_criterion, optimizer, scheduler=None, use_seq=False
):
    model.train()
    train_meter = Meter()
    total_loss = 0
    n_mol = 0
    for batch_id, batch_data in tqdm(enumerate(data_loader), total=len(list(data_loader))):
        sequence, smiles, rgcn_bg, labels, smask_idx, sub_name = batch_data

        # Move data to device once
        rgcn_bg = rgcn_bg.to(args["device"])
        labels = labels.unsqueeze(dim=1).float().to(args["device"])
        rgcn_node_feats = (
            rgcn_bg.ndata.pop(args["node_data_field"]).float().to(args["device"])
        )
        rgcn_edge_feats = (
            rgcn_bg.edata.pop(args["edge_data_field"]).long().to(args["device"])
        )
        smask_feats = (
            rgcn_bg.ndata.pop(args["substructure_mask"])
            .unsqueeze(dim=1)
            .float()
            .to(args["device"])
        )

        # Compute predictions and loss
        if use_seq:
            preds, weight = model(
                rgcn_bg, rgcn_node_feats, rgcn_edge_feats, smask_feats, sequence
            )
        else:
            preds, weight = model(
                rgcn_bg, rgcn_node_feats, rgcn_edge_feats, smask_feats
            )
        loss = (loss_criterion(preds, labels)).mean()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item() * len(smiles)
        n_mol += len(smiles)
        train_meter.update(preds, labels)

        del (
            sequence,
            labels,
            rgcn_bg,
            rgcn_edge_feats,
            rgcn_node_feats,
            loss,
            smask_idx,
            sub_name,
            smiles,
            smask_feats,
            preds,
            weight,
        )

        torch.cuda.empty_cache()

    train_score = round(train_meter.compute_metric(args["metric_name"]), 4)
    average_loss = total_loss / n_mol

    return train_score, average_loss


def run_an_eval_epoch(
    args,
    model,
    data_loader,
    loss_criterion,
    out_path,
    stage: str = "train",
    use_seq=False,
):
    print(f"{stage} set evaling:")
    model.eval()

    sequence_list = []
    smiles_list = []
    eval_meter = Meter()
    g_list = []
    total_loss = 0
    n_mol = 0
    smask_idx_list = []
    sub_name_list = []

    if isinstance(out_path, Path):
        out_path = str(out_path)

    with torch.no_grad():
        for batch_id, batch_data in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            sequence, smiles, rgcn_bg, labels, smask_idx, sub_name = batch_data

            # Move data to device once
            device = args["device"]

            rgcn_bg = rgcn_bg.to(device)
            labels = labels.unsqueeze(dim=1).float().to(device)
            rgcn_node_feats = (
                rgcn_bg.ndata.pop(args["node_data_field"]).float().to(device)
            )
            rgcn_edge_feats = (
                rgcn_bg.edata.pop(args["edge_data_field"]).long().to(device)
            )
            smask_feats = (
                rgcn_bg.ndata.pop(args["substructure_mask"])
                .unsqueeze(dim=1)
                .float()
                .to(device)
            )

            # Compute predictions and loss
            if use_seq:
                preds, weight = model(
                    rgcn_bg, rgcn_node_feats, rgcn_edge_feats, smask_feats, sequence
                )
            else:
                preds, weight = model(
                    rgcn_bg, rgcn_node_feats, rgcn_edge_feats, smask_feats
                )
            loss = (loss_criterion(preds, labels)).mean()

            sequence_list += sequence
            smask_idx_list += smask_idx
            sub_name_list += sub_name

            total_loss += loss.item() * len(smiles)
            n_mol += len(smiles)

            if out_path is not None:
                rgcn_bg.ndata["weight"] = weight
                rgcn_bg.edata["edge"] = rgcn_edge_feats
                g_list += dgl.unbatch(rgcn_bg)

            eval_meter.update(preds, labels)
            smiles_list += smiles

            del (
                sequence,
                labels,
                rgcn_bg,
                rgcn_edge_feats,
                rgcn_node_feats,
                loss,
                smask_idx,
                sub_name,
                smiles,
                smask_feats,
                preds,
                weight,
            )
            torch.cuda.empty_cache()

        average_loss = total_loss / n_mol

    prediction_pd = pd.DataFrame()
    y_true, y_pred = eval_meter.compute_metric("return_pred_true")
    y_true = y_true.squeeze().numpy()

    if args["classification"]:
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.squeeze().numpy()
    else:
        y_pred = y_pred.squeeze().numpy()

    y_true_list = y_true.tolist()
    y_pred_list = y_pred.tolist()

    # save prediction
    prediction_pd["sequence"] = sequence_list
    prediction_pd["smiles"] = smiles_list
    prediction_pd["label"] = y_true_list
    prediction_pd["pred"] = y_pred_list
    prediction_pd["sub_name"] = sub_name_list

    if out_path is not None:
        np.save(
            out_path + "_smask_index.npy",
            np.array(smask_idx_list, dtype=object),
            allow_pickle=True,
        )
        prediction_pd.to_csv(out_path + "_prediction.csv", index=False)

    if args["classification"]:
        y_pred_label = [1 if x >= 0.5 else 0 for x in y_pred_list]
        accuracy = round(metrics.accuracy_score(y_true_list, y_pred_label), 4)
        mcc = round(metrics.matthews_corrcoef(y_true_list, y_pred_label), 4)
        se, sp = sesp_score(y_true_list, y_pred_label)
        pre, rec, f1, sup = metrics.precision_recall_fscore_support(
            y_true_list, y_pred_label, zero_division=0
        )
        f1 = round(f1[1], 4)
        rec = round(rec[1], 4)
        pre = round(pre[1], 4)
        err = round(1 - accuracy, 4)
        result = [accuracy, se, sp, f1, pre, rec, err, mcc]
        return result, average_loss
    else:
        r2 = round(metrics.r2_score(y_true_list, y_pred_list), 4)
        mae = round(metrics.mean_absolute_error(y_true_list, y_pred_list), 4)
        rmse = round(metrics.mean_squared_error(y_true_list, y_pred_list) ** 0.5, 4)
        result = [r2, mae, rmse]
        return result, average_loss


class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""

    def __init__(self):
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true):
        """Update for the result of an iteration
        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        mask : float32 tensor
            Mask for indicating the existence of ground
            truth labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())

    def accuracy_score(self):
        """Compute accuracy score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """

        y_pred = torch.cat(self.y_pred, dim=0)
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.numpy()
        y_pred_label = np.array([pro2label(x) for x in y_pred])
        y_true = torch.cat(self.y_true, dim=0).numpy()
        scores = round(metrics.accuracy_score(y_true, y_pred_label), 4)
        return scores

    def r2(self):
        """Compute r2 score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        y_pred = torch.cat(self.y_pred, dim=0).numpy()
        y_true = torch.cat(self.y_true, dim=0).numpy()

        scores = round(metrics.r2_score(y_true, y_pred), 4)
        return scores

    def mse(self):
        y_pred = torch.cat(self.y_pred, dim=0).numpy()
        y_true = torch.cat(self.y_true, dim=0).numpy()
        scores = round(metrics.mean_squared_error(y_true, y_pred), 4)
        return scores

    def return_pred_true(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        return y_true, y_pred

    def compute_metric(self, metric_name):
        """Compute metric for each task.
        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.
        reduction : str
            Only comes into effect when the metric_name is l1_loss.
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        Returns
        -------
        list of float
            Metric value for each task
        """
        assert metric_name in [
            "accuracy",
            "r2",
            "mse",
            "return_pred_true",
        ], 'Expect metric name to be "roc_auc", "accuracy", "return_pred_true", got {}'.format(
            metric_name
        )
        if metric_name == "accuracy":
            return self.accuracy_score()
        if metric_name == "mse":
            return self.mse()
        if metric_name == "r2":
            return self.r2()
        if metric_name == "return_pred_true":
            return self.return_pred_true()


def sesp_score(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 1:
            tp = tp + 1
        if y_true[i] == y_pred[i] == 0:
            tn = tn + 1
        if y_true[i] == 0 and y_pred[i] == 1:
            fp = fp + 1
        if y_true[i] == 1 and y_pred[i] == 0:
            fn = fn + 1
    sensitivity = round(tp / (tp + fn + 0.0000001), 4)
    specificity = round(tn / (tn + fp + 0.0000001), 4)
    return sensitivity, specificity


def pro2label(x):
    if x < 0.5:
        return 0
    else:
        return 1


def add_params(yml_path: Union[str, Path], args: Dict, overwrite: bool = True):
    if isinstance(yml_path, str):
        path = Path(yml_path)
    else:
        path = yml_path

    # 检查路径是否存在
    try:
        if path.exists():
            with path.open() as f:
                config = yaml.safe_load(f)

            for k, v in config.items():
                if k in args:
                    if overwrite:
                        args[k] = v
                else:
                    args[k] = v
    except Exception as e:
        print(e)
        raise FileExistsError


def write_results(
    model,
    data_loader,
    data_set,
    loss_criterion,
    prediction_dir,
    seed,
    args,
    stage: str = "train",
    use_seq: bool = False,
):
    if seed is not None:
        out_path = (
            prediction_dir / f"{args['task_name']}_{args['sub_type']}_{seed}_{data_set}"
        )
    else:
        out_path = prediction_dir / f"{args['task_name']}_{args['sub_type']}_{data_set}"

    stop_list, _ = run_an_eval_epoch(
        args, model, data_loader, loss_criterion, out_path, stage, use_seq=use_seq
    )
    print(f"{data_set} results:", ",".join([str(i) for i in stop_list]))


def attr2tag(attr):
    return -1 if attr < 0 else 1


def get_seg(value, seg_type: Literal["bio", "del"] = "del") -> int:
    """
    Compute the segment value based on the provided segmentation type.

    :param value: The input value for which the segment is to be calculated.
    :param seg_type: The segmentation type ("bio" or "del").
    :return: The segment value.
    """
    if seg_type == "bio":
        # Convert value to -log10 scale, then floor it.
        return np.floor(-np.log10(value * 1e-9)).astype(np.int32)
    elif seg_type == "del":
        # Convert value to log10 scale, then floor it.
        return np.floor(np.log10(value)).astype(np.int32)
    else:
        raise ValueError(f"Unsupported seg_type: {seg_type}")


def get_label(value, threshold) -> int:
    return 1 if value < threshold else 0


def _store(data: Any, file_path: Path) -> None:
    """Store a Python object in a file using dill serialization.

    Args:
        data: The Python object to serialize.
        file_path: The path to the file where the object should be stored.
    """
    with file_path.open("wb") as file:
        dill.dump(data, file)


def _restore(file_path: Path) -> Any:
    """Restore a Python object from a file using dill serialization.

    Args:
        file_path: The path to the file from which the object should be restored.

    Returns:
        The deserialized Python object.
    """
    with file_path.open("rb") as file:
        return dill.load(file)


def store(data: Any, file_path: Union[str, Path], force: bool = False) -> None:
    """Store a Python object in a file with optional overwrite protection.

    Args:
        data: The Python object to serialize.
        file_path: The path to the file where the object should be stored.
        force: If True, will overwrite existing files without warning.

    Raises:
        FileExistsError: If the file already exists and force is not set to True.
    """
    path = Path(file_path)

    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Set force=True to overwrite.")

    _store(data, path)


def restore(file_path: Union[str, Path]) -> Any:
    """Restore a Python object from a file.

    Args:
        file_path: The path to the file from which the object should be restored.

    Returns:
        The deserialized Python object if the file exists, None otherwise.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    return _restore(path)


class Reaction:
    def __init__(self, reactants: Union[List[str], str]):
        """
        Initialize reaction class

        Args:
        reactants (List[str]): List of strings of fragment names

        """
        # Initialize fragments
        if isinstance(reactants, str):
            self._reactants = reactants.split("-")
        elif isinstance(reactants, list):
            self._reactants = reactants

        self.starter, self.frag1, self.frag2, self.ender, self.linker, self.acid = (
            getattr(thrombin, frag)() for frag in self._reactants
        )
        self.reactants = [
            self.starter,
            self.frag1,
            self.frag2,
            self.ender,
            self.linker,
            self.acid,
        ]

        # Remap atom indices
        self.remap_Idx()

        # Combine molecules
        self.get_combined_mol()

        # Success flag
        self.success = False

    def remap_Idx(self) -> None:
        """
        Remap atom indices in the reaction

        """
        mapped_molecules = []
        base_index = 0
        init_atom_map_nums(self.reactants[0].mol)
        current_index_range = range(
            base_index, base_index + self.reactants[0].num_atoms
        )
        mapped_molecules.append(list(current_index_range))
        base_index += self.reactants[0].num_atoms

        for i in range(1, len(self.reactants)):
            if self.reactants[i].mol:
                set_atom_map_nums(
                    self.reactants[i].mol,
                    [base_index + j for j in range(self.reactants[i].num_atoms)],
                )
                self.reactants[i].remap_Idx(base_index)
                current_index_range = range(
                    base_index, base_index + self.reactants[i].num_atoms
                )
                mapped_molecules.append(list(current_index_range))
                base_index += self.reactants[i].num_atoms

        self.mapped_molecules = mapped_molecules

    def get_combined_mol(self) -> Chem.rdchem.RWMol:
        """
        Get the combined molecule of the reaction

        Returns:
        Chem.rdchem.RWMol: The combined molecule

        """
        combined_mol = self.reactants[0].mol
        for i in range(1, len(self.reactants)):
            if self.reactants[i].mol:
                combined_mol = Chem.CombineMols(combined_mol, self.reactants[i].mol)
        self.combined_mol = Chem.RWMol(combined_mol)

    def remove_atoms(self, Idx: Union[List[int], int]) -> None:
        if isinstance(Idx, int):
            self.combined_mol.RemoveAtom(
                get_atom_Idx_by_map_num(self.combined_mol, Idx)
            )
        elif isinstance(Idx, list):
            for i in Idx:
                self.combined_mol.RemoveAtom(
                    get_atom_Idx_by_map_num(self.combined_mol, i)
                )

    def add_bond(self, Idx1: int, Idx2: int) -> None:
        """
        Add a bond between two atoms in the combined molecule

        Args:
        Idx1 (int): The index of the first atom
        Idx2 (int): The index of the second atom

        """
        self.combined_mol.AddBond(
            get_atom_Idx_by_map_num(self.combined_mol, Idx1),
            get_atom_Idx_by_map_num(self.combined_mol, Idx2),
            order=Chem.rdchem.BondType.SINGLE,
        )

    def add_bonds(self, Idx_list: List[Tuple[int, int]]) -> None:
        """
        Add bonds between two atoms in the combined molecule

        Args:
        Idx_list (List[Tuple[int, int]]): A list of tuples of atom indices

        """
        try:
            for Idx1, Idx2 in Idx_list:
                self.combined_mol.AddBond(
                    get_atom_Idx_by_map_num(self.combined_mol, Idx1),
                    get_atom_Idx_by_map_num(self.combined_mol, Idx2),
                    order=Chem.rdchem.BondType.SINGLE,
                )
        except Exception as e:
            logger.exception(f"Error Idx for bond addition since {e}")

    def show_combined_mol(self, *args, **kwargs) -> None:
        """
        Show the combined molecule

        """
        show_atom_map_nums(self.combined_mol, *args, **kwargs)

    @property
    def format(self) -> Optional[str]:
        """
        Get the format of the reaction

        Returns:
        Optional[str]: The format of the reaction

        """
        if "Cysteine" in self._reactants:  # 不同的 starter
            if self.frag1.group == "backbone" and self.frag2.group == "side_chain":
                return "format1"
            elif self.frag1.group == "side_chain" and self.frag2.group == "backbone":
                return "format2"
            else:
                return None
        elif "ThiomalicAcid" in self._reactants:  # 不同的 starter
            if self.frag1.group == "diamino_acid":
                return "format3"
            elif self.frag2.group == "diamino_acid":
                return "format4"
            else:
                return None
        else:
            return None

    def __call__(
        self, verbose: bool = False, *args, **kwargs
    ) -> Optional[Chem.rdchem.Mol]:
        if verbose:
            self.show_combined_mol()

        if not self.success:
            assert len(self._reactants) == 6, "Six fragments are required."
            logger.debug(
                f"run peptide {'-'.join(self._reactants)} reaction at {self.format}"
            )
            self.remove_atoms(self.dropped_atoms)
            self.add_bonds(self.structure_bond)

        if verbose:
            self.show_combined_mol()
        self.success = True
        return self.product

    @property
    def structure_bond(self) -> List[List[int]]:
        """
        List of lists of atom indices representing the bonds structure of the reaction

        """
        bonds_list = [
            [self.starter.C_Idx, self.frag1.N_Idx],
            [self.frag1.C_Idx, self.frag2.N_Idx],
            [self.frag2.C_Idx, self.ender.N_Idx],
            [self.linker.right, self.starter.S_Idx],
            [self.ender.S_Idx, self.linker.left],
        ]
        if self.format in ("format1", "format2"):
            if self.acid.mol:
                bonds_list.append([self.acid.C_Idx, self.starter.N_Idx])
        elif self.format in ("format3", "format4"):
            if self.acid.mol:
                if self.frag1.group == "diamino_acid":
                    bonds_list.append([self.acid.C_Idx, self.frag1.N2_Idx])
                elif self.frag2.group == "diamino_acid":
                    bonds_list.append([self.acid.C_Idx, self.frag2.N2_Idx])
        return bonds_list

    @property
    def dropped_atoms(self) -> List[int]:
        atom_list = [self.starter.O_Idx, self.frag1.O_Idx, self.frag2.O_Idx]
        if self.acid.mol:
            atom_list.append(self.acid.O_Idx)
        return atom_list

    @property
    def structure(self) -> List[List[int]]:
        """
        List of lists of atom indices representing the structure of the reaction

        """
        structure = []
        for molecule in self.mapped_molecules:
            substructure = []
            for idx in molecule:
                atom_idx = get_atom_Idx_by_map_num(self.combined_mol, idx)
                if atom_idx is not None:
                    substructure.append(atom_idx)
            structure.append(substructure)
        x, y = (
            len(list(chain.from_iterable(structure))),
            self.combined_mol.GetNumAtoms(),
        )
        assert x == y, f"Len structure {x} unequal to len combined_mol {y}"
        return structure

    @property
    def product(self) -> Optional[Chem.rdchem.Mol]:
        """
        Get the product molecule

        Returns:
        Optional[Chem.rdchem.Mol]: The product molecule

        """
        if self.success:
            return self.combined_mol.GetMol()
        else:
            return None

    @property
    def smiles(self) -> Optional[str]:
        """
        Get the SMILES string of the product molecule

        Returns:
        Optional[str]: The SMILES string

        """
        if self.success and self.product is not None:
            return Chem.MolToSmiles(self.product)
        else:
            return None

    @property
    def amino_acids(self) -> List[str]:
        return self._reactants[:-1] if "A0" in self._reactants else self._reactants
