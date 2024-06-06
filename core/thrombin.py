from rdkit import Chem

import core.utils


def check_Idx(mol, Idx, symbol):
    _atom = core.utils.get_atom_by_map_num(mol, Idx)
    assert _atom.GetSymbol() == symbol


##################### Base Amino Acids ################
class BaseAminoAcid:
    @property
    def num_atoms(self) -> int:
        return self.mol.GetNumAtoms()

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset
        self.N_Idx += offset

    def __init__(self, smiles: str = None) -> None:
        self.smiles = smiles


############### starter #######################
class Cysteine(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "O=C(O)[C@H](CS)N"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 1
        self.O_Idx = 2
        self.N_Idx = 6
        self.S_Idx = 5

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset
        self.N_Idx += offset
        self.S_Idx += offset


class ThiomalicAcid(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "O=C(O)CCS"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 1
        self.O_Idx = 2
        self.S_Idx = 5

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset
        self.S_Idx += offset


############### Side Chain #################
class X1(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "C[C@H](N)C(=O)O"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 3
        self.O_Idx = 5
        self.N_Idx = 2
        self.group = "side_chain"


class X2(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "CC(C)[C@H](N)C(=O)O"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 5
        self.O_Idx = 7
        self.N_Idx = 4
        self.group = "side_chain"


class X3(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "N[C@@H](CO)C(=O)O"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 4
        self.O_Idx = 6
        self.N_Idx = 0
        self.group = "side_chain"


class X4(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "N[C@@H](Cc1c[nH]cn1)C(=O)O"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 8
        self.O_Idx = 10
        self.N_Idx = 0
        self.group = "side_chain"


class X5(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "N[C@@H](Cc1ccccc1)C(=O)O"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 9
        self.O_Idx = 11
        self.N_Idx = 0
        self.group = "side_chain"


class X6(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 12
        self.O_Idx = 14
        self.N_Idx = 0
        self.group = "side_chain"


class X7(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "NC(=O)NCCC[C@H](N)C(=O)O"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 9
        self.O_Idx = 11
        self.N_Idx = 8
        self.group = "side_chain"


class X8(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "N[C@@H](Cc1ccccn1)C(=O)O"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 9
        self.O_Idx = 11
        self.N_Idx = 0
        self.group = "side_chain"


class X9(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "N[C@@H](CC1CCCCC1)C(=O)O"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 9
        self.O_Idx = 11
        self.N_Idx = 0
        self.group = "side_chain"


class X10(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "O=C(O)[C@@H]1CCCN1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 1
        self.O_Idx = 2
        self.N_Idx = 7
        self.group = "side_chain"


###################### Backbone ######################


class X21(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "O=C(O)[C@@H]1CCCN1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 1
        self.O_Idx = 2
        self.N_Idx = 7
        self.group = "backbone"


class X22(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "O=C(O)C[C@H]1CCCN1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 1
        self.O_Idx = 2
        self.N_Idx = 8
        self.group = "backbone"


class X23(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "NCc1cccc(C(=O)O)c1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 7
        self.O_Idx = 9
        self.N_Idx = 0
        self.group = "backbone"


class X24(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "O=C(O)[C@H]1CCCNC1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 1
        self.O_Idx = 0
        self.N_Idx = 7
        self.group = "backbone"


class X25(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "N[C@H]1CC[C@@H](C(=O)O)CC1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 5
        self.O_Idx = 7
        self.N_Idx = 0
        self.group = "backbone"


class X26(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "CNCCC(=O)O"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 4
        self.O_Idx = 6
        self.N_Idx = 1
        self.group = "backbone"


class X27(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "NCc1ccc(C(=O)O)cc1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 6
        self.O_Idx = 8
        self.N_Idx = 0
        self.group = "backbone"


class X28(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "NC1(C(=O)O)CCCC1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 2
        self.O_Idx = 4
        self.N_Idx = 0
        self.group = "backbone"


class X29(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "NCC1(C(=O)O)CC1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 3
        self.O_Idx = 5
        self.N_Idx = 0
        self.group = "backbone"


class X30(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "O=C(O)CC1CCNCC1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 1
        self.O_Idx = 2
        self.N_Idx = 7
        self.group = "backbone"


##################### Di-amino Acids #####################


class X11(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "NC1(C(=O)O)CCNCC1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 2
        self.O_Idx = 4
        self.N_Idx = 7
        self.N2_Idx = 0
        self.group = "diamino_acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset
        self.N_Idx += offset
        self.N2_Idx += offset


class X12(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "O=C(O)C1CNCCN1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 1
        self.O_Idx = 2
        self.N_Idx = 8
        self.N2_Idx = 5
        self.group = "diamino_acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset
        self.N_Idx += offset
        self.N2_Idx += offset


class X13(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "NC[C@H](N)C(=O)O"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 4
        self.O_Idx = 6
        self.N_Idx = 3
        self.N2_Idx = 0
        self.group = "diamino_acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset
        self.N_Idx += offset
        self.N2_Idx += offset


class X14(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "N[C@@H]1CN[C@H](C(=O)O)C1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 5
        self.O_Idx = 7
        self.N_Idx = 0
        self.N2_Idx = 3
        self.group = "diamino_acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset
        self.N_Idx += offset
        self.N2_Idx += offset


class X15(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "NCc1ccc(C[C@H](N)C(=O)O)cc1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 9
        self.O_Idx = 11
        self.N_Idx = 8
        self.N2_Idx = 0
        self.group = "diamino_acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset
        self.N_Idx += offset
        self.N2_Idx += offset


class X16(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "NCC[C@H](N)C(=O)O"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 5
        self.O_Idx = 7
        self.N_Idx = 4
        self.N2_Idx = 0
        self.group = "diamino_acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset
        self.N_Idx += offset
        self.N2_Idx += offset


class X17(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "O=C(O)C1CNCCN1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 1
        self.O_Idx = 2
        self.N_Idx = 5
        self.N2_Idx = 8
        self.group = "diamino_acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset
        self.N_Idx += offset
        self.N2_Idx += offset


class X18(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "NC[C@H](N)C(=O)O"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 4
        self.O_Idx = 6
        self.N_Idx = 0
        self.N2_Idx = 3
        self.group = "diamino_acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset
        self.N_Idx += offset
        self.N2_Idx += offset


class X19(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "N[C@@H]1CN[C@H](C(=O)O)C1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 5
        self.O_Idx = 7
        self.N_Idx = 3
        self.N2_Idx = 0
        self.group = "diamino_acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset
        self.N_Idx += offset
        self.N2_Idx += offset


class X20(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "NCc1ccc(CC(N)C(=O)O)cc1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 9
        self.O_Idx = 11
        self.N_Idx = 0
        self.N2_Idx = 8
        self.group = "diamino_acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset
        self.N_Idx += offset
        self.N2_Idx += offset


##################### Ender #####################


class Cysteamine(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "NCCS"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.N_Idx = 0
        self.S_Idx = 3
        self.group = "ender"

    def remap_Idx(self, offset) -> None:
        self.N_Idx += offset
        self.S_Idx += offset


#################### Linker ######################


class L1(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "Cc1cccc(C)n1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.left = 0
        self.right = 6
        self.group = "linker"

    def remap_Idx(self, offset) -> None:
        self.left += offset
        self.right += offset


class L2(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "CCS(=O)(=O)CC"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.left = 0
        self.right = 6
        self.group = "linker"

    def remap_Idx(self, offset) -> None:
        self.left += offset
        self.right += offset


class L3(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "Cc1ccc(C)cc1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.left = 0
        self.right = 5
        self.group = "linker"

    def remap_Idx(self, offset) -> None:
        self.left += offset
        self.right += offset


class L4(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "C/C=C/C"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.left = 0
        self.right = 3
        self.group = "linker"

    def remap_Idx(self, offset) -> None:
        self.left += offset
        self.right += offset


#################### Acid ####################


class A5(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "O=C(O)c1cccnc1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 1
        self.O_Idx = 2
        self.group = "acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset


class A6(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "O=C(O)C1CCOCC1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 1
        self.O_Idx = 2
        self.group = "acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset


class A7(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "CC(C)CC(=O)O"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 4
        self.O_Idx = 6
        self.group = "acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset


class A8(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "O=C(O)c1c[nH]c2ccccc12"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 1
        self.O_Idx = 2
        self.group = "acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset


class A9(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "O=C(O)c1cc(F)cc(F)c1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 1
        self.O_Idx = 2
        self.group = "acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset


class A10(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "COc1ccc(C(=O)O)cn1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 6
        self.O_Idx = 8
        self.group = "acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset


class A11(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "O=C(O)c1ccc(Cl)s1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 1
        self.O_Idx = 2
        self.group = "acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset


class A12(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "O=C(O)Cc1ccc(Cl)cc1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 1
        self.O_Idx = 2
        self.group = "acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset


class A13(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "Cc1cc(C(=O)O)no1"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 4
        self.O_Idx = 6
        self.group = "acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset


class A14(BaseAminoAcid):
    def __init__(self) -> None:
        self.smiles = "O=C(O)c2cccc1OC(F)(F)Oc12"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.C_Idx = 1
        self.O_Idx = 2
        self.group = "acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset


class A0(BaseAminoAcid):
    """No Acid"""

    def __init__(self) -> None:
        self.smiles = None
        self.mol = None
        self.C_Idx = 0
        self.O_Idx = 0
        self.group = "acid"

    def remap_Idx(self, offset) -> None:
        self.C_Idx += offset
        self.O_Idx += offset
