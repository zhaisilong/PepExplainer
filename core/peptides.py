from rdkit import Chem
from rdkit.Chem import rdchem


def cyclize_peptide(peptide: rdchem.Mol) -> rdchem.Mol:
    """线肽环化
    Args:
        peptide (rdchem.Mol): 多肽的 Mol

    Returns:
        rdchem.Mol: 环肽的 Mol
    """
    # 找到N端和C端的原子
    N_terminal_atom = [
        atom
        for atom in peptide.GetAtoms()
        if atom.GetSymbol() == "N" and atom.GetTotalNumHs() == 2
    ][0]
    C_terminal_atom = [
        atom
        for atom in peptide.GetAtoms()
        if atom.GetSymbol() == "C" and atom.GetTotalNumHs() == 2
    ][-1]
    print(f"N 端点 Idx: {N_terminal_atom.GetIdx()}")
    print(f"C 端点 Idx: {C_terminal_atom.GetIdx()}")

    # 使用编辑器添加键
    rw_peptide = Chem.RWMol(peptide)
    rw_peptide.AddBond(
        N_terminal_atom.GetIdx(),
        C_terminal_atom.GetIdx(),
        order=Chem.rdchem.BondType.SINGLE,
    )

    # 减少氢数量
    N_terminal_atom.SetNumExplicitHs(1)
    C_terminal_atom.SetNumExplicitHs(1)

    # 获取结果分子
    cyclic_peptide = rw_peptide.GetMol()

    # 修复价状态
    Chem.SanitizeMol(cyclic_peptide)

    return cyclic_peptide
