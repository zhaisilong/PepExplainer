from pathlib import Path
import torch
import pandas as pd
import numpy as np
from core.utils import amino_acid_smiles, build_mol_graph_data, AminoAcid, \
    show_atom_idx, Peptide, set_random_seed, load_graph_from_csv_bin_for_splited, collate_molgraphs, pos_weight, \
    EarlyStopping, run_a_train_epoch, run_an_eval_epoch, build_mol_graph_data, make_fake_cv, add_params
from rdkit.Chem import rdchem
from torch.utils.data import DataLoader
from dgl.data.graph_serialize import save_graphs
from rdkit import Chem
from core.models import RGCN
from torch.optim import Adam
from typing import Dict
from tqdm.auto import tqdm, trange

import warnings
warnings.filterwarnings('ignore')

task = "ic50_reg"
sub_type = "mol"
data_name = "bio_reg"
label_name = "target"
peptide_name = "peptide"
methods = "thioether"
max_workers = None  # os.cpu_count() -> 12

origin_data_dir = Path("../data/origin_data")
graph_data_dir = Path("../data/graph_data")
prediction_dir = Path("../prediction")
graph_data_dir.mkdir(exist_ok=True)
prediction_dir.mkdir(exist_ok=True)

origin_data_path = origin_data_dir / f"{data_name}.csv"
save_g_path = graph_data_dir / f"{task}_{sub_type}.bin"
save_g_group_path = graph_data_dir / f"{task}_{sub_type}_group.csv"

if save_g_path.exists():
        print(f"Molecules graph already exists: {save_g_path}")
else:
    data_origin = pd.read_csv(origin_data_path)
    data_set_gnn_for_peptide = build_mol_graph_data(
        dataset_peptide=data_origin,
        label_name=label_name,
        peptide_name=peptide_name,
                methods=methods,
        max_workers=max_workers,
    )
    sequence, smiles, g_rgcn, labels, split_index = map(
        list, zip(*data_set_gnn_for_peptide)
    )
    graph_labels = {"labels": torch.tensor(labels)}
    split_index_pd = pd.DataFrame(columns=["sequence", "smiles", "group"])
    split_index_pd.sequence = sequence
    split_index_pd.smiles = smiles
    split_index_pd.group = split_index
    split_index_pd.to_csv(save_g_group_path, index=False, columns=None)
    save_graphs(str(save_g_path), g_rgcn, graph_labels)
    print("Molecules graph is saved!")