from pathlib import Path
import torch
import pandas as pd
from core.utils import (
    build_mol_graph_data_for_peptide,
    set_random_seed,
    load_graph_from_csv_bin_for_splited,
    collate_molgraphs,
    pos_weight,
    EarlyStopping,
    run_a_train_epoch,
    run_an_eval_epoch,
    add_params,
    write_results,
)

from torch.utils.data import DataLoader
from dgl.data.graph_serialize import save_graphs
from core.models import RGCN
from torch.optim import Adam
from tqdm.auto import trange
from torch.utils.data.distributed import DistributedSampler
from typing import Optional
import numpy as np
from fire import Fire

import warnings

warnings.filterwarnings("ignore")


def run_a_model(seed: Optional[int] = None):
    set_random_seed(seed)
    print("*" * 80)
    print(f"Run at seed {seed}")

    task = "ic50_reg"
    sub_type = "aa"
    data_name = "bio_reg"
    label_name = "target"
    peptide_name = "peptide"
    methods = "thioether"
    max_workers = 16  # os.cpu_count() -> 12

    origin_data_dir = Path("./data/il17c/origin_data")
    graph_data_dir = Path("./data/il17c/graph_data")
    prediction_dir = Path("./prediction")
    graph_data_dir.mkdir(exist_ok=True)
    prediction_dir.mkdir(exist_ok=True)

    config_path = Path("./config")

    origin_data_path = origin_data_dir / f"{data_name}.csv"
    save_g_path = graph_data_dir / f"{task}_{sub_type}.bin"
    save_g_smask_path = graph_data_dir / f"{task}_{sub_type}_smask.npy"
    save_g_group_path = graph_data_dir / f"{task}_{sub_type}_group.csv"

    if save_g_path.exists():
        print(f"Molecules graph already exists: {save_g_path}")
    else:
        data_origin = pd.read_csv(origin_data_path)
        data_set_gnn_for_peptide = build_mol_graph_data_for_peptide(
            dataset_peptide=data_origin,
            label_name=label_name,
            peptide_name=peptide_name,
            methods=methods,
            max_workers=max_workers,
        )
        sequence, smiles, g_rgcn, labels, split_index, smask, sname = map(
            list, zip(*data_set_gnn_for_peptide)
        )
        graph_labels = {"labels": torch.tensor(labels)}
        split_index_pd = pd.DataFrame(
            columns=["sequence", "smiles", "group", "sub_name"]
        )
        split_index_pd.sequence = sequence
        split_index_pd.smiles = smiles
        split_index_pd.group = split_index
        split_index_pd.sub_name = sname
        split_index_pd.to_csv(save_g_group_path, index=False, columns=None)
        smask_np = np.array(smask, dtype=object)
        np.save(save_g_smask_path, smask_np)
        save_graphs(str(save_g_path), g_rgcn, graph_labels)
        print("Molecules graph is saved!")

    args = {}
    args["dist"] = False  # Eval mode need dist setted to False
    args["node_data_field"] = "node"
    args["edge_data_field"] = "edge"
    args["substructure_mask"] = "smask"

    # model parameter
    args["num_epochs"] = 500
    args["report_epochs"] = 5
    args["patience"] = 40
    args["in_feats"] = 40
    args["max_evals"] = 30
    args["loop"] = True

    args["task_name"] = task
    args["sub_type"] = sub_type

    args["bin_path"] = str(save_g_path)
    args["save_g_smask_path"] = str(save_g_smask_path)
    args["group_path"] = str(save_g_group_path)

    # 模型参数
    add_params(config_path / f"{task}_{data_name}_{sub_type}.yml", args)
    print(f"Args: {args}")

    train_set, valid_set, test_set, task_number = load_graph_from_csv_bin_for_splited(
        bin_path=args["bin_path"],
        group_path=args["group_path"],
        classification=args["classification"],
        smask_path=args["save_g_smask_path"],
        seed=2023,
        random_shuffle=False,
    )

    train_sampler, valid_sampler, test_sampler = None, None, None

    if args["dist"]:
        # 分布式初始化
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        args["device"] = torch.device("cuda", local_rank)

        train_sampler = DistributedSampler(train_set, shuffle=True)
        valid_sampler = DistributedSampler(valid_set, shuffle=False)
        test_sampler = DistributedSampler(test_set, shuffle=False)
    else:
        args["device"] = "cuda"

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args["batch_size"],
        collate_fn=collate_molgraphs,
        sampler=train_sampler,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args["batch_size"],
        collate_fn=collate_molgraphs,
        sampler=valid_sampler,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args["batch_size"],
        collate_fn=collate_molgraphs,
        sampler=test_sampler,
        pin_memory=True,
    )
    print("Molecule graph is loaded!")

    if args["classification"]:
        pos_weight_np = pos_weight(train_set)
        loss_criterion = torch.nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=pos_weight_np.to(args["device"])
        )
    else:
        loss_criterion = torch.nn.HuberLoss()

    model = RGCN(
        ffn_hidden_feats=args["ffn_hidden_feats"],
        ffn_dropout=args["ffn_drop_out"],
        rgcn_node_feats=args["in_feats"],
        rgcn_hidden_feats=args["rgcn_hidden_feats"],
        rgcn_drop_out=args["rgcn_drop_out"],
        classification=args["classification"],
    )

    stopper = EarlyStopping(
        patience=args["patience"],
        task_name=args["task_name"],
        sub_type=args["sub_type"],
        seed=seed,
        mode=args["mode"],
    )

    print(stopper.filename)
    stopper.load_checkpoint(model)

    model.to(args["device"])

    if args["dist"] and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 5) 封装
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    if args["classification"]:
        print("head_result:", "accuracy, se, sp, f1, pre, rec, err, mcc")
    else:
        print("head_result:", "r2, mae, rmse")

    write_results(
        model,
        train_loader,
        "train_set",
        loss_criterion,
        prediction_dir,
        seed,
        args,
        "train",
    )
    write_results(
        model,
        valid_loader,
        "valid_set",
        loss_criterion,
        prediction_dir,
        seed,
        args,
        "valid",
    )
    write_results(
        model,
        test_loader,
        "test_set",
        loss_criterion,
        prediction_dir,
        seed,
        args,
        "test",
    )


if __name__ == "__main__":
    Fire(run_a_model)