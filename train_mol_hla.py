from pathlib import Path
import torch
import pandas as pd
from core.utils import (
    build_mol_graph_data,
    set_random_seed,
    load_graph_from_csv_bin_for_splited,
    collate_molgraphs,
    pos_weight,
    EarlyStopping,
    run_a_train_epoch,
    run_an_eval_epoch,
    add_params,
)

from torch.utils.data import DataLoader
from dgl.data.graph_serialize import save_graphs
from core.models import RGCN
from torch.optim import Adam
from tqdm.auto import trange
from torch.utils.data.distributed import DistributedSampler
from typing import Optional

from fire import Fire
import warnings

warnings.filterwarnings("ignore")

global_inital_dist = True

def run_a_model(seed: Optional[int] = None):
    set_random_seed(seed)

    global global_inital_dist

    print("*" * 80)
    print(f"Run at seed {seed}")

    task = "hla"
    sub_type = "mol"
    data_name = "hla"
    label_name = "target"
    peptide_name = "peptide"
    methods = "linear"
    max_workers = None

    origin_data_dir = Path("./data/hla/origin_data")
    graph_data_dir = Path("./data/hla/graph_data")
    prediction_dir = Path("./prediction")
    graph_data_dir.mkdir(exist_ok=True)
    prediction_dir.mkdir(exist_ok=True)

    config_path = Path("config")

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

    args = {}
    args["dist"] = False

    args["node_data_field"] = "node"
    args["edge_data_field"] = "edge"
    args["substructure_mask"] = "smask"

    # model parameter
    args["num_epochs"] = 500
    args["report_epochs"] = 5
    args["patience"] = 100
    args["in_feats"] = 40
    args["max_evals"] = 30
    args["loop"] = True

    args["task_name"] = task
    args["sub_type"] = sub_type

    args["bin_path"] = str(save_g_path)
    args["group_path"] = str(save_g_group_path)

    # 模型参数
    add_params(config_path / f"{task}_{data_name}_{sub_type}.yml", args)
    print(f"Args: {args}")

    train_set, valid_set, test_set, task_number = load_graph_from_csv_bin_for_splited(
        bin_path=args["bin_path"],
        group_path=args["group_path"],
        classification=args["classification"],
        seed=2024,
        random_shuffle=False,
    )

    train_sampler, valid_sampler, test_sampler = None, None, None

    if args["dist"]:
        # 分布式初始化 run once
        if global_inital_dist:
            torch.distributed.init_process_group(backend="nccl")
            global_inital_dist = False

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
        # loss_criterion = torch.nn.HuberLoss()
        loss_criterion = torch.nn.L1Loss()

    model = RGCN(
        ffn_hidden_feats=args["ffn_hidden_feats"],
        ffn_dropout=args["ffn_drop_out"],
        rgcn_node_feats=args["in_feats"],
        rgcn_hidden_feats=args["rgcn_hidden_feats"],
        rgcn_drop_out=args["rgcn_drop_out"],
        classification=args["classification"],
        fixed=False,
    )

    stopper = EarlyStopping(
        patience=args["patience"],
        task_name=args["task_name"],
        sub_type=args["sub_type"],
        seed=seed,
        mode=args["mode"],
        # pretrained_model="../model/enrich_reg_mol_0_early_stop.pth",
    )

    model.to(args["device"])

    # Finetune
    # stopper.load_pretrained_model(model)

    if args["dist"] and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 5) 封装
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
    lr = args["lr"]
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.01)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    for epoch in range(args["num_epochs"]):
        # Train

        print(f"Training {epoch+1}/{args['num_epochs']}")
        _, total_loss = run_a_train_epoch(
            args, model, train_loader, loss_criterion, optimizer
        )

        # Validation every step for checking early stop
        val_score, val_loss = run_an_eval_epoch(
            args, model, valid_loader, loss_criterion, out_path=None, stage="valid"
        )

        early_stop = stopper.step(val_score[0], model, dist=args["dist"])
        if early_stop:
            break

        if (epoch + 1) % args["report_epochs"] == 0:
            train_score, trian_loss = run_an_eval_epoch(
                args, model, train_loader, loss_criterion, out_path=None, stage="train"
            )
            # Test
            test_score, test_loss = run_an_eval_epoch(
                args, model, test_loader, loss_criterion, out_path=None, stage="test"
            )
            print(
                f"epoch {epoch + 1}/{args['num_epochs']}, {args['metric_name']}, lr: {lr:.6f} | train_loss: {total_loss:.4f}, val_loss: {val_loss:.4f}, test_loss: {test_loss:.4f} | train: {train_score[0]:.4f}, valid: {val_score[0]:.4f}, test: {test_score[0]:.4f} | best valid score {stopper.best_score:.4f}\n"
            )

    print(f"Training done! Model was saved to {stopper.filename}")

if __name__ == "__main__":
    Fire(run_a_model)
