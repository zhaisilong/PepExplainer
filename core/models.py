import torch.nn.functional as F
from dgl.nn.pytorch.conv import RelGraphConv
from torch import nn
import torch as th
from dgl.readout import sum_nodes
import dgl
from pathlib import Path
import torch
from tqdm.auto import tqdm
from core.utils import add_params, construct_RGCN_mol_graph_from_mol, Peptide
import numpy as np


class WeightAndSum(nn.Module):
    """Compute importance weights for atoms and perform a weighted sum.

    Parameters
    ----------
    in_feats : int
        Input atom feature size
    """

    def __init__(self, in_feats):
        super(WeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(nn.Linear(in_feats, 1), nn.Sigmoid())

    def forward(self, g, feats, smask):
        """Compute molecule representations out of atom representations

        Parameters
        ----------
        g : DGLGraph
            DGLGraph with batch size B for processing multiple molecules in parallel
        feats : FloatTensor of shape (N, self.in_feats)
            Representations for all atoms in the molecules
            * N is the total number of atoms in all molecules
        smask: substructure mask, atom node for 0, substructure node for 1.

        Returns
        -------
        FloatTensor of shape (B, self.in_feats)
            Representations for B molecules
        """
        with g.local_scope():
            g.ndata["h"] = feats
            weight = self.atom_weighting(g.ndata["h"]) * smask
            g.ndata["w"] = weight
            h_g_sum = sum_nodes(g, "h", "w")
        return h_g_sum, weight


class RGCNLayer(nn.Module):
    """Single layer RGCN for updating node features
    Parameters
    ----------
    in_feats : int
        Number of input atom features
    out_feats : int
        Number of output atom features
    num_rels: int
        Number of bond type
    activation : activation function
        Default to be ReLU
    loop: bool:
        Whether to use self loop
        Default to be False
    residual : bool
        Whether to use residual connection, default to be True
    batchnorm : bool
        Whether to use batch normalization on the output,
        default to be True
    rgcn_drop_out : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    hyperbolic: str
        Riemannian Manifolds. Defalt: 'Poincare'
    fix: bool
        是否固定参数. Default to: False
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_rels=65,
        activation=F.relu,
        loop=False,
        residual=True,
        batchnorm=True,
        rgcn_drop_out=0.5,
        fixed=False,
    ):
        super(RGCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv_layer = RelGraphConv(
            in_feats,
            out_feats,
            num_rels=num_rels,
            regularizer="basis",
            num_bases=None,
            bias=True,
            activation=activation,
            self_loop=loop,
            dropout=rgcn_drop_out,
        )
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, bg, node_feats, etype, norm=None):
        """Update atom representations
        Parameters
        ----------
        bg : BatchedDGLGraph
            Batched DGLGraphs for processing multiple molecules in parallel
        node_feats : FloatTensor of shape (N, M1)
            * N is the total number of atoms in the batched graph
            * M1 is the input atom feature size, must match in_feats in initialization
        etype: int
            bond type
        norm: th.Tensor
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`
        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output atom feature size, must match out_feats in initialization
        """
        new_feats = self.graph_conv_layer(bg, node_feats, etype, norm)
        if self.residual:
            res_feats = self.activation(self.res_connection(node_feats))
            new_feats = new_feats + res_feats
        if self.bn:
            new_feats = self.bn_layer(new_feats)
        del res_feats
        th.cuda.empty_cache()
        return new_feats


class BaseGNN(nn.Module):
    """HRGCN based predictor for multitask prediction on molecular graphs
    We assume each task requires to perform a binary classification.
    Parameters
    ----------
    gnn_out_feats : int
        Number of atom representation features after using GNN
    len_descriptors : int
        length of descriptors
    hyperbolic: str
        Riemannian Manifolds. Defalt: 'Poincare'
    rgcn_drop_out: float
        dropout rate for HRGCN layer
    n_tasks : int
        Number of prediction tasks
    classifier_hidden_feats : int
        Number of molecular graph features in hidden layers of the MLP Classifier
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    return_weight: bool
        Wether to return atom weight defalt=False
    """

    def __init__(
        self,
        gnn_rgcn_out_feats,
        ffn_hidden_feats,
        ffn_dropout=0.25,
        classification=True,
    ):
        super(BaseGNN, self).__init__()
        self.classification = classification
        self.rgcn_gnn_layers = nn.ModuleList()
        self.readout = WeightAndSum(gnn_rgcn_out_feats)
        self.fc_layers1 = self.fc_layer(
            ffn_dropout, gnn_rgcn_out_feats, ffn_hidden_feats
        )
        self.fc_layers2 = self.fc_layer(ffn_dropout, ffn_hidden_feats, ffn_hidden_feats)
        self.fc_layers3 = self.fc_layer(ffn_dropout, ffn_hidden_feats, ffn_hidden_feats)
        self.predict = self.output_layer(ffn_hidden_feats, 1)

    def forward(
        self, rgcn_bg, rgcn_node_feats, rgcn_edge_feats, smask_feats, get_embds=None
    ):
        """Multi-task prediction for a batch of molecules"""
        # Update atom features with GNNs
        for rgcn_gnn in self.rgcn_gnn_layers:
            rgcn_node_feats = rgcn_gnn(rgcn_bg, rgcn_node_feats, rgcn_edge_feats)
        # Compute molecule features from atom features and bond features
        graph_feats, weight = self.readout(rgcn_bg, rgcn_node_feats, smask_feats)
        h1 = self.fc_layers1(graph_feats)
        h2 = self.fc_layers2(h1)
        h3 = self.fc_layers3(h2)
        out = self.predict(h3)
        if get_embds:
            return out, weight, graph_feats
        else:
            return out, weight

    def fc_layer(self, dropout, in_feats, hidden_feats):
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats),
        )

    def output_layer(self, hidden_feats, out_feats):
        return nn.Sequential(nn.Linear(hidden_feats, out_feats))


class RGCN(BaseGNN):
    """HRGCN based predictor for multitask prediction on molecular graphs
    We assume each task requires to perform a binary classification.
    Parameters
    ----------
    in_feats : int
        Number of input atom features
    Rgcn_hidden_feats : list of int
        rgcn_hidden_feats[i] gives the number of output atom features
        in the i+1-th HRGCN layer
    n_tasks : int
        Number of prediction tasks
    len_descriptors : int
        length of descriptors
    return_weight : bool
        Wether to return weight
    classifier_hidden_feats : int
        Number of molecular graph features in hidden layers of the MLP Classifier
    is_descriptor: bool
        Wether to use descriptor
    loop : bool
        Wether to use self loop
    gnn_drop_rate : float
        The probability for dropout of HRGCN layer. Default to be 0.5
    dropout : float
        The probability for dropout of MLP layer. Default to be 0.
    """

    def __init__(
        self,
        ffn_hidden_feats,
        rgcn_node_feats,
        rgcn_hidden_feats,
        rgcn_drop_out=0.25,
        ffn_dropout=0.25,
        classification=True,
        fixed=False,
    ):
        super(RGCN, self).__init__(
            gnn_rgcn_out_feats=rgcn_hidden_feats[-1],
            ffn_hidden_feats=ffn_hidden_feats,
            ffn_dropout=ffn_dropout,
            classification=classification,
        )
        for i in range(len(rgcn_hidden_feats)):
            rgcn_out_feats = rgcn_hidden_feats[i]
            self.rgcn_gnn_layers.append(
                RGCNLayer(
                    rgcn_node_feats,
                    rgcn_out_feats,
                    loop=True,
                    rgcn_drop_out=rgcn_drop_out,
                    fixed=fixed,
                )
            )
            rgcn_node_feats = rgcn_out_feats


def predict_one(model, rgcn_bg, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        rgcn_bg = rgcn_bg.to(device)
        rgcn_node_feats = rgcn_bg.ndata.pop("node").float().to(device)
        rgcn_edge_feats = rgcn_bg.edata.pop("edge").long().to(device)
        smask_feats = rgcn_bg.ndata.pop("smask").unsqueeze(dim=1).float().to(device)
        preds, _ = model(rgcn_bg, rgcn_node_feats, rgcn_edge_feats, smask_feats)
        y_pred = preds.squeeze().detach().cpu().numpy()
        return y_pred


def get_embedding_one(model, rgcn_bg, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        rgcn_bg = rgcn_bg.to(device)
        rgcn_node_feats = rgcn_bg.ndata.pop("node").float().to(device)
        rgcn_edge_feats = rgcn_bg.edata.pop("edge").long().to(device)
        smask_feats = rgcn_bg.ndata.pop("smask").unsqueeze(dim=1).float().to(device)
        preds, _, embeddings = model(
            rgcn_bg, rgcn_node_feats, rgcn_edge_feats, smask_feats, True
        )

        y_pred = preds.squeeze().detach().cpu().numpy()

        return y_pred, embeddings.detach().cpu().numpy()


def load_checkpoint(model, checkpoints_path, verbose=False):
    if verbose:
        print(f"Load checkpoint from {checkpoints_path}")
    state_dict = torch.load(checkpoints_path)
    model.load_state_dict(state_dict["model_state_dict"])


class AME:
    def __init__(
        self, config_path: str = "../config/ame.yml", checkpoints_dir: str = "../model"
    ):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.model = self.init_model(config_path)

    def init_model(self, config_path):
        path = Path(config_path)
        self.args = {}
        add_params(path, self.args)
        print(f"Args: {self.args}")

        model = RGCN(
            ffn_hidden_feats=self.args["ffn_hidden_feats"],
            ffn_dropout=self.args["ffn_drop_out"],
            rgcn_node_feats=self.args["in_feats"],
            rgcn_hidden_feats=self.args["rgcn_hidden_feats"],
            rgcn_drop_out=self.args["rgcn_drop_out"],
            classification=self.args["classification"],
        )

        load_checkpoint(
            model,
            self.checkpoints_dir
            / f"{self.args['task']}_{self.args['sub_type']}_0_early_stop.pth",
        )

        return model

    def build_batch_data(self, peptideList, verbose=True):
        peptides = []
        peptide_number = len(peptideList)
        aa_mask_list = []
        aa_name_list = []
        g_rgcns_list = []

        with tqdm(
            enumerate(peptideList), total=peptide_number, disable=not verbose
        ) as pbar:
            for i, _peptide in pbar:
                peptide = Peptide(_peptide, methods=self.args["methods"])
                g_rgcns = []
                aa_mask = []
                aa_name = []

                try:
                    g_rgcn = construct_RGCN_mol_graph_from_mol(peptide.mol, smask=[])
                    g_rgcns.append(g_rgcn)
                    aa_mask.append([])
                    aa_name.append("noname")
                except:
                    print(f"Failed to build mol for {peptide}")

                for substructure, amino_acid in zip(
                    peptide.structure, peptide.amino_acids
                ):
                    aa_mask.append(substructure)
                    aa_name.append(amino_acid)

                for j, aa_mask_j in enumerate(peptide.structure):
                    try:
                        g_rgcn = construct_RGCN_mol_graph_from_mol(
                            peptide.mol, smask=aa_mask_j
                        )
                        g_rgcns.append(g_rgcn)
                        aa_mask.append(aa_mask_j)
                        aa_name.append(peptide.amino_acids)

                    except Exception as e:
                        print(f"Failed to build mask for {peptide}")

                g_rgcns_list.append(g_rgcns)
                aa_name_list.append(aa_name)
                aa_mask_list.append(aa_mask)
                peptides.append(peptide)

        return peptides, g_rgcns_list, aa_mask_list, aa_name_list

    def load_model_for_task(self, task_name):
        """Load a model checkpoint based on task name and seed."""
        checkpoint_path = self.checkpoints_dir / f"{task_name}_early_stop.pth"
        load_checkpoint(self.model, checkpoint_path)

    def predict_for_batch(self, rgcn_bg_list):
        """Predict for a batch of sequences."""
        y_preds = []
        for rgcn_bg in rgcn_bg_list:
            y_pred = predict_one(self.model, dgl.batch(rgcn_bg), self.args["device"])
            y_preds.append(y_pred)
        return y_preds

    def get_embeddings_for_batch(self, rgcn_bg_list):
        embeddings = []
        for rgcn_bg in rgcn_bg_list:
            _, embedding = get_embedding_one(
                self.model, dgl.batch(rgcn_bg), self.args["device"]
            )
            embeddings.append(embedding)
        return embeddings

    def get_embedding(self, seqs, verbose=True):
        if isinstance(seqs, str):
            seqs = [seqs]

        g_rgcns = []
        peptide_number = len(seqs)

        with tqdm(enumerate(seqs), total=peptide_number, disable=not verbose) as pbar:
            for i, _peptide in pbar:
                peptide = Peptide(_peptide, methods=self.args["methods"])
                try:
                    g_rgcn = construct_RGCN_mol_graph_from_mol(peptide.mol, smask=[])
                    g_rgcns.append(g_rgcn)
                except:
                    print(f"Failed to build mol for {peptide}")

        embedding_mols = []

        for seed in range(10):
            # Only for mol task
            task_name = f"{self.args['task']}_mol_{seed}"
            self.load_model_for_task(task_name)
            _, embedding = get_embedding_one(
                self.model, dgl.batch(g_rgcns), self.args["device"]
            )
            embedding_mols.append(embedding)

        return np.array(embedding_mols).mean(axis=0)

    def predict(self, seqs, verbose=True):
        if isinstance(seqs, str):
            seqs = [seqs]

        peptides, g_rgcns_list, _, _ = self.build_batch_data(seqs, verbose=verbose)

        y_preds_mol = []
        y_preds_mask = []

        for seed in range(10):
            # For mol task
            task_name = f"{self.args['task']}_mol_{seed}"
            self.load_model_for_task(task_name)
            y_preds_mol_temp = self.predict_for_batch(g_rgcns_list)
            y_preds_mol.append([y[0] for y in y_preds_mol_temp])

            # For mask task
            task_name = f"{self.args['task']}_{self.args['sub_type']}_{seed}"
            self.load_model_for_task(task_name)
            y_preds_mask_temp = self.predict_for_batch(g_rgcns_list)
            y_preds_mask.append(
                [y[1:] for y in y_preds_mask_temp]
            )  # this is for the mol task

        # Compute means and stds
        y_preds_mol_mean = np.mean(y_preds_mol, axis=0)
        y_preds_mol_std = np.std(y_preds_mol, axis=0)

        y_preds_mask_mean = np.mean(y_preds_mask, axis=0)
        y_preds_mask_std = np.std(y_preds_mask, axis=0)

        # Calculate attributions
        attribution = y_preds_mol_mean[:, np.newaxis] - y_preds_mask_mean
        attribution_normalized = (np.exp(attribution) - np.exp(-attribution)) / (
            np.exp(attribution) + np.exp(-attribution)
        )

        return (
            peptides,
            y_preds_mol_mean,
            y_preds_mol_std,
            y_preds_mask_mean,
            y_preds_mask_std,
            attribution_normalized,
        )

    def predict_with_batch_size(self, seqs, batch_size=2048, verbose=True):
        peptides_list = []
        y_preds_mol_mean_list = []
        y_preds_mol_std_list = []
        y_preds_mask_mean_list = []
        y_preds_mask_std_list = []
        attribution_normalized_list = []
        num_batches = (
            len(seqs) + batch_size - 1
        ) // batch_size  # This ensures we handle the last batch correctly

        for i in range(0, len(seqs), batch_size):
            if verbose:
                print(
                    f"{i // batch_size + 1}/{num_batches}"
                )  # Correct batch progress output
            batch_seqs = seqs[i : i + batch_size]
            (
                peptides,
                y_preds_mol_mean,
                y_preds_mol_std,
                y_preds_mask_mean,
                y_preds_mask_std,
                attribution_normalized,
            ) = self.predict(batch_seqs, verbose=verbose)

            peptides_list.extend(peptides)
            y_preds_mol_mean_list.append(y_preds_mol_mean)
            y_preds_mol_std_list.append(y_preds_mol_std)
            y_preds_mask_mean_list.append(y_preds_mask_mean)
            y_preds_mask_std_list.append(y_preds_mask_std)
            attribution_normalized_list.append(attribution_normalized)

        # Ensure concatenation is done for each correct list
        return (
            peptides_list,
            np.concatenate(y_preds_mol_mean_list),
            np.concatenate(y_preds_mol_std_list),
            np.concatenate(y_preds_mask_mean_list),
            np.concatenate(y_preds_mask_std_list),
            np.concatenate(attribution_normalized_list),
        )

    def predict_with_threshold(self, seqs, verbose=True, threshold=0.2):
        """abboorted"""
        if isinstance(seqs, str):
            seqs = [seqs]

        peptides, g_rgcns_list, _, _ = self.build_batch_data(seqs, verbose=verbose)

        y_preds_mol = []
        y_preds_mask = []

        for seed in range(10):
            # For mol task
            task_name = f"{self.args['task']}_mol_{seed}"
            self.load_model_for_task(task_name)
            y_preds_mol_temp = self.predict_for_batch(g_rgcns_list)
            y_preds_mol.append([y[0] for y in y_preds_mol_temp])

            # For mask task
            task_name = f"{self.args['task']}_{self.args['sub_type']}_{seed}"
            self.load_model_for_task(task_name)
            y_preds_mask_temp = self.predict_for_batch(g_rgcns_list)
            y_preds_mask.append(
                [y[1:] for y in y_preds_mask_temp]
            )  # this is for the mol task

        # Compute means and stds
        y_preds_mol_std = np.std(y_preds_mol, axis=0)
        y_preds_mol_mean = np.mean(y_preds_mol, axis=0)

        y_preds_mask_std = np.std(y_preds_mask, axis=0)
        y_preds_mask_mean = np.mean(y_preds_mask, axis=0)

        # Calculate attributions
        attribution = y_preds_mol_mean[:, np.newaxis] - y_preds_mask_mean
        attribution_normalized = (np.exp(attribution) - np.exp(-attribution)) / (
            np.exp(attribution) + np.exp(-attribution)
        )

        return (
            peptides,
            y_preds_mol_mean,
            y_preds_mol_std,
            y_preds_mask_mean,
            y_preds_mask_std,
            attribution_normalized,
        )
