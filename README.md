# PepExplainer

❗️❗️❗️[The code will be updated in the next two days.]

This is the official implementation for the paper titled 'PepExplainer: an explainable deep learning model for selection-based macrocyclic peptide bioactivity prediction and optimization'.

![flowchart](flowchart.png)

## Setup

The code was tested on `Python 3.7` with `PyTorch 1.12.1`, utilizing an NVIDIA GPU 3090.

```bash
mamba create -nPepExplainer python=3.7
mamba activate PepExplainer
mamba install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
mamba install -c dglteam/label/cu113 dgl
python -m pip install rdkit==2022.3.3 scikit-learn==1.0.2 seaborn numpy pandas scipy ipykernel tqdm six black rich hyperopt pyyaml fire biopython
python -m pip install -e .

pip install peptides  # for peptide encodings
```

## Dataset

The dataset will be uploaded soon.

The model ckpt could be found at dir `model`.

## Scripts

```bash
cd scripts

# trainning and evalation for enrichment prediction
python train_mol.py | tee ../logs/train_mol.log
python eval_mol.py | tee ../logs/eval_mol.log

# trainning and evalation for bioactivity prediction
python train_mol_bio.py | tee ../logs/train_mol_bio.log
python eval_mol_bio.py | tee ../logs/eval_mol_bio.log
```

The output of the model is saved in `./prediction/`.

## Notebooks

The data visualization and interpretation notebooks are available in `./notebooks/`.

- `plot_tmap.ipynb` -> [Atlas of macrocyclic peptides](notebooks/index.html)
- `Amino acid explanation.ipynb`: The visualaztion for macrocyclic peptides
- `scan-base.ipynb`: Single amino acid mutations of peptide `17C-L20`
