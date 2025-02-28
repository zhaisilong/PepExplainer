# PepExplainer

This is the official implementation for the paper titled 'PepExplainer: an explainable deep learning model for selection-based macrocyclic peptide bioactivity prediction and optimization'.

![flowchart](flowchart.png)

## Setups

```bash
mamba create -npepexplainer python=3.9
mamba activate pepexplainer
mamba install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install -c dglteam/label/th21_cu118 dgl
pip install rdkit scikit-learn seaborn numpy pandas scipy ipykernel dill six black rich hyperopt pyyaml fire biopython peptides pydantic loguru transformers matplotlib==3.8.1

# For embedding visualization
pip install faerun
mamba install -c tmap tmap
pip install networkx torchdata -U  # patch the network

git clone https://github.com/zhaisilong/PepExplainer.git
cd PepExplainer
pip install -e .
```

**Tips**: The Python environment settings were updated and validated on Python 3.9 with Nvidia 3090, CUDA 11.8, and PyTorch 2.1.2. If you encounter any issues with the settings, please submit a pull request.

## Experiments

### 1. Training and Evaluation for Enrichment Prediction

To perform training and evaluation for enrichment prediction, execute the following scripts:

```bash
bash run_mol.sh
bash run_mask.sh
```

### 2. Training and Evaluation for Bioactivity Prediction

For training and evaluation focused on bioactivity prediction, run these scripts:

```bash
bash run_mol_bio.sh
bash run_mask_bio.sh
```

The output of the evaluation PepExplainer is saved in `./prediction/`.

### 3. Application of PepExplainer

The applications of PepExplainer are demonstrated in the `notebooks` directory:

1. **`Amino acid explanation.ipynb`**: Visualizes macrocyclic peptides.
2. **`attribution.ipynb`**: Computes the contribution of each amino acid at every position.
3. **`plot_tmap.ipynb`**: Visualizes molecule embeddings corresponding to peptide labels. -> [Atlas of macrocyclic peptides](notebooks/index_del.html)
4. **`scan-base.ipynb`**: Generates a heatmap through single amino acid predictions, similar to alanine scanning.

## Examples for running on public datasets

### HLA dataset

The HLA dataset was derived from the paper ['A transformer-based model to predict peptide-HLA class I binding and optimize mutated peptides for vaccine design'](https://www.nature.com/articles/s42256-022-00459-7) and is available on [GitHub](https://github.com/a96123155/TransPHLA-AOMP).

To build the HLA dataset for PepExplainer, please execute the preprocessing script `build_hla_datasets.ipynb`. We provide the raw data and trained checkpoints for PepExplainer, excluding the processed data, as it was too large to upload to GitHub.

To train the model, please run the code below:

```bash
cd scripts/
bash run_mol_hla.sh
bash run_mask_hla.sh
```

For prediction and analysis, you can run `notebooks/example_hla.ipynb`.

### Nonfuling dataset

We extended the nonfouling dataset to demonstrate PepExplainer’s capability for linear peptide property prediction. The nonfouling dataset, introduced by [Ansari](https://doi.org/10.1021/acs.jcim.2c01317), initially established baseline models for binary classification (predicting peptide activity/non-activity). The dataset comprises 17,185 positive and negative sequences, with peptide lengths ranging from 5 to 20 residues. Recent work using transformer-based methods, specifically [PeptideBERT](https://doi.org/10.1021/acs.jpclett.3c02398), achieved state-of-the-art (SOTA) results on this dataset and others related to hemolysis and solubility. We excluded the latter datasets due to their sequence lengths exceeding our model’s capabilities, focusing instead on the nonfouling dataset suitable for short peptides.

```bash
cd scripts/
bash run_mol_nf.sh
bash run_mask_nf.sh
```

### Thrombin dataset

We introduce the thioether-cyclized peptides dataset (referred to in this paper as the thrombin dataset) to showcase PepExplainer’s screening capabilities. [Merz et al.](https://doi.org/10.1038/s41589-023-01496-y) developed a combinatorial synthesis and screening approach based on sequential cyclization and one-pot peptide acylation and screening, enabling simultaneous investigation of activity and permeability. As a proof of concept, they synthesized a library of 8,448 cyclic peptides and screened them against the disease target thrombin.

```bash
cd scripts/
bash run_mol_thrombin.sh
bash run_mask_thrombin.sh
```

## Citation

See [paper](https://www.sciencedirect.com/science/article/pii/S0223523424005087) and the citation:

```txt
@article{zhaiPepExplainerExplainableDeep2024,
  title = {{{PepExplainer}}: {{An}} Explainable Deep Learning Model for Selection-Based Macrocyclic Peptide Bioactivity Prediction and Optimization},
  author = {Zhai, Silong and Tan, Yahong and Zhu, Cheng and Zhang, Chengyun and Gao, Yan and Mao, Qingyi and Zhang, Youming and Duan, Hongliang and Yin, Yizhen},
  date = {2024-09-05},
  journaltitle = {European Journal of Medicinal Chemistry},
  shortjournal = {European Journal of Medicinal Chemistry},
  volume = {275},
  pages = {116628},
  issn = {0223-5234},
  doi = {10.1016/j.ejmech.2024.116628},
  url = {https://www.sciencedirect.com/science/article/pii/S0223523424005087},
  abstract = {Macrocyclic peptides possess unique features, making them highly promising as a drug modality. However, evaluating their bioactivity through wet lab experiments is generally resource-intensive and time-consuming. Despite advancements in artificial intelligence (AI) for bioactivity prediction, challenges remain due to limited data availability and the interpretability issues in deep learning models, often leading to less-than-ideal predictions. To address these challenges, we developed PepExplainer, an explainable graph neural network based on substructure mask explanation (SME). This model excels at deciphering amino acid substructures, translating macrocyclic peptides into detailed molecular graphs at the atomic level, and efficiently handling non-canonical amino acids and complex macrocyclic peptide structures. PepExplainer's effectiveness is enhanced by utilizing the correlation between peptide enrichment data from selection-based focused library and bioactivity data, and employing transfer learning to improve bioactivity predictions of macrocyclic peptides against IL-17C/IL-17~RE interaction. Additionally, PepExplainer underwent further validation for bioactivity prediction using an additional set of thirteen newly synthesized macrocyclic peptides. Moreover, it enabled the optimization of the IC50 of a macrocyclic peptide, reducing it from 15~nM to 5.6~nM based on the contribution score provided by PepExplainer. This achievement underscores PepExplainer's skill in deciphering complex molecular patterns, highlighting its potential to accelerate the discovery and optimization of macrocyclic peptides.},
  keywords = {Bioactivity prediction,Graph neural network (GNN),Machine learning (ML),Macrocyclic peptide,Optimization,Structure-activity relationship (SAR)}
}

``
