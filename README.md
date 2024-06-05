# PepExplainer

❗️❗️❗️**The code will be updated in the next two days.**

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

### Training

## Examples

### HLA dataset

The HLA dataset was derived from the paper ['A transformer-based model to predict peptide-HLA class I binding and optimize mutated peptides for vaccine design'](https://www.nature.com/articles/s42256-022-00459-7) and is available on [GitHub](https://github.com/a96123155/TransPHLA-AOMP).

To build the HLA dataset for PepExplainer, please execute the preprocessing script `build_hla_datasets.ipynb`. We provide the raw data and trained checkpoints for PepExplainer, excluding the processed data, as it was too large to upload to GitHub.

For prediction and analysis, you can run `notebooks/example_hla.ipynb`.

---





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


---

## Dataset

### 是否含氨基酸 C

- 生成长度为10的10000条序列
  - positive: 400
  - negative: 600
- results:
  - prediction: 0.9984732031822204
  - label: 1.0

### 亲和力 分类

- 初始数据：163949
- 取头部数据的大小: 1639
  - mean: 787.2031726662599
  - std: 540.2880711651959
  - median: 594.0
  - max: 5333
  - min: 370
- results
  - prediction: 0.5549856454133988
  - label: 1.0

### 回归数据

亲和力分类数据 -> 回归数据（按标签划分, pos:neg=1:1）
修改模型参数

```yml
rgcn_hidden_feats: [64, 128, 254, 512]
ffn_hidden_feats: 256
rgcn_drop_out: 0.1
ffn_drop_out: 0.1
lr: 0.001
mode: higher
metric_name: r2
classification: false
batch_size: 2
```

`mol` 上的 `r2` 有 0.7

以下是 BERT 回归模型的结果。

| cv | task      | set   | mse                 | r2                  |
|----|-----------|-------|---------------------|---------------------|
| 0  | del       | train | 0.03611788526177406 | 0.9043081402778625  |
| 0  | del       | valid | 0.0410902239382267  | 0.8909390568733215  |
| 0  | del       | test  | 0.05169228464365005 | 0.9553111791610718  |
| 1  | del       | train | 0.03511447459459305 | 0.906822979450225   |
| 1  | del       | valid | 0.04296974092721939 | 0.8866583108901978  |
| 1  | del       | test  | 0.0513150580227375  | 0.9556372761726379  |
| 2  | del       | train | 0.0358717143535614  | 0.9050713777542114  |
| 2  | del       | valid | 0.04295651987195015 | 0.885456383228302   |
| 2  | del       | test  | 0.05018607899546623 | 0.956613302230835   |
| 3  | del       | train | 0.0357465967535972  | 0.9053764939308167  |
| 3  | del       | valid | 0.04178217798471451 | 0.8887107968330383  |
| 3  | del       | test  | 0.04678865149617195 | 0.9595504403114319  |
| 4  | del       | train | 0.03606702387332916 | 0.9042273163795471  |
| 4  | del       | valid | 0.04191279411315918 | 0.8897522687911987  |
| 4  | del       | test  | 0.049294617027044296| 0.9573839902877808  |
| 0  | bio_pure  | test  | 0.5470705032348633  | 0.7661590576171875  |
| 1  | bio_pure  | test  | 0.7223392724990845  | 0.5398973226547241  |
| 2  | bio_pure  | test  | 0.8636692762374878  | 0.5976643562316895  |
| 3  | bio_pure  | test  | 0.3343295156955719  | 0.8492527008056641  |
| 4  | bio_pure  | test  | 0.623153805732727   | 0.7249959707260132  |
| 0  | bio       | test  | 0.13694481551647186 | 0.9414640069007874  |
| 1  | bio       | test  | 0.2927936613559723  | 0.8135015964508057  |
| 2  | bio       | test  | 0.47147688269615173 | 0.7803650498390198  |
| 3  | bio       | test  | 0.1496168076992035  | 0.932538628578186   |
| 4  | bio       | test  | 0.4038373529911041  | 0.8217824697494507  |

以下是我们的模型的结果

```yml
head_result: r2, mae, rmse
training_result: 0.896,0.0541,0.0674
valid_result: 0.7173,0.1018,0.1267
test_result: 0.7173,0.1018,0.1267
```

### 回归数据全

在掩码数据上，显存不够，得上云。

```bash
# 上传
# tgt_host=autodl
tgt_host=light_
sme_version=3
rsync -aPvz --delete ~/light/SME/v$sme_version/ $tgt_host:~/zsl/sme.v$sme_version

# 下载
src_host=autodl
sme_version=3
rsync -aPvz --delete $src_host:~/zsl/sme.v$sme_version/ ~/light/SME/v$sme_version
```

## Run

`notebooks` and `scripts`

| 实验 | 描述                            | 数据           | 训练 mol                           | 训练 子结构                                | 解释        |
| ---------- | --------------------------------------- | -------------- | ---------------------------------------------------- | ---------------------------------------------------- | --------------------- |
| has_c      | 寻找线性肽中的C氨基酸 | `data-c.ipynb` | `run_mol_cls_c.ipynb` and `run_mol_cls_seed_c.ipynb` | `run_mask_cls_c.ipynb` and `run_mask_cls_seed_c.ipynb` | `explain_cls_c.ipynb` |
| affinity | classification on sampled del datset using linear peptide | `data.ipynb` | `run_mol_cls.ipynb` and `run_mol_cls_seed.ipynb` | `run_mask_cls_seed.ipynb` and `run_mask_cls_seed.ipynb` | `explain_cls.ipynb` |
| affinity_reg | regression on sampled del datset using linear peptide | `data.ipynb` | `run_mol_reg.ipynb` and `run_mol_reg_seed.ipynb` | `run_mask_reg.ipynb` and `run_mol_reg_seed.ipynb` | `explain_reg.ipynb`|

### 分布式训练

- [【分布式训练】单机多卡的正确打开方式（三）：PyTorch](https://zhuanlan.zhihu.com/p/74792767)

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run_dist_mask_reg.py
```
