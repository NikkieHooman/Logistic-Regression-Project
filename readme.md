
<!-- ================================================================ -->
<!--  README – FAME: Fairness-Aware Multimodal Embedding              -->
<!--  Place this file at the repo root                                -->
<!-- ================================================================ -->

<p align="center">
  <!-- Hero / architecture figure (replace with your own if you like) -->
  <img src="docs/figures/fame_hero.png"
       alt="FAME architecture overview"
       width="75%">
  <br><br>
  <b>FAME · Fairness-Aware Multimodal Embedding</b><br>
  <i>PyTorch implementation of our MLHC 2025 paper<br>
  “Equitable Electronic Health Record Prediction with FAME”</i>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2506.13104">
    <img src="https://img.shields.io/badge/arXiv-2506.13104-b31b1b.svg" alt="arXiv:2506.13104">
  </a>
  <a href="https://github.com/your-org/FAME/actions">
    <img src="https://github.com/your-org/FAME/workflows/CI/badge.svg" alt="CI status">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT license">
  </a>
  <a href="https://pytorch.org">
    <img src="https://img.shields.io/badge/PyTorch-2.1%20%2B-ee4c2c.svg" alt="PyTorch ≥ 2.1">
  </a>
</p>

---

## 📑 Table of Contents
1. [Overview](#overview)   •   2. [Key Features](#key-features) • 3. [Architecture](#architecture)  
4. [Folder Structure](#folder-structure) • 5. [Quick Start](#quick-start) • 6. [Data Preparation](#data-preparation)  
7. [Training & Evaluation](#training--evaluation) • 8. [Expected Results](#expected-results) • 9. [Custom Use](#custom-use)  
10. [Contributing](#contributing) • 11. [Changelog](#changelog) • 12. [Citation](#citation)  
13. [License](#license) • 14. [Contact](#contact)

---

## 🔍 Overview
**FAME** is a *fairness-aware* multimodal AI framework that fuses **structured EHR**, **clinical notes**, and **demographics** to make clinical predictions **without amplifying bias** across patient sub-groups (age, ethnicity, insurance).

*Core idea*: **Weight each modality by how *fair* it is.**  
During training FAME computes **EDDI (Error-Distribution Disparity Index)** and dynamically up-weights modalities that are more equitable.

The repo reproduces every experiment in the paper—baselines, ablations & the full FAME model—on public **MIMIC-III/IV** data.

---

## ✨ Key Features
| | |
|---|---|
| 🚀 **One-command pipeline** | From raw ICU tables to final metrics |
| 📈 **Automatic fairness tracking** | EDDI & Equalized-Odds logged each epoch |
| 🧰 **Out-of-the-box baselines** | DfC, AdvDebias, FPM, FairEHR-CLP |
| 🔌 **Plug-and-play modalities** | Swap encoders or add new ones (e.g. imaging) |
| 🔒 **Reproducible** | Seeds fixed, deterministic Torch ops where feasible |

---

## 🖼️ Architecture
*See the hero figure above.*  
FAME combines **BEHRT** (structured), **BioClinicalBERT** (text) & demographic embeddings. A fusion layer multiplies each modality by a learnable *fairness weight* (EDDI-guided) plus a **sigmoid gate**, then optimises a joint **BCE + β·LEDDI** loss.

---

## 📁 Folder Structure
| Path | Purpose |
|------|---------|
| `00_data.py` | Extract & preprocess MIMIC (structured + notes) |
| `01_BEHRT.py` | Baseline – BEHRT (structured-only) |
| `02_BioClinicalBERT.py` | Baseline – BioClinicalBERT (text-only) |
| `03_DfC.py` | Demographic-free baseline |
| `04_AdvDebias.py` | Adversarial debiasing baseline |
| `05_FPM.py` | Fair Patient Model baseline |
| `06_FairEHR-CLP.py` | Contrastive debiasing baseline |
| `07_multimodal_average_fusion.py` | Average-fusion ablation |
| `08_multimodal_eddi_fusion.py` | EDDI-only fusion ablation |
| `09_multimodal_sigmoid_fusion.py` | Sigmoid-only fusion ablation |
| `10_FAME.py` | **Full FAME** model |
| `docs/figures/` | Images for README / paper |
| `tests/` | Unit tests & CI scripts |
| `requirements.txt` | Python dependencies |

---

## ⚡ Quick Start
> Tested on **Python ≥ 3.9** & **PyTorch ≥ 2.1** with a single GPU (≥ 12 GB VRAM).

```bash
# 1  Clone
git clone https://github.com/your-org/FAME.git
cd FAME

# 2  (Optional) virtual env
python -m venv venv && source venv/bin/activate   # Win: venv\Scripts\activate

# 3  Install deps
pip install -r requirements.txt
# or: conda env create -f environment.yml && conda activate fame

# 4  Pre-process MIMIC (≈ 15 min)
python 00_data.py --mimic_root /path/to/mimic --out_dir data/

# 5  Train FAME on three tasks
python 10_FAME.py --tasks mortality los ventilation \
                  --lambda 0.8 --epochs 30 --bsz 32
````

CI runs `pytest` automatically on every push / PR.

---

## 🗄️ Data Preparation

1. **Download MIMIC-III v1.4 / MIMIC-IV** and pass the PhysioNet credentialing quiz.
   Place all `*.csv.gz` files in, e.g., `/mnt/mimic`.

2. Run

   ```bash
   python 00_data.py --mimic_dir /mnt/mimic --out_dir data/
   ```

   which creates:

   | CSV                                                   | What’s inside                                                              |
   | ----------------------------------------------------- | -------------------------------------------------------------------------- |
   | `final_structured_with_feature_set_C_24h_2h_bins.csv` | 2-h-binned vitals / labs + demographics                                    |
   | `unstructured_with_demographics.csv`                  | First-stay notes split into ≤512-token chunks                              |
   | *(both)*                                              | Task labels `short_term_mortality`, `los_binary`, `mechanical_ventilation` |

   Sensitive attributes (`age_bucket`, `ethnicity_category`, `insurance_category`) are added automatically.

*No raw PHI is committed—only de-identified aggregates.*

---

## 🏋️‍♂️ Training & Evaluation

All scripts share **one CLI** – just change the filename:

```bash
# 1) BEHRT (structured-only)
python 01_BEHRT.py                     --task mortality

# 2) BioClinicalBERT (text-only)
python 02_BioClinicalBERT.py           --task ventilation   --epochs 30

# 3) Demographic-free
python 03_DfC.py                       --task los

# 4) Adversarial debiasing
python 04_AdvDebias.py                 --task mortality     --alpha 2

# 5) Fair Patient Model
python 05_FPM.py                       --task los

# 6) FairEHR-CLP (contrastive)
python 06_FairEHR-CLP.py               --task ventilation   --temp 0.07

# 7) Average-fusion ablation
python 07_multimodal_average_fusion.py --task mortality

# 8) EDDI-weighted ablation
python 08_multimodal_eddi_fusion.py    --task los           --lambda 0.8

# 9) Sigmoid-only ablation
python 09_multimodal_sigmoid_fusion.py --task ventilation

# 10) ★ Full FAME
python 10_FAME.py                      --task mortality     --lambda 0.8 --epochs 50
```

### Common flags

| Flag                                   | Purpose                                 | Default        |
| -------------------------------------- | --------------------------------------- | -------------- |
| `--task {mortality, los, ventilation}` | Choose prediction head                  | `mortality`    |
| `--epochs`                             | Max epochs (early-stopping on val-loss) | `50`           |
| `--bsz`                                | Mini-batch size                         | `16`           |
| `--lr`                                 | AdamW learning-rate                     | model-specific |
| `--lambda`                             | Fairness loss weight (EDDI)             | script default |
| `--temp`                               | Contrastive temp (FairEHR-CLP)          | `0.07`         |
| `--tensorboard`                        | Stream metrics to TensorBoard           | off            |

### Outputs

| Path                              | Contents                       |
| --------------------------------- | ------------------------------ |
| `outputs/checkpoints/<run-id>.pt` | Best model (*lowest val-loss*) |
| `outputs/logs/<run-id>.csv`       | Per-epoch metrics              |
| `outputs/tensorboard/<run-id>/`   | TensorBoard events             |

Typical log line:

```
Epoch 5 │ AUROC 0.943 │ AUPRC 0.817 │ EDDI 0.44 │ EO 4.25
```

### Monitoring

```bash
tensorboard --logdir outputs/tensorboard
```

### Hardware

`1 × A100 40 GB` trains full FAME on **all three tasks in \~2.5 h**.
On smaller GPUs, reduce `--bsz`, call `--freeze_backbone`, or pre-train modalities then fine-tune fusion.

---

## 🎯 Expected Results

<details>
<summary>Paper (Table 3) – 5-run averages</summary>

| Task        | AUROC ↑  | AUPRC ↑  | EDDI % ↓ | EO % ↓   |
| ----------- | -------- | -------- | -------- | -------- |
| Mortality   | **0.94** | **0.82** | **0.44** | **4.25** |
| LOS ≥ 7 d   | **1.00** | **1.00** | **0.02** | **0.06** |
| Ventilation | **0.84** | **0.97** | **2.77** | **0.55** |

</details>

---

## 🛠️ Custom Use

1. **Dataset** – implement `CustomDataset` in `data_loader.py` returning
   `structured`, `text`, `demo`, `label` *(+ `image`, etc.)*.
2. **Encoder** – add your module in `models/encoders.py` and import it.
3. **Train** with `--image_encoder resnet50` (or your name).
   The fairness engine handles extra modalities automatically.

---

## 🤝 Contributing

PRs are welcome 🎉 – see **[`CONTRIBUTING.md`](CONTRIBUTING.md)** and abide by the **[Code of Conduct](CODE_OF_CONDUCT.md)**.

---

## 📝 Changelog

See **[`CHANGELOG.md`](CHANGELOG.md)**.

---

## 📚 Citation

```bibtex
@misc{lastname2024fame,
  title  = {Equitable Electronic Health Record Prediction with FAME: Fairness-Aware Multimodal Embedding},
  author = {Lastname, Firstname and Lastname, Firstname},
  year   = {2024},
  note   = {Machine Learning for Healthcare (under review)},
  url    = {https://github.com/your-org/FAME}
}
```

---

## ⚖️ License

Released under the **MIT License** – see [`LICENSE`](LICENSE).

---

## 📨 Contact

Questions? Open an issue or email **[your.name@uni.edu](mailto:your.name@uni.edu)**.

