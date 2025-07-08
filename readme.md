````markdown
<!-- ================================================================ -->
<!--  README – FAME: Fairness-Aware Multimodal Embedding              -->
<!--  Copy-paste this file at the root of your repository             -->
<!-- ================================================================ -->

<p align="center">
  <img src="docs/figures/fame_hero.png"
       alt="FAME architecture overview"
       width="75%"><br><br>

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
    <img src="https://img.shields.io/badge/PyTorch-2.1%20%2B-ee4c2c.svg" alt="PyTorch ≥2.1">
  </a>
</p>

---

## 📑 Table of Contents
1. [Overview](#overview)   &bull;   2. [Key Features](#key-features)   &bull;   3. [Architecture](#architecture)  
4. [Folder Structure](#folder-structure)   &bull;   5. [Quick Start](#quick-start)   &bull;   6. [Data Preparation](#data-preparation)  
7. [Training & Evaluation](#training--evaluation)   &bull;   8. [Expected Results](#expected-results)   &bull;   9. [Custom Use](#custom-use)  
10. [Contributing](#contributing)   &bull;   11. [Changelog](#changelog)   &bull;   12. [Citation](#citation)  
13. [License](#license)   &bull;   14. [Contact](#contact)

---

## 🔍 Overview
**FAME** is a *fairness-aware* multimodal AI framework that fuses **structured EHR**, **clinical notes**, and **demographics** to make clinical predictions **without amplifying bias** across patient sub-groups (age, ethnicity, insurance).

*Core insight*: **Weight each modality by how *fair* it is.**  
During training FAME computes **EDDI (Error-Distribution Disparity Index)** and dynamically up-weights modalities that are more equitable, while down-weighting unfair ones.

This repo reproduces every experiment in the paper—baselines, ablations, and the full FAME model—using public **MIMIC-III/IV** data.

---

## ✨ Key Features
| &nbsp;| Description |
|---|---|
| 🚀 **One-command pipeline** | From raw ICU tables to final metrics |
| 📈 **Automatic fairness tracking** | EDDI & Equalized-Odds logged every epoch |
| 🧰 **Out-of-the-box baselines** | DfC, AdvDebias, FPM, FairEHR-CLP |
| 🔌 **Plug-and-play modalities** | Swap encoders or add new ones (e.g. imaging) |
| 🔒 **Reproducible** | Seeds fixed, deterministic Torch ops where feasible |

---

## 🖼️ Architecture
*(See hero diagram above.)*  
FAME combines **BEHRT** (structured sequence), **BioClinicalBERT** (clinical text), and demographic embeddings. The **fusion layer** multiplies each modality by a learnable *fairness weight* (EDDI-guided) and a **sigmoid gate** before a joint **BCE + β·LEDDI** loss.

---

## 📁 Folder Structure
| File / Dir | Purpose |
|------------|---------|
| `00_data.py` | Extract & pre-process MIMIC (structured + notes) |
| `01_BEHRT.py` | Baseline – BEHRT (structured-only) |
| `02_BioClinicalBERT.py` | Baseline – BioClinicalBERT (text-only) |
| `03_DfC.py` | Demographic-free baseline |
| `04_AdvDebias.py` | Adversarial debiasing baseline |
| `05_FPM.py` | Fair Patient Model baseline |
| `06_FairEHR-CLP.py` | Contrastive debiasing baseline |
| `07_multimodal_average_fusion.py` | Average fusion ablation |
| `08_multimodal_eddi_fusion.py` | EDDI-only fusion ablation |
| `09_multimodal_sigmoid_fusion.py` | Sigmoid-only fusion ablation |
| `10_FAME.py` | **Full FAME** model |
| `docs/figures/` | All images for the README / paper |
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
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate

# 3  Install deps
pip install -r requirements.txt
#    or:
# conda env create -f environment.yml && conda activate fame

# 4  Pre-process MIMIC (≈ 15 min)
python 00_data.py --mimic_root /path/to/mimic --out_dir data/

# 5  Train FAME on three tasks
python 10_FAME.py --tasks mortality los ventilation \
                  --lambda 0.8 --epochs 30 --bsz 32
````

The CI workflow automatically runs `pytest` on push / PR.

---

## 🗄️ Data Preparation

1. **Get MIMIC-III v1.4 / MIMIC-IV**  ▶  finish the PhysioNet credentialing quiz, then place the CSV \*.gz files in `/mnt/mimic`.

2. Run

   ```bash
   python 00_data.py --mimic_dir /mnt/mimic --out_dir data/
   ```

   which produces:

   | CSV                                                   | What’s inside                                                              |
   | ----------------------------------------------------- | -------------------------------------------------------------------------- |
   | `final_structured_with_feature_set_C_24h_2h_bins.csv` | 2-h binned vitals/labs + demographics                                      |
   | `unstructured_with_demographics.csv`                  | First-stay notes split into ≤512-token chunks                              |
   | *(both)*                                              | Task labels `short_term_mortality`, `los_binary`, `mechanical_ventilation` |

   **Sensitive attributes** (`age_bucket`, `ethnicity_category`, `insurance_category`) are added automatically.

*No raw PHI is ever committed—only de-identified aggregates.*

---

## 🏋️‍♂️ Training & Evaluation

All experiment scripts share the *same* CLI.
Swap the filename to run a different baseline / ablation / full FAME:

```bash
# 1)  Unimodal baseline – BEHRT
python 01_BEHRT.py                     --task mortality

# 2)  Text-only baseline – BioClinicalBERT
python 02_BioClinicalBERT.py           --task ventilation   --epochs 30

# 3)  Demographic-free baseline
python 03_DfC.py                       --task los

# 4)  Adversarial debiasing baseline
python 04_AdvDebias.py                 --task mortality     --alpha 2

# 5)  Fair Patient Model
python 05_FPM.py                       --task los

# 6)  Contrastive debiasing (FairEHR-CLP)
python 06_FairEHR-CLP.py               --task ventilation   --temp 0.07

# 7)  Average-fusion ablation
python 07_multimodal_average_fusion.py --task mortality

# 8)  EDDI-weighted ablation
python 08_multimodal_eddi_fusion.py    --task los           --lambda 0.8

# 9)  Sigmoid-gate ablation
python 09_multimodal_sigmoid_fusion.py --task ventilation

# 10) ★ Full FAME
python 10_FAME.py                      --task mortality     --lambda 0.8 --epochs 50
```

### Common flags

| Flag                                   | Meaning                                 | Default        |
| -------------------------------------- | --------------------------------------- | -------------- |
| `--task {mortality, los, ventilation}` | Choose prediction head                  | `mortality`    |
| `--epochs`                             | Max epochs (early-stopping on val-loss) | `50`           |
| `--bsz`                                | Mini-batch size                         | `16`           |
| `--lr`                                 | AdamW learning-rate                     | model-specific |
| `--lambda`                             | Fairness loss weight (EDDI)             | script default |
| `--temp`                               | Contrastive temp (FairEHR-CLP only)     | `0.07`         |
| `--tensorboard`                        | Stream metrics to TensorBoard           | *off*          |

Run *any* script with `-h` for full options.

### Outputs

| Path                              | Contents                       |
| --------------------------------- | ------------------------------ |
| `outputs/checkpoints/<run-id>.pt` | Best model (*lowest val-loss*) |
| `outputs/logs/<run-id>.csv`       | Per-epoch metrics              |
| `outputs/tensorboard/<run-id>/`   | TensorBoard events (if on)     |

Typical log line:

```
Epoch 5 │ AUROC 0.943 │ AUPRC 0.817 │ EDDI 0.44 │ EO 4.25
```

### Monitoring

```bash
tensorboard --logdir outputs/tensorboard
```

### Hardware notes

*FAME* back-propagates through **BEHRT + BioClinicalBERT** *plus* the fairness loss.
`1 × A100 40 GB` trains the full model on **all three tasks in \~2 ½ h**.
For lighter GPUs: lower `--bsz`, use `--freeze_backbone`, or pre-train each modality then fine-tune fusion.

---

## 🎯 Expected Results <a name="expected-results"></a>

<details>
<summary>5-run average (Table 3 of the paper)</summary>

| Task                   | AUROC ↑  | AUPRC ↑  | EDDI % ↓ | EO % ↓   |
| ---------------------- | -------- | -------- | -------- | -------- |
| Mortality              | **0.94** | **0.82** | **0.44** | **4.25** |
| LOS ≥ 7 days           | **1.00** | **1.00** | **0.02** | **0.06** |
| Mechanical Ventilation | **0.84** | **0.97** | **2.77** | **0.55** |

</details>

---

## 🛠️ Custom Use

*Bring FAME to your own dataset (or add imaging):*

1. **Create** a `CustomDataset` in `data_loader.py` that returns a dict with keys
   `structured`, `text`, `demo`, `label` *(+ `image` if you add imaging)*.
2. **Add** your encoder in `models/encoders.py` and register it in `__init__.py`.
3. **Train** with e.g. `--image_encoder resnet50`.
   The fairness engine automatically handles the extra modality.

---

## 🤝 Contributing

Contributions are welcome 🎉
Please read **[`CONTRIBUTING.md`](CONTRIBUTING.md)** for guidelines, branch strategy, and the local test suite. By contributing you agree to abide by our **[Code of Conduct](CODE_OF_CONDUCT.md)**.

---

## 📝 Changelog

See **[`CHANGELOG.md`](CHANGELOG.md)** for a curated list of updates.

---

## 📚 Citation

If FAME helps your research, please cite us:

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

Questions or ideas? Open an issue or email **[your.name@uni.edu](mailto:your.name@uni.edu)**.

*Happy (fair) modelling ♥️*

```
```
