Below is a *drop-in* **`README.md`** that follows the polished style you see in well-maintained ML research repos (e.g. Segment Anything, LLaMA, DINOv2).
Just copy–paste it over the existing file.  <br>*Feel free to tweak links, author names, and figures paths.*

````markdown
<!-- --------------------------------------------------------------------- -->
<!--  README for FAME: Fairness-Aware Multimodal Embedding                  -->
<!-- --------------------------------------------------------------------- -->

<p align="center">
  <img src="docs/figures/fame_logo.svg" height="140"><br>
  <b>FAME · Fairness-Aware Multimodal Embedding</b><br>
  <i>PyTorch implementation of our MLHC 2025 paper<br>
  “Equitable Electronic Health Record Prediction with FAME”</i>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2506.13104"><img src="https://img.shields.io/badge/arXiv-2506.13104-b31b1b"></a>
  <a href="https://github.com/NikkieHooman/FAME/actions"><img src="https://github.com/NikkieHooman/FAME/workflows/CI/badge.svg"></a>
  <a href="https://github.com/NikkieHooman/FAME/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.1%20%2B-ee4c2c"></a>
</p>

---

## 🌐 Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Folder Structure](#folder-structure)
5. [Quick Start](#quick-start)
6. [Data Preparation](#data-preparation)
7. [Training & Evaluation](#training--evaluation)
8. [Expected Results](#expected-results)
9. [Custom Use](#custom-use)
10. [Contributing](#contributing)
11. [Changelog](#changelog)
12. [Citation](#citation)
13. [License](#license)
14. [Contact](#contact)

---

## 1 · Overview
**FAME** is a *fairness-aware* multimodal framework that fuses **structured EHR**, **clinical notes**, and **demographics** to make ICU predictions **without amplifying bias** across patient sub-groups (age, ethnicity, insurance).

> **Core idea – EDDI-weighted fusion.**  
> Each epoch we compute **EDDI** (Error-Distribution Disparity Index) per modality
> and *up-weight* the fairer modalities while down-weighting biased ones.

The repo reproduces every experiment in the paper—baselines, ablations, and the full FAME model—on **MIMIC-III / IV**.

---

## 2 · Key Features
* 🛠 **One-command pipeline** – from raw CSV to final metrics.  
* 📈 **Built-in fairness logging** – EDDI & Equalized Odds every epoch.  
* 🏗 **Out-of-the-box baselines** – DfC, AdvDebias, FPM, FairEHR-CLP.  
* 🔌 **Plug-and-play modalities** – swap encoders or add imaging, waveforms…  
* ♻ **Reproducible** – seeds set, deterministic ops where possible.

---

## 3 · Architecture
<p align="center">
  <img src="docs/figures/fame_architecture.png" width="800">
</p>

**Figure 1 –** BEHRT encodes structured EHR; BioClinicalBERT encodes notes;  
demographics use lightweight embeddings. A fusion layer applies **EDDI weights × sigmoid gate** and optimises a joint **BCE + EDDI** loss.

---

## 4 · Folder Structure

| Path / File | Purpose |
|-------------|---------|
| `00_data.py` | Extract & preprocess MIMIC tables + notes |
| `01_BEHRT.py` | Structured-only baseline |
| `02_BioClinicalBERT.py` | Text-only baseline |
| `03_DfC.py` | Demographic-free classifier |
| `04_AdvDebias.py` | Adversarial debiasing baseline |
| `05_FPM.py` | Fair Patient Model |
| `06_FairEHR-CLP.py` | Contrastive debiasing |
| `07_multimodal_average_fusion.py` | Average-fusion ablation |
| `08_multimodal_eddi_fusion.py` | EDDI-only fusion |
| `09_multimodal_sigmoid_fusion.py` | Sigmoid-only fusion |
| `10_FAME.py` | **Full FAME** |
| `docs/` | Figures & docs |
| `tests/` | Unit tests / CI |
| `requirements.txt` | Python deps |

---

## 5 · Quick Start
<details open>
<summary><b>30-second setup (1 GPU ≥ 12 GB)</b></summary>

```bash
# 1 · Install
git clone https://github.com/NikkieHooman/FAME.git && cd FAME
pip install -r requirements.txt               # or conda env create -f environment.yml

# 2 · Put raw MIMIC-III v1.4 CSVs in /mnt/mimic   (requires PhysioNet credential)

# 3 · Pre-process  (≈ 15 min on SSD)
python 00_data.py --mimic_dir /mnt/mimic --out_dir data/

# 4 · Train full FAME on all 3 tasks
python 10_FAME.py --tasks mortality los ventilation \
                  --lambda 0.8 --epochs 30 --bsz 32 --tensorboard
````

</details>

> **Tip:** CI runs `pytest` automatically on every push / PR.

---

## 6 · Data Preparation

1. Download **MIMIC-III v1.4** or **MIMIC-IV** CSVs → `/mnt/mimic`.
2. Run:

```bash
python 00_data.py --mimic_dir /mnt/mimic --out_dir data/
```

This creates:

| File (in `data/`)                                     | Contents                                       |
| ----------------------------------------------------- | ---------------------------------------------- |
| `final_structured_with_feature_set_C_24h_2h_bins.csv` | 2 h-binned labs + vitals + demographics        |
| `unstructured_with_demographics.csv`                  | First-stay notes split into ≤ 512-token chunks |

Both include task labels **`short_term_mortality`**, **`los_binary`** (> 7 d), **`mechanical_ventilation`** and sensitive attrs `age_bucket`, `ethnicity_category`, `insurance_category`.

*No PHI is committed to the repo – outputs are fully de-identified.*

---

## 7 · Training & Evaluation

Each script shares the **same CLI** – change only the filename.

```bash
# 1) Structured baseline – BEHRT
python 01_BEHRT.py                     --task mortality

# 2) Text baseline – BioClinicalBERT
python 02_BioClinicalBERT.py           --task ventilation --epochs 30

# 3) Demographic-free
python 03_DfC.py                       --task los

# 4) Adversarial debiasing
python 04_AdvDebias.py                 --task mortality --alpha 2

# 5) Fair Patient Model
python 05_FPM.py                       --task los

# 6) Contrastive debiasing (FairEHR-CLP)
python 06_FairEHR-CLP.py               --task ventilation --temp 0.07

# 7) Average-fusion ablation
python 07_multimodal_average_fusion.py --task mortality

# 8) EDDI-only fusion (no sigmoid)
python 08_multimodal_eddi_fusion.py    --task los --lambda 0.8

# 9) Sigmoid-only fusion (no EDDI)
python 09_multimodal_sigmoid_fusion.py --task ventilation

# 10) ★ Full FAME
python 10_FAME.py                      --task mortality --lambda 0.8 --epochs 50
```

### Common flags

| Flag                                 | Purpose                      | Default        |
| ------------------------------------ | ---------------------------- | -------------- |
| `--task {mortality,los,ventilation}` | Select prediction head       | `mortality`    |
| `--epochs N`                         | Max epochs (early-stop)      | `50`           |
| `--bsz N`                            | Batch size                   | `16`           |
| `--lr FLOAT`                         | AdamW LR                     | model-specific |
| `--lambda FLOAT`                     | EDDI loss weight             | script default |
| `--temp FLOAT`                       | Contrastive temperature (06) | `0.07`         |
| `--tensorboard`                      | Live metrics                 | off            |

### Outputs

```
outputs/
 ├─ checkpoints/<run>.pt       # best model (min val-loss)
 ├─ logs/<run>.csv             # per-epoch metrics
 └─ tensorboard/<run>/         # TB event files   (if --tensorboard)
```

Example log line

```
Epoch 5 │ AUROC 0.943 │ AUPRC 0.817 │ EDDI 0.44 │ EO 4.25
```

### Hardware

* 1 × A100 40 GB → full FAME in \~2 h 30 m.
* On ≤ 16 GB cards: use `--bsz 8`, add `--freeze_backbone`, or pre-train encoders separately.

---

## 8 · Expected Results

<details>
<summary>Five-run averages (Table 3 of the paper)</summary>

| Task            |  AUROC ↑ |  AUPRC ↑ |     EDDI ↓ |       EO ↓ |
| --------------- | -------: | -------: | ---------: | ---------: |
| **Mortality**   | **0.94** | **0.82** | **0.44 %** | **4.25 %** |
| **LOS ≥ 7 d**   | **1.00** | **1.00** | **0.02 %** | **0.06 %** |
| **Ventilation** | **0.84** | **0.97** | **2.77 %** | **0.55 %** |

</details>

---

## 9 · Custom Use

Want to add imaging or your own dataset?

1. **Dataset** – implement `CustomDataset` that returns a `dict` with keys
   `structured`, `text`, `demo`, `label` *(+ your new key, e.g. `image`)*.
2. **Encoder** – add your model in `models/encoders.py` and register it.
3. **Run** –

   ```bash
   python 10_FAME.py --image_encoder resnet50 --task mortality …
   ```

The fairness engine handles extra modalities automatically.

---

## 10 · Contributing

PRs 🆗! Please read **[CONTRIBUTING.md](CONTRIBUTING.md)** and follow our **[Code of Conduct](CODE_OF_CONDUCT.md)**.

---

## 11 · Changelog

See **[CHANGELOG.md](CHANGELOG.md)** for release notes.

---

## 12 · Citation

```bibtex
@misc{lastname2024fame,
  title  = {Equitable Electronic Health Record Prediction with FAME: Fairness-Aware Multimodal Embedding},
  author = {Lastname, Firstname and Lastname, Firstname},
  year   = {2024},
  note   = {Machine Learning for Healthcare (under review)},
  url    = {https://github.com/NikkieHooman/FAME}
}
```

---

## 13 · License

**MIT** – see [`LICENSE`](LICENSE).

---

## 14 · Contact

Open an issue or email **name @ email.edu** – we’re happy to help.

<p align="center"><i>Happy (fair) modelling! 🩺✨</i></p>
```
