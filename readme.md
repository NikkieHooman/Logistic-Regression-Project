<!-- ==================================================================== -->
<!--  README for FAME: Fairness-Aware Multimodal Embedding                -->
<!-- ==================================================================== -->

<p align="center">
  <img width="1075"
       alt="FAME architecture"
       src="https://github.com/user-attachments/assets/b3f603fa-b594-4723-bf5f-d8a5bacbf384" />
  <br>
  <b>FAME · Fairness-Aware Multimodal Embedding</b><br>
  <i>PyTorch implementation of our MLHC 2025 paper<br>
     “Equitable Electronic Health Record Prediction with FAME”</i>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2506.13104">
    <img src="https://img.shields.io/badge/arXiv-2506.13104-b31b1b">
  </a>
  <a href="https://github.com/&lt;your-org&gt;/FAME/actions">
    <img src="https://github.com/&lt;your-org&gt;/FAME/workflows/CI/badge.svg">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green">
  </a>
  <a href="https://pytorch.org">
    <img src="https://img.shields.io/badge/PyTorch-2.1%20%2B-ee4c2c">
  </a>
</p>


---

## 📜 Table of Contents
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

> **Key idea – EDDI-weighted fusion**  
> Each epoch we compute **EDDI** (Error-Distribution Disparity Index) per modality  
> and *up-weight* modalities that are more equitable, while down-weighting biased ones.

This repository reproduces every experiment in the paper—baselines, ablations, and the full model—on **MIMIC-III / IV**.

---

## 2 · Key Features
* 🛠 **One-command pipeline** – from raw CSV to final metrics.  
* 📈 **Built-in fairness logging** – EDDI & Equalized Odds every epoch.  
* 🏗 **Rich baselines** – DfC, AdvDebias, FPM, FairEHR-CLP.  
* 🔌 **Plug-and-play modalities** – add imaging, waveforms, etc.  
* ♻ **Reproducible** – fixed seeds, deterministic ops where possible.

---

## 3 · Architecture
<p align="center">
  <img src="docs/figures/fame_architecture.png" width="820">
</p>

**Figure 1 –** BEHRT encodes structured EHR; BioClinicalBERT encodes notes;  
demographics use lightweight embeddings. A fusion layer applies **EDDI weights × sigmoid gate** and optimises a joint **BCE + EDDI** loss.

---

## 4 · Folder Structure

| Path / File | Description |
|-------------|-------------|
| `00_data.py` | Extract & preprocess MIMIC tables + notes |
| `01_BEHRT.py` → `09_multimodal_sigmoid_fusion.py` | Baselines & ablations |
| `10_FAME.py` | **Full FAME model** |
| `docs/` | Figures & supplementary docs |
| `tests/` | Unit tests (run via CI) |
| `requirements.txt` / `environment.yml` | Python / Conda deps |

---

## 5 · Quick Start
<details open>
<summary><b>30-second setup (single GPU ≥ 12 GB)</b></summary>

```bash
# 1 · Install
git clone https://github.com/<your-org>/FAME.git && cd FAME
pip install -r requirements.txt          # or: conda env create -f environment.yml

# 2 · Place raw MIMIC-III v1.4 CSVs in /mnt/mimic   (requires PhysioNet credential)

# 3 · Pre-process  (≈15 min on SSD)
python 00_data.py --mimic_dir /mnt/mimic --out_dir data/

# 4 · Train full FAME on all three tasks
python 10_FAME.py --tasks mortality los ventilation \
                  --lambda 0.8 --epochs 30 --bsz 32 --tensorboard
