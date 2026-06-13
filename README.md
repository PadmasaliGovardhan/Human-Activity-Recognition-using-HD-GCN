<div align="center">

<br/>

# Human Activity Recognition using HD-GCN

### *Teaching a machine to understand how humans move — using only the skeleton.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![ICCV 2023](https://img.shields.io/badge/Based%20on-ICCV%202023%20Paper-blueviolet?style=for-the-badge)](https://arxiv.org/abs/2208.10741)
[![Dataset](https://img.shields.io/badge/Dataset-NTU%20RGB%2BD%2060-orange?style=for-the-badge)]()
[![Ensemble Accuracy](https://img.shields.io/badge/Ensemble%20Top--1-78.79%25-brightgreen?style=for-the-badge)]()

<br/>

> Implemented and trained the **HD-GCN architecture** (ICCV 2023) from scratch on a **single RTX 3060** —
> processing **56,880 skeleton sequences** across **60 action classes** using a two-stream ensemble approach.

<br/>

---

</div>

## 🧠 The Idea in 30 Seconds

Strip away the camera. Strip away the background, the lighting, the clothes.
What's left? **The skeleton.** 25 joints. Pure movement.

This project asks: *can a graph neural network learn to recognize 60 human activities — walking, punching, eating, falling — just from how those 25 joints move through time?*

The answer: **78.79% Top-1 Accuracy. 96.56% Top-5.**

---

## 📌 Why This Problem Is Hard

Traditional RGB-based approaches fail in the real world because they depend on:

- 🌦️ Consistent lighting conditions
- 🧹 Clean, uncluttered backgrounds
- 👕 Clothing and appearance cues

Skeleton-based recognition throws all of that away — and forces the model to learn what *motion itself* looks like. That's the challenge. That's also what makes it powerful for real-world deployment in healthcare, surveillance, robotics, and smart environments.

---

## 🏗️ Architecture: How HD-GCN Thinks About a Body

The key insight of HD-GCN is **hierarchical graph decomposition** — the body isn't just a bag of joints; it has structure. Fingers connect to hands. Hands connect to arms. Arms connect to the spine.

HD-GCN captures this at multiple levels simultaneously:

```
Input: 3D Skeleton Sequence (T frames × 25 joints × 3 coordinates)
              │
              ▼
   ┌─────────────────────────┐
   │   HD-Graph Construction  │  ← Decomposes joints into hierarchical edge sets
   └──────────┬──────────────┘
              │
    ┌─────────┴──────────┐
    │                    │
    ▼                    ▼
┌────────────┐    ┌────────────┐
│ Joint      │    │ Bone       │
│ Stream     │    │ Stream     │
│            │    │            │
│ Raw (x,y,z)│    │ Joint_i    │
│ coordinates│    │ − Joint_j  │
└─────┬──────┘    └──────┬─────┘
      │                  │
      ▼                  ▼
  ┌───────┐          ┌───────┐
  │GCN+TCN│          │GCN+TCN│   ← Spatial + Temporal convolutions
  │Layers │          │Layers │
  └───┬───┘          └───┬───┘
      │                  │
      └────────┬─────────┘
               │  Score Fusion
               ▼
    P_final = P_joint + P_bone
               │
               ▼
        Predicted Action Class
```

**GCN** handles *spatial* relationships between joints at each frame.
**TCN** handles *temporal* evolution of those relationships across frames.
**Hierarchical Attention** learns which joint connections matter most for each action.

---

## 📊 Results

### Stream-wise Performance (Cross-Subject Protocol, NTU RGB+D 60)

| Stream | What It Sees | Top-1 Accuracy |
|---|---|---|
| 🔵 Joint Stream | Raw 3D joint coordinates | **72.63%** |
| 🟠 Bone Stream | Relative bone direction vectors | **72.43%** |
| 🟢 **Ensemble** | **Joint + Bone combined** | **78.79%** |
| ⭐ **Top-5 (Ensemble)** | — | **96.56%** |

> The **+6.16% jump** from single stream to ensemble shows the two streams capture fundamentally different and complementary information — posture vs. motion direction.

---

### How This Compares to the Field

| Method | Venue | Top-1 Accuracy |
|---|---|---|
| ST-GCN | AAAI 2018 | 81.5% |
| 2S-AGCN | CVPR 2019 | 88.5% |
| DGNN | CVPR 2019 | 89.9% |
| Shift-GCN | CVPR 2020 | 90.7% |
| MS-G3D | CVPR 2020 | 91.5% |
| CTR-GCN | ICCV 2021 | 92.4% |
| **This Work — Ensemble** | — | **78.79%** |

> 📝 **Context matters:** State-of-the-art methods use multi-GPU clusters with 200+ epoch training runs. This work ran **80 epochs on a single RTX 3060 (12GB)**. The goal was successful implementation and experimentation — and a 78.79% ensemble on constrained hardware is a legitimate result.

---

## 🗃️ Dataset

**NTU RGB+D 60** — one of the largest and most widely used benchmarks for skeleton-based action recognition.

| Property | Value |
|---|---|
| Action Classes | 60 |
| Total Samples | 56,880 |
| Subjects | 40 |
| Joints per Skeleton | 25 |
| Capture Setup | 3× Kinect V2 Cameras |
| Modality Used | 3D Skeleton Coordinates only |

60 classes span a wide range: daily activities (eating, writing), health events (falling down, chest pain), and two-person interactions (handshaking, punching).

---

## ⚙️ Training Setup

| Hyperparameter | Value |
|---|---|
| Optimizer | SGD |
| Learning Rate | 0.1 |
| Weight Decay | 0.0004 |
| Batch Size | 8 |
| Epochs | 80 |
| Window Size | 64 Frames |
| GPU | NVIDIA RTX 3060 (12 GB) |
| OS | Ubuntu 22.04 |
| Framework | PyTorch |

**Augmentations:** Random Rotation · Temporal Shifting · Skeleton Perturbation

---

## 🚀 Reproduce This

### 1. Clone & Install

```bash
git clone https://github.com/PadmasaliGovardhan/Human-Activity-Recognition-using-HD-GCN.git
cd Human-Activity-Recognition-using-HD-GCN
pip install -r requirements.txt
```

### 2. Prepare the Data

Download the NTU RGB+D 60 dataset and preprocess:

```bash
python tools/preprocess_data.py
```

### 3. Train Joint Stream

```bash
python main.py --config ./config/nturgbd60-cross-subject/joint_com_1.yaml --device 0
```

### 4. Train Bone Stream

```bash
python main.py --config ./config/nturgbd60-cross-subject/bone_com_1.yaml --device 0
```

### 5. Run Ensemble

```bash
python ensemble.py
```

---

## 📁 Repository Structure

```
Human-Activity-Recognition-using-HD-GCN/
│
├── config/                   # YAML configs for training runs
│   └── nturgbd60-cross-subject/
├── data/                     # Processed dataset files
├── feeders/                  # Data loading and augmentation
├── graph/                    # HD-Graph construction logic
├── model/                    # HD-GCN model definition
│   └── hdgcn.py
├── tools/                    # Data preprocessing utilities
├── work_dir/                 # Training logs and saved checkpoints
│
├── train.py                  # Training entry point
├── ensemble.py               # Ensemble score fusion
├── requirements.txt
└── README.md
```

---

## 🗺️ What's Next

This is V1 — a full implementation of the ICCV 2023 paper under constrained hardware. The roadmap ahead:

- [ ] 🔥 **Multi-GPU training** — scale to NTU RGB+D 120 with full epochs
- [ ] 🎯 **Hyperparameter optimization** — push ensemble accuracy further
- [ ] 👁️ **Attention visualization** — heatmaps showing which joints the model focuses on per action
- [ ] 📱 **Real-time inference** — live skeleton feed via MediaPipe + webcam
- [ ] 🤝 **Pose estimation integration** — end-to-end pipeline from raw video
- [ ] 📦 **Model export** — ONNX / TensorRT deployment

---

## 📄 Reference Paper

This project implements:

> **Lee et al., "Hierarchically Decomposed Graph Convolutional Networks for Skeleton-Based Action Recognition"**
> *IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 10444–10453*

```bibtex
@InProceedings{Lee_2023_ICCV,
  author    = {Lee, Jungho and Lee, Minhyeok and Lee, Dogyoon and Lee, Sangyoun},
  title     = {Hierarchically Decomposed Graph Convolutional Networks for Skeleton-Based Action Recognition},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2023},
  pages     = {10444-10453}
}
```

---

## 👨‍💻 Authors

**P. Govardhan** · Electronics & Communication Engineering, KL University

> B.Tech final year project · Supervised by Dr. S. Vamsee Krishna · Co-authored with K. Santha Vardhan

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-PadmasaliGovardhan-181717?style=flat&logo=github)](https://github.com/PadmasaliGovardhan)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Padmasali%20Govardhan-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/YOUR-LINKEDIN-ID)

<br/>

*Open to roles in AI/ML, Data Science, and Decision Intelligence.*
*If this project resonates with what your team builds — let's talk.*

[![Email](https://img.shields.io/badge/Email-padmasali.govardhan.p@gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:padmasali.govardhan.p@gmail.com)

---

<div align="center">

**Built under resource constraints. Designed to understand them.**

*If this helped you — a ⭐ keeps the momentum going.*

</div>
