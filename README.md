# ML-Playground

An end-to-end sandbox for building and benchmarking machine-learning models, with most implementations coded **from scratch** in NumPy 

---

## Directory layout

| Folder | Contents | Status |
| :-- | :-- | :-- |
| `l_layer_nn/` | Fully-connected **L-layer network** (He init, ReLU/Sigmoid, GD / Momentum / Adam, mini-batch, L2, dropout) | Stable |
| `cnn/` | Convolutional layers, pooling, and helpers that will grow into a production CNN | In progress |
| `segmentation/` | Classical and modern **image-segmentation** nets (U-Net, DeepLab-v3) | Planned |
| `language_models/` | Lightweight sequence models (RNN → GRU → micro-Transformer) | Planned |

---

## Current focus

**Large-scale skin-cancer detection**

*   Extend the CNN codebase to handle dermatoscopic image sets.  
*   Package and serve the model on AWS (container in ECR, workload on EKS) with infrastructure declared in Terraform.  

---

## Roadmap

| Phase | Target |
| :-- | :-- |
| **1. CNN scale-up** | Finish training + evaluation on the skin-cancer dataset; add experiment tracking |
| **2. Segmentation** | Implement U-Net for lesion masks; compare to dense-net baseline |
| **3. Language models** | Bring up a small Transformer for text classification to round out the repo |
