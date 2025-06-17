## Convolutional Neural Networks (CNNs)

> Notes distilled from CMU 11-785 and Andrew Ng’s Coursera DL-Spec.

| What I noticed | Why it matters |
| :-- | :-- |
| **Sparse connections** | Each filter looks at a small patch → way fewer weights than a fully-connected layer. |
| **Parameter sharing** | The *same* filter slides everywhere, so the network naturally handles translations. |
| **Progressive abstraction** | Early layers pick up edges; deeper ones combine them into textures, parts, and whole objects. |

### Core building blocks (cheat-sheet)

| Layer | Job | Hyper-params I tweak most |
| :-- | :-- | :-- |
| Convolution | Grab local features | kernel size, stride, padding, number of filters |
| Activation (ReLU/GELU) | Add non-linearity | — |
| Pooling (Max/Avg) | Down-sample & add invariance | window size, stride |
| Normalization (Batch/Layer) | Stabilize training | — |
| Dropout | Fight overfitting | keep-prob |
| Fully-connected | Final decision | units |

### My go-to training workflow

1. **Data prep** – per-channel standardization + heavy augmentation (crop, flip, rotate).  
2. **Init** – He/Kaiming keeps activations sane for ReLU.  
3. **Optimizers** – SGD + momentum or Adam, with a schedule (step, cosine).  
4. **Regularization** – L2 weight decay, dropout, data aug, early-stopping.  