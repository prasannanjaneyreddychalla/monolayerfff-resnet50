# monolayerfff-resnet50

ResNet50 transfer learning for monolayer FFF image classification.

This repo builds on the IEEE paper linked in `details.txt` and implements the same general idea in a compact ResNet50 pipeline: take monolayer FFF image data, preprocess it, train a CNN-based classifier, and track how well it separates the classes during validation.

> Note: the IEEE page is the source reference for the paper and dataset. The repo keeps those links in `details.txt`; the implementation here is our ResNet50 version of that image-classification workflow.

---

## Reference paper and dataset

| Item | Detail |
|---|---|
| Paper source | IEEE Xplore link in `details.txt` |
| Dataset context | Monolayer FFF image dataset described by the linked paper |
| Task type | Image classification |
| Input type | Monolayer FFF images |
| Goal | Classify images using visual patterns learned by a CNN |

Links from `details.txt`:

- https://ieeexplore.ieee.org/abstract/document/11143237
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11143237

---

## What the paper setup gives us

The linked paper/dataset gives the project its base problem: classify monolayer FFF images using image-based learning. In plain terms, the data is visual, the signal comes from image features, and the model needs to learn patterns that separate the target classes.

That makes a CNN a natural fit. ResNet50 is useful here because it already has strong image feature extraction layers, so we do not have to train a deep network from scratch like we enjoy wasting GPUs for sport.

---

## Our implementation

```mermaid
flowchart LR
    A[Monolayer FFF images] --> B[Preprocessing]
    B --> C[ResNet50 backbone]
    C --> D[Custom classification head]
    D --> E[Training]
    E --> F[Validation accuracy]
```

Our setup uses ResNet50 as the backbone and adds a task-specific classifier head on top. The model is trained over 8 epochs, with training and validation accuracy tracked after each epoch.

---

## Paper vs our implementation

| Area | Paper / dataset reference | Our implementation |
|---|---|---|
| Problem | Monolayer FFF image classification | Same classification problem |
| Data | Dataset described in the IEEE paper | Uses the referenced monolayer FFF image data |
| Model style | Deep-learning image classification | ResNet50 transfer learning |
| Feature learning | CNN learns visual patterns from image data | ResNet50 extracts features, classifier head predicts class |
| Evaluation focus | Classification performance | Training vs validation accuracy over 8 epochs |
| Reported result here | Paper is referenced through `details.txt` | Best validation accuracy: **0.9404** |

---

## Result

The model reached a best validation accuracy of **0.9404**. Validation accuracy climbed quickly after the first epoch and stayed close to training accuracy, which suggests the model learned useful image features without obvious severe overfitting in this run.

![Training vs Validation Accuracy](assets/graph.jpeg)

| Metric | Value |
|---|---:|
| Best validation accuracy | **0.9404** |
| Best epoch marked in graph | 5 |
| Final training accuracy | ~0.912 |
| Final validation accuracy | ~0.927 |
| Total epochs | 8 |

---

## What the curve shows

| Observation | Meaning |
|---|---|
| Validation jumps from epoch 1 to 2 | The pretrained ResNet50 features adapt fast |
| Training accuracy rises steadily | The classifier keeps improving through training |
| Validation stays near training | No obvious major overfitting from this graph alone |
| Best validation score appears around epoch 5 | The strongest saved checkpoint should likely come from that region |

---

## Quick takeaway

This repo applies ResNet50 transfer learning to the monolayer FFF image-classification task referenced by the IEEE paper in `details.txt`. The current run gives a strong baseline, reaching **94.04% validation accuracy** across 8 epochs.