# Adversarial Manipulation of Explanations in Face Recognition

An HCML bonus project investigating whether an adversary can craft a small,
imperceptible perturbation to a face image that **keeps the face-recognition decision
intact but changes the saliency map** to point at irrelevant regions.

This is an attack on the *explanation*, not on the classifier — a largely unexplored
problem distinct from standard adversarial attacks in biometrics, which try to flip the
decision itself. Explanations matter in practice: a forensic expert at a border crossing
or in a court case relies on the saliency map to judge whether the system's reasoning is
trustworthy.

## Pipeline

The experiment builds up in stages, each depending on the previous:

1. **Data & model** — load and prepare the LFW pairs, get ElasticFace-Arc producing
   meaningful similarity scores.
2. **Explainability** — apply an XAI method (xFace / xSSAB / FAME) and verify the
   saliency maps look sensible.
3. **Warm-up attacks** — FGSM and I-FGSM targeting the FR decision, to confirm the
   gradient pipeline works.
4. **Explanation attack (main contribution)** — perturb the input to keep the decision
   stable while pushing the saliency map away from the original.
5. **Corrected & improved explanation attack** (`04`) — fixes a gradient-sign bug in
   stage 4, restricts the attack to genuine (accepted-match) pairs, and optimises a
   scale-invariant correlation objective with a hard decision constraint.
6. **Cross-method transfer** (`05`) — implements an xSSAB-style explanation and tests
   whether an attack crafted on the gradient map transfers to xSSAB and xFace.

See [approach.md](approach.md) for the full methodology, loss functions, preprocessing
details, and evaluation metrics.

## Repository contents

```
.
├── approach.md                    # Full methodology and design notes
├── elastic_resnet.py              # ElasticFace / iResNet backbone definition
├── ElasticFace_paper.pdf          # Reference paper
└── notebooks/
    ├── 01-data-and-model.ipynb        # LFW loading + ElasticFace-Arc sanity checks
    ├── 02-xai.ipynb                   # Explainability exploration
    ├── 03-attacks.ipynb               # Warm-up + explanation attacks (has the sign bug; see 04)
    ├── 04-explanation-attack-v2.ipynb # Corrected & improved explanation attack (genuine pairs)
    └── 05-xssab-and-transfer.ipynb    # xSSAB-style XAI + cross-method attack transfer
```

> **Note on notebook 03.** Its warm-up attacks all report `flip_rate = 0.00` because
> the I-FGSM update ascends the loss while the loss functions already carry the attack
> sign, so similarity moves *away* from the decision boundary. The explanation attack
> inherits the same inverted step (it makes the saliency map *more* similar and degrades
> the decision). Notebook 04 rewrites everything as gradient **descent on a loss to
> minimise** and fixes the objective/metric mismatch (it maximised MSE but evaluated
> Pearson correlation, two different things). Treat 03 as the buggy baseline and 04 as
> the corrected version.

The notebooks are structured to run on **Kaggle**.

## Data & model weights

Large files are **not** tracked in this repository (see [.gitignore](.gitignore)). The
model weights (`ElasticFaceArc.pth`) and the LFW image set (`imgs/`) come from the
Fraunhofer owncloud link:

```
https://owncloud.fraunhofer.de/index.php/s/OoLxFvESM9aUkR9
```

On Kaggle, upload these as a dataset and point the notebooks at the attached path.

### LFW pair layout

Images are named sequentially (`0000.png`, `0001.png`, …) and grouped in pairs
(image 0 + image 1, image 2 + image 3, …):

- Pairs 0–299: genuine (same person)
- Pairs 300–599: imposter (different people)
- Pairs 600–899: genuine

Genuine pairs use an obfuscation attack (push embeddings apart); imposter pairs use an
impersonation attack (pull embeddings together).

## Dependencies

```bash
pip install torch torchvision numpy opencv-python matplotlib Pillow timm scipy
```

A CUDA-capable GPU is assumed. Without one, replace `.to("cuda")` calls with `.to("cpu")`
and expect slow adversarial optimisation.

## Key references

- **ElasticFace** — https://arxiv.org/abs/2109.09416 · [code](https://github.com/fdbtrs/ElasticFace)
- **xFace** — [WACVW 2023](https://openaccess.thecvf.com/content/WACV2023W/XAI4B/html/Knoche_Explainable_Model-Agnostic_Similarity_and_Confidence_in_Face_Verification_WACVW_2023_paper.html) · [code](https://github.com/Martlgap/x-face-verification)
- **xSSAB** — https://arxiv.org/abs/2304.13409 · [code](https://github.com/marcohuber/xSSAB)
- **FAME** — [CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/XAI4CV/papers/Zhang_FAME_Feature_Activation_Map_Explanation_on_Image_Classification_and_Face_CVPRW_2026_paper.pdf)
- Heo et al. 2019, *Fooling Neural Network Interpretations via Adversarial Model Manipulation* (NeurIPS) — closest conceptual predecessor.
- Adebayo et al. 2018, *Sanity Checks for Saliency Maps* — background on saliency fragility.
