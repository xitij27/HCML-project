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

The experiment builds up in four stages, each depending on the previous:

1. **Data & model** — load and prepare the LFW pairs, get ElasticFace-Arc producing
   meaningful similarity scores.
2. **Explainability** — apply an XAI method (xFace / xSSAB / FAME) and verify the
   saliency maps look sensible.
3. **Warm-up attacks** — FGSM and I-FGSM targeting the FR decision, to confirm the
   gradient pipeline works.
4. **Explanation attack (main contribution)** — perturb the input to keep the decision
   stable while pushing the saliency map away from the original.

See [approach.md](approach.md) for the full methodology, loss functions, preprocessing
details, and evaluation metrics.

## Repository contents

```
.
├── approach.md                    # Full methodology and design notes
├── elastic_resnet.py              # ElasticFace / iResNet backbone definition
├── ElasticFace_paper.pdf          # Reference paper
└── notebooks/
    ├── 01_data_and_model.ipynb    # LFW loading + ElasticFace-Arc sanity checks
    ├── 02_xai.ipynb               # Explainability exploration
    └── 03_attacks.ipynb           # Warm-up + explanation attacks
```

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
