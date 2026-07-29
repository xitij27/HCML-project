# Adversarial Manipulation of Explanations in Face Recognition

## What this project is about

Face recognition (FR) systems are increasingly accompanied by explainability methods that
produce saliency maps, highlighting which facial regions the model used to reach its decision.
These explanations matter in practice: a forensic expert at a border crossing or in a court
case will look at the saliency map to verify that the system's reasoning is trustworthy.

This project investigates whether an adversary can craft a small, imperceptible perturbation
to a face image that keeps the FR decision intact but changes the saliency map to point at
irrelevant regions. This is an attack on the explanation, not on the classifier. It is a
different and largely unexplored problem from standard adversarial attacks in biometrics,
which all try to flip the decision itself.

The pipeline builds up in four stages, each one depending on the previous:

1. Load and prepare the LFW dataset and verify the data structure
2. Get ElasticFace-Arc running and confirm you get meaningful similarity scores
3. Apply an XAI method and check that the explanations look sensible
4. Run adversarial attacks, first targeting the FR decision as a warm-up, then targeting
   the explanation while keeping the decision stable (the main contribution)

---

## Repository structure

```
project/
├── approach.md                  # This file
├── requirements.txt
├── data/
│   └── lfw/                     # LFW images downloaded from Fraunhofer owncloud
│       ├── 0000.png
│       ├── 0001.png
│       └── ...
├── models/
│   └── elasticface_arc.pth      # Downloaded from Fraunhofer owncloud
├── src/
│   ├── data_loader.py           # Pair loading and preprocessing
│   ├── fr_model.py              # ElasticFace-Arc wrapper
│   ├── xai/
│   │   ├── xface.py             # xFace explainability
│   │   ├── xssab.py             # xSSAB explainability
│   │   └── fame.py              # FAME explainability (most recent)
│   ├── attacks/
│   │   ├── fgsm.py              # FGSM on FR decision (warm-up)
│   │   ├── ifgsm.py             # I-FGSM on FR decision (warm-up)
│   │   └── explanation_attack.py # Main contribution: attack the explanation
│   └── evaluate.py              # Metrics and visualisation
└── experiments/
    ├── 01_data_and_model.ipynb
    ├── 02_xai_exploration.ipynb
    └── 03_adversarial_attacks.ipynb
```

---

## Downloads

Both the model weights and the dataset come from the same Fraunhofer owncloud link:

```
https://owncloud.fraunhofer.de/index.php/s/OoLxFvESM9aUkR9
```

Place the model file at `models/elasticface_arc.pth` and the LFW images under `data/lfw/`.

---

## Dependencies

```
torch>=2.0
torchvision
numpy
opencv-python
matplotlib
Pillow
timm
scipy
```

Install with:

```bash
pip install torch torchvision numpy opencv-python matplotlib Pillow timm scipy
```

The project assumes a CUDA-capable GPU. If you are running without one, replace all
`.to("cuda")` calls with `.to("cpu")` but expect slow adversarial optimisation.

---

## Stage 1: Data loading and structure

### How the LFW pairs are organised

The images are named sequentially: 0000, 0001, 0002, ... and they are always in pairs.
Image 0 and image 1 form a pair, image 2 and image 3 form a pair, and so on.

The ordering within the dataset is:

- Images 0000 to 0599: 300 genuine pairs (same person, should match)
- Images 0600 to 1199: 300 imposter pairs (different people, should not match)
- Images 1200 to 1799: another 300 genuine pairs

This ordering matters when you set up the loss function for adversarial attacks, because
genuine pairs need an obfuscation attack (push embeddings apart) while imposter pairs need
an impersonation attack (pull embeddings together).

### Preprocessing

All images should be preprocessed before passing into the FR model:

```python
import numpy as np
import torch

def preprocess_image(img):
    # img should be a numpy array with shape (H, W, 3) and values in [0, 255]
    img = ((img / 255) - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1)).astype('float32')
    img = torch.tensor(img).to("cuda")
    return img
```

The images are already pre-aligned and cropped to 112x112, so no resizing is needed.

### Suggested data loader

```python
import os
import cv2

def load_pair(data_dir, idx):
    # idx is the pair index (0-indexed), so pair 0 is images 0 and 1
    img_a = cv2.imread(os.path.join(data_dir, f"{idx*2:04d}.png"))
    img_b = cv2.imread(os.path.join(data_dir, f"{idx*2+1:04d}.png"))
    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
    return preprocess_image(img_a), preprocess_image(img_b)

def is_genuine_pair(pair_idx):
    # Returns True if the pair is a genuine (same person) pair
    # Pairs 0-299: genuine, 300-599: imposter, 600-899: genuine
    if pair_idx < 300:
        return True
    elif pair_idx < 600:
        return False
    else:
        return True
```

---

## Stage 2: Face recognition model

### ElasticFace-Arc

The model is ElasticFace-Arc, a ResNet-based face recognition model trained with an elastic
ArcFace loss. The paper is at https://arxiv.org/abs/2109.09416 and the code is at
https://github.com/fdbtrs/ElasticFace.

The model takes a batch of 112x112 face images and outputs a 512-dimensional embedding
vector. Similarity between two images is computed as the cosine similarity of their embeddings.
A pair is considered a match if the similarity exceeds a threshold (typically around 0.25 to
0.35 on LFW, check the ElasticFace paper for the reported threshold).

### Loading the model

```python
import torch
import torch.nn as nn

# The ElasticFace repo uses iresnet from their own codebase.
# Clone the repo or copy the backbone definition, then:

from backbone import iresnet100  # from the ElasticFace repo

def load_elasticface(weights_path, device="cuda"):
    model = iresnet100(num_features=512)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def get_embedding(model, img_tensor):
    # img_tensor: (1, 3, 112, 112) or (B, 3, 112, 112)
    with torch.no_grad():
        embedding = model(img_tensor.unsqueeze(0) if img_tensor.dim() == 3 else img_tensor)
    return embedding

def cosine_similarity(e1, e2):
    return torch.nn.functional.cosine_similarity(e1, e2, dim=1)
```

### Sanity check

Before moving on, verify that genuine pairs score clearly higher than imposter pairs on
the 900 pairs in the dataset. You should see a clear separation between the two distributions
when you plot them as a histogram.

---

## Stage 3: Explainability methods

Three methods are available. Start with one and add a second later for comparison.

### Option A: xFace

Paper: https://openaccess.thecvf.com/content/WACV2023W/XAI4B/html/Knoche_Explainable_Model-Agnostic_Similarity_and_Confidence_in_Face_Verification_WACVW_2023_paper.html
Code: https://github.com/Martlgap/x-face-verification

xFace is model-agnostic. It works by masking different regions of the input face and
measuring how much the similarity score drops when each region is hidden. Regions that
cause a large drop when masked are considered important. The output is a saliency map of
the same size as the input image.

### Option B: xSSAB

Paper: https://arxiv.org/abs/2304.13409
Code: https://github.com/marcohuber/xSSAB

Note: this is your supervisor's own method, so the code is likely well maintained and you
can ask directly if something is unclear. xSSAB is gradient-based and adapted for the face
verification setting rather than classification.

### Option C: FAME (most recent, recommended as second method)

Paper: https://openaccess.thecvf.com/content/CVPR2026W/XAI4CV/papers/Zhang_FAME_Feature_Activation_Map_Explanation_on_Image_Classification_and_Face_CVPRW_2026_paper.pdf

FAME is the most recent of the three. Using it alongside one of the two above would make
the evaluation more convincing.

### What a saliency map should look like

A sensible explanation for a genuine pair should highlight shared facial regions between
the two images, typically the periocular region, the nose bridge and the jawline. For an
imposter pair the explanation should either show weak activation everywhere or highlight
regions that happen to look similar by chance (glasses, beard). If the maps look like
random noise, something is wrong with the integration.

---

## Stage 4: Adversarial attacks

### Warm-up: attacking the FR decision

Before attacking the explanation, build and test attacks that change the FR decision.
This gives you confidence that the gradient pipeline is working correctly.

Reference repo (use with caution, the supervisor flagged bugs):
https://github.com/dev-yoonik/youverse-adversarial-attacks

#### Loss functions for face verification

Standard adversarial attacks use cross-entropy loss against a class label. Face verification
does not have class labels in the same sense. Instead the loss should be defined over the
cosine similarity between the two embeddings.

For a genuine pair (obfuscation attack, goal: make the model think they are different people):

```python
def obfuscation_loss(model, img_a, img_b):
    e_a = model(img_a.unsqueeze(0))
    e_b = model(img_b.unsqueeze(0))
    # Maximise distance, so minimise similarity
    return torch.nn.functional.cosine_similarity(e_a, e_b, dim=1).mean()
```

For an imposter pair (impersonation attack, goal: make the model think they are the same person):

```python
def impersonation_loss(model, img_a, img_b):
    e_a = model(img_a.unsqueeze(0))
    e_b = model(img_b.unsqueeze(0))
    # Minimise distance, so maximise negative similarity
    return -torch.nn.functional.cosine_similarity(e_a, e_b, dim=1).mean()
```

#### FGSM

```python
def fgsm(model, img_a, img_b, loss_fn, epsilon=0.03):
    img_a_adv = img_a.clone().detach().requires_grad_(True)
    loss = loss_fn(model, img_a_adv, img_b)
    loss.backward()
    perturbation = epsilon * img_a_adv.grad.sign()
    img_a_adv = torch.clamp(img_a + perturbation, -1, 1).detach()
    return img_a_adv
```

#### I-FGSM

```python
def ifgsm(model, img_a, img_b, loss_fn, epsilon=0.03, alpha=0.005, steps=20):
    img_a_adv = img_a.clone().detach()
    for _ in range(steps):
        img_a_adv.requires_grad_(True)
        loss = loss_fn(model, img_a_adv, img_b)
        loss.backward()
        grad_sign = img_a_adv.grad.sign()
        img_a_adv = img_a_adv.detach() + alpha * grad_sign
        # Project back into the epsilon ball around the original
        delta = torch.clamp(img_a_adv - img_a, -epsilon, epsilon)
        img_a_adv = torch.clamp(img_a + delta, -1, 1).detach()
    return img_a_adv
```

### Main contribution: attacking the explanation

This is the novel part. The goal is to keep the FR decision intact while changing
the saliency map to point at irrelevant regions.

The loss has two terms:

```python
def explanation_attack_loss(model, xai_method, img_a, img_b,
                             original_map, target_map=None,
                             lambda_pred=1.0, lambda_expl=2.0,
                             genuine_pair=True):

    # Term 1: keep the prediction stable
    e_a = model(img_a.unsqueeze(0))
    e_b = model(img_b.unsqueeze(0))
    sim = torch.nn.functional.cosine_similarity(e_a, e_b, dim=1).mean()

    if genuine_pair:
        # For genuine pairs: punish if similarity drops too much
        pred_loss = torch.relu(0.5 - sim)
    else:
        # For imposter pairs: punish if similarity rises too much
        pred_loss = torch.relu(sim - 0.2)

    # Term 2: push the saliency map away from the original
    attacked_map = xai_method(model, img_a, img_b)
    if target_map is not None:
        # If you have a specific misleading target in mind (e.g. background)
        expl_loss = -torch.nn.functional.mse_loss(attacked_map, target_map)
    else:
        # Otherwise just maximise distance from the original map
        expl_loss = -torch.nn.functional.mse_loss(attacked_map, original_map.detach())

    return lambda_pred * pred_loss + lambda_expl * expl_loss
```

Then wrap this inside I-FGSM style iterations, optimising over `img_a` while keeping the
perturbation within an epsilon ball.

A reasonable target map for evaluation purposes is either a uniform map (flat, no
highlighted region) or a map that concentrates activation on the background pixels outside
the face segmentation mask.

---

## Evaluation

For each pair you should record:

- Cosine similarity before and after the attack (confirm the decision did not flip)
- Saliency map correlation between original and attacked explanation (confirm the map changed)
- SSIM or L2 norm of the pixel perturbation (confirm the perturbation is imperceptible)

A good result looks like: perturbation is small (L-inf below epsilon), FR decision is
preserved (similarity stays on the same side of the threshold), but saliency map correlation
drops significantly.

---

## Key references

- ElasticFace: https://arxiv.org/abs/2109.09416
- xFace: https://openaccess.thecvf.com/content/WACV2023W/XAI4B/html/Knoche_Explainable_Model-Agnostic_Similarity_and_Confidence_in_Face_Verification_WACVW_2023_paper.html
- xSSAB: https://arxiv.org/abs/2304.13409
- FAME: https://openaccess.thecvf.com/content/CVPR2026W/XAI4CV/papers/Zhang_FAME_Feature_Activation_Map_Explanation_on_Image_Classification_and_Face_CVPRW_2026_paper.pdf
- Heo et al. 2019 "Fooling Neural Network Interpretations via Adversarial Model Manipulation" (NeurIPS): closest conceptual predecessor for the explanation attack
- Adebayo et al. 2018 "Sanity Checks for Saliency Maps": important background on saliency fragility
- Adversarial attacks reference repo (check for bugs before using): https://github.com/dev-yoonik/youverse-adversarial-attacks
