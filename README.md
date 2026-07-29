# Adversarial Manipulation of Explanations in Face Verification

An HCML bonus project asking whether an adversary can craft a small, imperceptible
perturbation to one image of a verification pair that **keeps the face-recognition decision
intact while changing the saliency map**. This is an attack on the *explanation*, not on the
classifier — a different problem from standard adversarial attacks in biometrics, which try
to flip the decision itself.

**Headline result.** The decision and the similarity score can be held essentially fixed
(Δsim = +0.011, decision preserved 100%) while 75% of the saliency map's hot-spots relocate
and map correlation falls to 0.236. But the relocated map retains **~89% of its
faithfulness** — it changed without becoming wrong. Two mechanisms set that ceiling:
redundancy inside the face, and the model's learned invariance to background.

## Read these first

| Document | What it is |
|---|---|
| **[REPORT_DRAFT.md](REPORT_DRAFT.md)** | The research report — motivation, method, results, discussion |
| **[RESULTS.md](RESULTS.md)** | Every final number, in tables |
| [FEEDBACK_RESPONSE.md](FEEDBACK_RESPONSE.md) | Each point of the supervisor review → what changed → where |
| [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md) | Follow-ups worth running, prioritised |
| [approach.md](approach.md) | The original project brief (predates the corrections) |

## Layout

```
.
├── REPORT_DRAFT.md  RESULTS.md  FEEDBACK_RESPONSE.md  NEXT_EXPERIMENTS.md  approach.md
├── elastic_resnet.py                  # iResNet-100 backbone (inlined into each notebook)
├── references/ElasticFace_paper.pdf
└── experiments/
    ├── 06-protocol-and-threshold/     # notebook + outputs/ + figures/
    ├── 07-xai-faithfulness/
    ├── 08-decision-attacks/
    ├── 09-explanation-attack/
    ├── 10-results-and-discussion/
    ├── 11-attack-diagnostics/
    └── legacy/                        # notebooks 01-05, superseded
```

Each experiment folder holds its notebook, an `outputs/` directory (CSV, JSON, NPZ) and a
`figures/` directory (PNG). Output files keep their `NN_` prefix so they stay unambiguous if
they are ever collected into one folder — which is what notebook 10 expects.

## Experiments

| # | Experiment | What it establishes | GPU |
|---|---|---|---|
| 06 | [protocol-and-threshold](experiments/06-protocol-and-threshold/06-protocol-and-threshold.ipynb) | All 6,000 pairs; layout verified at *d* = 8.32; EER = 0.300% at τ = 0.2136; LFW 10-fold 99.80% ± 0.25% | ~5 min |
| 07 | [xai-faithfulness](experiments/07-xai-faithfulness/07-xai-faithfulness.ipynb) | Insertion/deletion over all 6,000 pairs, 3 methods + random floor | ~47 min |
| 08 | [decision-attacks](experiments/08-decision-attacks/08-decision-attacks.ipynb) | FGSM/I-FGSM both directions, 100% success at 4/255, all signs correct | ~50 min |
| 09 | [explanation-attack](experiments/09-explanation-attack/09-explanation-attack.ipynb) | **Main contribution** — decision fixed, map moved, faithfulness measured before vs after | ~105 min |
| 10 | [results-and-discussion](experiments/10-results-and-discussion/10-results-and-discussion.ipynb) | Cross-experiment figures and tables | none |
| 11 | [attack-diagnostics](experiments/11-attack-diagnostics/11-attack-diagnostics.ipynb) | Why the attack has a ceiling: score-band sweep, saliency relocation, direct faithfulness objective | ~28 min |

Legacy notebooks `01`–`05` are kept unchanged as the reviewed record. `03` contains a
gradient-ascent sign bug (`flip_rate = 0.00` everywhere); `04` fixed the sign but ran genuine
pairs only; `05` used a `grad × input` stand-in for xSSAB. All are superseded by `06`–`11`.

## Running them

Each of `06`–`09` and `11` is self-contained — it inlines the backbone and recomputes the
12,000 embeddings in about 40 s — so they can be run in any order. Only `10` consumes
upstream artifacts: attach the `outputs/` of `06`–`09` as a Kaggle Dataset, or via
*Add Input → Notebooks* if those notebooks were committed. Its loader searches
`/kaggle/input` recursively and matches both `thresholds.json` and `06_thresholds.json`.

Every notebook has a `MAX_PAIRS` (or `DIAG_PAIRS`) knob — set it to a few hundred for a
smoke test before committing to a full run. All of them use every visible GPU; on Kaggle's
*GPU T4 x2* that roughly halves wall time. Set `N_GPUS = 1` to force single-device.

## Dataset

**12,000 images = 6,000 pairs**, not the 900 assumed in `approach.md` and notebooks `01`–`05`.
Images are named sequentially (`0.jpg`, `1.jpg`, …), 112×112, pre-aligned; consecutive images
form a pair. The layout is standard **LFW view-2** — 10 folds × 600 pairs, each fold being 300
genuine followed by 300 imposter:

```python
is_genuine(pair) = (pair % 600) < 300     # 3,000 genuine, 3,000 imposter
fold(pair)       = pair // 600            # 10 folds
```

Notebook `06` verifies this empirically before anything is built on it.

**Probe and reference.** Every pair is split into a *probe* — the single image an attack may
modify — and a *reference*, never touched. The split is asserted pairwise, and with
`PROBE_SIDE="random"` the probe is chosen by a seeded per-pair coin flip, so the attacked
image is not systematically image 0 while the run stays reproducible.

## Data & model weights

Large files are not tracked (see [.gitignore](.gitignore)). The weights
(`ElasticFaceArc.pth`) and the LFW image set (`imgs/`) come from:

```
https://owncloud.fraunhofer.de/index.php/s/OoLxFvESM9aUkR9
```

On Kaggle, upload them as a dataset and point `IMAGES_DIR` / `WEIGHTS_PATH` at the attached
path.

## Dependencies

```bash
pip install torch torchvision numpy opencv-python matplotlib Pillow pandas scipy
```

All preinstalled on Kaggle. A CUDA GPU is assumed.

## Key references

- **ElasticFace** — Boutros et al., CVPRW 2022 · https://arxiv.org/abs/2109.09416 · [code](https://github.com/fdbtrs/ElasticFace)
- **xSSAB** — Huber et al., WACV 2024 · https://arxiv.org/abs/2304.13409 · [code](https://github.com/marcohuber/xSSAB)
- **xFace** — Knoche et al., [WACVW 2023](https://openaccess.thecvf.com/content/WACV2023W/XAI4B/html/Knoche_Explainable_Model-Agnostic_Similarity_and_Confidence_in_Face_Verification_WACVW_2023_paper.html) · [code](https://github.com/Martlgap/x-face-verification)
- **RISE / insertion–deletion** — Petsiuk et al., BMVC 2018 — the faithfulness metric used in `07`, `09`, `11`
- Dombrowski et al. 2019, *Explanations can be manipulated and geometry is to blame* (NeurIPS)
- Heo et al. 2019, *Fooling Neural Network Interpretations via Adversarial Model Manipulation* (NeurIPS)
- Adebayo et al. 2018, *Sanity Checks for Saliency Maps* (NeurIPS)
