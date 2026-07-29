# Response to supervisor feedback

Each point below maps to the notebook that addresses it. **Notebooks 01–05 are unchanged**
— they are the record that was reviewed. All corrections are in the new series **06–10**.

| Notebook | Purpose | GPU time |
|---|---|---|
| `06-protocol-and-threshold` | All 6,000 pairs; layout verified; EER / FMR thresholds; LFW 10-fold | ~5 min |
| `07-xai-faithfulness` | Insertion/Deletion over the whole dataset, 3 methods + random control | ~60–80 min |
| `08-decision-attacks` | FGSM / I-FGSM, one image per pair, both directions, 6,000 pairs | ~80–90 min |
| `09-explanation-attack` | Main contribution; Insertion/Deletion before vs after; transfer | ~2 h |
| `10-results-and-discussion` | Cross-experiment figures and the written discussion | none |

Run them in order. 06–09 are self-contained (each recomputes the embeddings, ~40 s) and
each writes CSV/JSON to `/kaggle/working`. Only `10` consumes those artifacts — attach the
outputs of 06–09 to it via *Add data → Notebook Output*.

---

## 1. "You seem not to use all the data/pairs, it should be 6k pairs in total"

**Correct, and the gap was larger than it looks.** `imgs/` holds 12,000 images
(`0.jpg`…`11999.jpg`) = **6,000 pairs**. Notebooks 01–05 hardcoded `N_PAIRS = 900`, so they
used 15% of the data — and at a 2:1 genuine:imposter ratio (600/300) rather than the
balanced 3,000/3,000 of the real protocol. That imbalance is why the imposter tail was too
thin to estimate a false match rate from.

**Fixed in 06, inherited by 07–09.** The layout is standard LFW view-2:

```
is_genuine(pair) = (pair % 600) < 300        fold(pair) = pair // 600
```

10 folds × 600 pairs, each fold being 300 genuine then 300 imposter. `approach.md`'s
"0–299 genuine, 300–599 imposter, 600–899 genuine" is this same pattern seen through a
900-pair window.

**It is verified, not assumed** (notebook 06, step 3), because if it were wrong every
result in 07–10 would be wrong:

- three competing hypotheses are scored by the separation they produce; the notebook
  asserts the assumed layout wins and stops if it does not;
- the genuine/imposter gap is checked **in each of the 10 folds independently**;
- the LFW 10-fold accuracy is compared against the figure published for ElasticFace-Arc,
  which only comes out right if the weights, the preprocessing *and* the labelling are all
  correct simultaneously.

---

## 2. "The decision threshold seems to be set randomly at 0.3 … use FMR or EER"

**Fixed in 06.** The threshold is now derived on LFW, as suggested:

- **EER** is the headline operating point — one number, no free parameters — and is what
  08 and 09 report against.
- Thresholds at **FMR = 10⁻¹, 10⁻², 10⁻³** are computed as well, and every attack result in
  08 and 09 carries a sensitivity table across all of them, so no conclusion rests on a
  single threshold choice.
- The notebook also reports **what 0.30 actually was** in FMR terms, and runs the standard
  **LFW 10-fold protocol** (threshold fitted on 9 folds, applied to the held-out one).

One caveat stated in the notebook: with 3,000 imposter pairs, FMR = 10⁻³ is resolved by
exactly **3 pairs**, so that row is indicative. FMR = 10⁻² (30 pairs) is the tightest
operating point this dataset supports comfortably.

*(Incidentally, the reference xSSAB implementation also selects its threshold by EER, which
was a useful confirmation that this is the expected convention here.)*

---

## 3. "A proper evaluation of the explainability method is missing … Insertion and Deletion curves for the whole dataset … before and after the attack"

**Split across 07 (before) and 09 (after), both over the full dataset.**

Both curves perturb the **probe** and track its cosine similarity to the untouched
**reference**:

| | Procedure | A faithful map… | Want |
|---|---|---|---|
| Deletion | replace the most salient pixels with mid-grey, progressively | destroys the match immediately | **low AUC** |
| Insertion | restore the most salient pixels over a blurred probe | rebuilds the match immediately | **high AUC** |

**Notebook 07** measures four maps over all 6,000 pairs: squared input-gradient, xSSAB,
xFace occlusion, and a **uniform-random control**. The random control is the important
one — it fixes the *scale*. Without it, a post-attack AUC is a number with no reference
point; with it, the question becomes "how far did the attacked explanation move toward
having no information at all?" The notebook asserts that every real method beats the random
floor, and fails if one does not, since that would mean the metric itself is misconfigured.

**Notebook 09** recomputes exactly the same quantities on the attacked images, in the same
session and on the same pairs, so both halves of the comparison are measured on identical
footing. The specific claim it tests: after the attack, deletion AUC should **rise** and
insertion AUC should **fall**, moving toward the random floor, while the decision does not
move at all.

This also replaces the previous evidence, which was Pearson correlation between maps.
Correlation says a map *changed*; it cannot distinguish "changed" from "became wrong".
Insertion/deletion can. (Correlation and top-10% hot-spot IoU are still reported alongside.)

**On xSSAB.** Notebook 05 used a `grad × input` stand-in. Notebooks 07 and 09 implement the
published algorithm: for L2-normalised embeddings, `cos = Σ_d e_probe[d]·e_ref[d]`, and the
per-dimension arguments are split into those that *support* the match (≥ `thr/D`, actor) and
those that *oppose* it (counteractor); each partial score is back-propagated separately, to
the probe image only, and reduced as `mean(|grad|)` over colour channels, with the combined
map being actor − counteractor. This is **our implementation**, written from the WACV 2024
paper and the reference implementation's documented behaviour — the repository carries no
licence file, so no code from it is redistributed. If you would prefer the official
implementation used instead, `saliency_xssab` is a single self-contained function and can be
swapped directly.

*Note:* Patch-LFW from your paper is the closest published relative of this evaluation. We
used insertion/deletion as you asked; adding Patch-LFW would need the patch dataset from the
xSSAB repository and is listed as future work.

---

## 4. "You seem to attack the same image … attack only one image of the pair … genuine similarity should decrease, imposter should increase"

Two separate problems were behind that, and only one had been fixed.

**(a) The sign — was fixed in notebook 04.** Notebook 03 stepped
`img + alpha·grad.sign()` (gradient *ascent*) while its loss functions already carried the
attack sign, so every attack drove similarity confidently the wrong way. That is why all
four warm-ups reported `flip_rate = 0.00` despite a large mean `|Δsim|`.

**(b) The imposter direction — was never fixed.** Notebook 04 corrected the sign but
**dropped imposter pairs entirely and ran genuine only**. So the direction you flagged had
still never been demonstrated. **Notebook 08 closes this**, running both directions over all
6,000 pairs:

| Pair type | Attack | Goal | Correct sign of Δsim |
|---|---|---|---|
| genuine | obfuscation | make them look like different people | **negative** |
| imposter | impersonation | make them look like the same person | **positive** |

The notebook **asserts** both signs and fails if either is wrong, so this cannot silently
regress.

**(c) One image per pair.** Every pair is split into a **probe** — the single image an
attack may modify — and a **reference**, which is never touched. `probe_reference()` asserts
the two are different files; `PROBE_SIDE="random"` picks the side by a seeded per-pair coin
flip so the attacked image is not systematically image 0 of every pair, while staying
reproducible; and every figure and CSV row names both source files. Notebook 08 prints the
assignment explicitly and verifies that probe + reference together cover all 12,000 images
exactly once.

**(d) On notebook 04's genuine similarity rising 0.73 → 0.82.** That was not a bug — its
constraint was one-sided (`relu(0.45 − sim)`), so the attack was free to push the score
*up*. But it reads as one. Notebook 09 uses a **two-sided** constraint,
`relu(|sim − sim₀| − tol)`, plus a hard per-sample projection, so the reported Δsim is
centred on zero. The score barely moves in either direction — a strictly stronger claim than
"the decision was preserved".

---

## 5. "Now I arrived at Stage 4 – please indicate if things are outdated/wrong"

| Notebook | Status | Superseded by |
|---|---|---|
| `01-data-and-model` | outdated — 900 pairs, threshold 0.30 | **06** |
| `02-xai` | outdated — qualitative only, 8 pairs | **07** |
| `03-attacks` | **wrong** — gradient-ascent sign bug; `flip_rate = 0.00` everywhere | **08**, **09** |
| `04-explanation-attack-v2` | partly superseded — sign fixed, but 40 pairs, genuine only, one-sided constraint | **09** |
| `05-xssab-and-transfer` | partly superseded — 20 pairs, `grad × input` stand-in for xSSAB | **09** |
| `06`–`10` | **current** | |

This table is also at the top of notebook 10 and in the README.

---

## 6. "A proper discussion of the results is missing"

Every new notebook ends with a **Results and discussion** section covering what was found,
how to read it, and what it does not show. **Notebook 10** is the synthesis: it loads the
artifacts from 06–09 and produces the figures that span more than one experiment — attack
success versus operating threshold, explanation faithfulness before/after against the random
floor, and transfer across explanation families — followed by the overall discussion,
limitations, and future work.

---

## Smaller corrections made along the way

- **Perturbation budget units.** Notebooks 03–05 set `EPSILON = 0.03` and described it as
  "≈ ±8/255 per pixel". Images are normalised by `x/127.5 − 1`, so 0.03 is **3.8** grey
  levels, not 8. The new notebooks specify the budget in grey levels out of 255 and convert
  explicitly.
- **Objective/metric mismatch.** Notebook 03 maximised MSE between saliency maps but
  *reported* Pearson correlation. Doubling every value of a map gives a large MSE and leaves
  correlation at 1.0 — the same pixels still light up. This is why its correlation barely
  moved. Notebook 09 optimises correlation directly.
- **Scale.** Everything is batched (BatchNorm runs in `eval`, so batch elements are
  independent and `autograd.grad(..., create_graph=True)` gives correct per-sample
  gradients). Long loops checkpoint to CSV so a Kaggle timeout does not lose the run, and the
  explanation attack halves its batch automatically on CUDA OOM.

## Known limitations

- White-box only; no black-box or transfer-to-another-model setting.
- Digital pipeline only — no JPEG re-compression, resizing, or print-scan robustness.
- One dataset, one model. LFW is close to saturated for ElasticFace-Arc, so error rates come
  from few tail events.
- xSSAB is our implementation, not the official code (see point 3).
- Insertion/deletion has a masking confound: part of any score change comes from masked
  inputs being off-distribution. The random control bounds it — it suffers the same artefact
  — but does not eliminate it.
- The explanation attack targets accepted genuine pairs. Attacking the *dissimilarity*
  explanation of a rejected pair (the counteractor side of xSSAB) is untouched.
