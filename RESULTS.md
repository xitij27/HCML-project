# Results — all final numbers

ElasticFace-Arc (iResNet-100, 512-d) on LFW, 6,000 pairs. Every number below is from an
executed notebook; the source is named in each section.

---

## 1. Protocol and threshold (notebook 06)

**Dataset.** 12,000 images → 6,000 pairs → 3,000 genuine / 3,000 imposter, 10 folds of 600.
Layout `is_genuine(p) = (p % 600) < 300`, verified:

| hypothesis | mean gap | Cohen's *d* |
|---|---|---|
| **A: alternating 300-blocks (LFW view-2)** | **+0.7165** | **8.32** |
| B: first 3,000 genuine | −0.0024 | −0.01 |
| C: even pairs genuine | +0.0052 | 0.01 |

Holds in all 10 folds independently: per-fold gaps 0.7039–0.7279, genuine/imposter overlap
≤ 0.43%.

**Scores.**

| | n | mean | std | min | max |
|---|---|---|---|---|---|
| genuine | 3,000 | +0.7196 | 0.0993 | +0.0006 | +0.9683 |
| imposter | 3,000 | +0.0032 | 0.0706 | −0.2475 | +0.3024 |

**Operating points.**

| point | threshold | FMR | FNMR | accuracy |
|---|---|---|---|---|
| **EER** | **0.21356** | 0.300% | 0.300% | 99.700% |
| FMR ≤ 10⁻¹ | 0.09426 | 10.000% | 0.167% | 94.917% |
| FMR ≤ 10⁻² | 0.16826 | 1.000% | 0.267% | 99.367% |
| FMR ≤ 10⁻³ | 0.24829 | 0.100% | 0.300% | 99.800% |
| legacy (0.30) | 0.30000 | 0.033% | 0.333% | 99.817% |

- **EER = 0.300% at threshold 0.2136**
- **LFW 10-fold accuracy = 99.800% ± 0.246%** (matches the published ElasticFace-Arc figure), mean fold threshold 0.2612
- 2,991 of 3,000 genuine pairs accepted at EER. 9 genuine rejected (scores 0.001–0.174), 9 imposters accepted
- Accepted-genuine count is 2,990–2,995 across *every* operating point, so nothing downstream depends on the threshold choice
- FMR = 10⁻³ is resolved by exactly 3 imposter pairs — read as indicative

---

## 2. Explanation faithfulness, before any attack (notebook 07)

All 6,000 pairs, 4 maps, insertion/deletion with 21 points. Runtime 46.8 min on 2×T4
(778 forward passes/s).

**Genuine pairs (n = 3,000)** — the meaningful subset:

| method | deletion AUC ↓ | insertion AUC ↑ | gap (ins − del) ↑ |
|---|---|---|---|
| gradient | 0.1793 | 0.5597 | **0.3803** |
| xface | 0.1772 | 0.5287 | 0.3516 |
| xssab (combined) | 0.1971 | 0.5319 | 0.3349 |
| **random (floor)** | **0.3698** | **0.4806** | **0.1108** |

**Genuine and accepted at EER (n = 2,991)** — the population attacked in 09:

| method | deletion AUC | insertion AUC | gap |
|---|---|---|---|
| gradient | 0.1799 | 0.5612 | 0.3813 |
| xface | 0.1777 | 0.5301 | 0.3524 |
| xssab | 0.1976 | 0.5334 | 0.3358 |
| random | 0.3708 | 0.4819 | 0.1111 |

**Effect sizes** (paired, n = 3,000): gradient vs **random d = 3.44**; gradient vs xssab
d = 1.14; gradient vs xface d = 0.31; xface vs xssab d = 0.16. The spread between methods
is far smaller than the distance from any of them to random.

**Implementation validated on endpoints** — all four methods share identical curve
endpoints, since those depend on the start/end state and not the ranking: deletion
0.7196 → 0.0019 (grey), insertion 0.0272 (blurred) → 0.7196.

**Imposter pairs are degenerate** (base similarity 0.003): gradient 0.0001/0.0036,
xssab 0.0012/0.0056, random 0.0030/0.0045. xFace deletion AUC is **−0.0760** — its curve
dips to −0.1231 after 20% of pixels are deleted, because it correctly finds the few regions
supplying positive evidence and removing them drives the score below zero. Not a bug; the
gap metric is simply not comparable across genuine and imposter subsets.

**Do not quote the "all pairs" row** — it averages a meaningful and a degenerate population.

---

## 3. Decision attacks (notebook 08)

All 6,000 pairs × 8 configurations = 48,000 rows. Runtime 50.3 min. **All 16 sign checks
correct.** Budget in grey levels /255 (images are `x/127.5 − 1`, so 0.03 in normalised
space = 3.8/255, not 8/255 as notebooks 03–05 claimed).

| attack | ε | direction | mean Δsim | success @ EER |
|---|---|---|---|---|
| FGSM | 1/255 | genuine (obfuscation) | −0.2137 | 2.07% |
| FGSM | 1/255 | imposter (impersonation) | +0.1898 | 37.73% |
| FGSM | 2/255 | genuine | −0.3702 | 15.08% |
| FGSM | 2/255 | imposter | +0.2992 | 87.36% |
| FGSM | 4/255 | genuine | −0.5410 | 59.68% |
| FGSM | 4/255 | imposter | +0.3954 | 99.46% |
| FGSM | 8/255 | genuine | −0.6655 | 88.83% |
| FGSM | 8/255 | imposter | +0.4384 | 99.90% |
| I-FGSM | 1/255 | genuine | −0.2800 | 6.65% |
| I-FGSM | 1/255 | imposter | +0.2371 | 62.44% |
| I-FGSM | 2/255 | genuine | −0.6227 | 72.12% |
| I-FGSM | 2/255 | imposter | +0.4497 | 99.73% |
| **I-FGSM** | **4/255** | **genuine** | **−1.1155** | **100.00%** |
| **I-FGSM** | **4/255** | **imposter** | **+0.6909** | **100.00%** |
| I-FGSM | 8/255 | genuine | −1.4058 | 100.00% |
| I-FGSM | 8/255 | imposter | +0.8123 | 100.00% |

**Threshold sensitivity** (I-FGSM @ 4/255): obfuscation 100% / 99.77% / 99.97% / 100% and
impersonation 100% everywhere, at EER / FMR≤10⁻¹ / 10⁻² / 10⁻³. Zero failures at 4/255
(0 of 2,991 and 0 of 2,990). Final mean similarity: genuine −0.3959, imposter +0.6941.

**The asymmetry.** At low budget impersonation is *easier* (62.4% vs 6.7% at 1/255), from
two competing effects:
- obfuscation displaces the score **1.18× / 1.38× / 1.61× / 1.73×** more per unit of ε
- but it has **2.41× further to travel** (0.506 vs 0.210 to reach τ = 0.2136)

**Probe/reference.** Probe is image 0 in 49.8% of pairs, image 1 in 50.2%. Probe ∪ reference
= all 12,000 images exactly once.

---

## 4. Explanation attack (notebook 09)

2,991 accepted genuine pairs, ε = 4/255, 40 steps, two-sided band `|Δsim| ≤ 0.05`.
Runtime 105.3 min.

**The decision is untouched:**

- mean Δsim **+0.0110**, mean |Δsim| **0.0261**, 95% of pairs below 0.0474
- **decision preserved 100.00% at all four operating points**
- mean L∞ **2.67/255** (budget 4), mean L2 2.551
- 44.8% of pairs reach full budget; **49.9% shrunk below 3/255** by the band
- hard projection: 0.31 halvings per sample-step, cap hit **0.00%**

**The explanation changes but stays faithful:**

| method | Pearson r | top-10% IoU | deletion AUC | insertion AUC | gap lost | → random floor |
|---|---|---|---|---|---|---|
| gradient (attacked) | **0.2359** | 0.2528 | 0.1799 → 0.2007 | 0.5612 → 0.5675 | 5.4% | **10.9%** |
| xssab | 0.3407 | 0.2252 | 0.1976 → 0.2232 | 0.5334 → 0.5392 | 8.8% | 14.9% |
| xssab_actor | 0.4584 | 0.2513 | 0.1801 → 0.2005 | 0.5613 → 0.5685 | 4.9% | 10.8% |
| xface (gradient-free) | **0.7198** | 0.4434 | 0.1777 → 0.1952 | 0.5301 → 0.5408 | 2.8% | 9.1% |

- gradient r: median **0.128**; 38.8% of pairs below 0.1; 75.7% below 0.3
- top-10% IoU 0.253 → **75% of the originally-hottest pixels relocate**
- deletion AUC rise is highly significant (p ≈ 1e-221) but small: **~89% of faithfulness retained**
- per-pair corr(map movement, faithfulness lost) = **−0.347** (p ≈ 1e-85)
- the 200 most-disrupted pairs (r ≤ 0.034): deletion 0.2253 → 0.2819, ~3× the average shift

**Transfer** is a monotone staircase — gradient 0.236 → xssab 0.341 → xssab_actor 0.458 →
xface 0.720. The attack is confined to *gradient-based* explanations.

**xSSAB variant.** `xssab_actor` scores deletion 0.1801 / insertion 0.5613 — **identical to
the raw gradient map** and clearly better than the combined actor−counteractor map
(0.1976 / 0.5334). Notebook 07 ranked xSSAB last only because it scored the combined map.
**Quote `xssab_actor`.**

---

## 5. Diagnostics (notebook 11, 300 accepted genuine pairs)

Subset baselines: unattacked deletion AUC **0.1915**, random floor **0.3698**.

### Part A — loosening the score band (mask-independent, valid as run)

| SIM_TOL | mean \|Δsim\| | decision kept | mean L∞ | shrunk | Pearson r | deletion AUC | → floor |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.0118 | 100% | 1.22 | 88% | 0.479 | 0.1983 | 3.9% |
| 0.05 | 0.0260 | 100% | 2.81 | 44% | 0.220 | 0.2135 | 12.4% |
| 0.10 | 0.0462 | 100% | 3.64 | 14% | 0.109 | 0.2254 | 19.1% |
| 0.20 | 0.0606 | 100% | 3.95 | **2%** | **0.065** | 0.2298 | **21.5%** |

Marginal gains **+8.5, +6.7, +2.5 points** — saturating. At the widest band the constraint
is essentially off (2% shrunk, L∞ 3.95/4.00) and correlation is floored at 0.065, yet the
attack reaches only 21.5% of the way. **Both effects are real; the objective sets the
ceiling.**

Notebook 04 reported r = 0.06 with a ~+0.09 score drift — that is the SIM_TOL ≈ 0.20 point
on this curve (r = 0.065, |Δsim| = 0.061), reaching 21.5%. It is no longer an incomparable
anecdote.

### Part B — where the saliency went (measured under the 67.3% frame mask)

- saliency mass in region: 0.2390 → 0.2745
- top-10% hot-spots inside the central 64×64: 76.9% → 72.0%
- centre-of-mass shift: mean 10.08 px, **median 7.84 px**, p95 25.64 px (on a 112 px image)

**Redundancy confirmed** — the mass stayed on the face and moved within it.

### Part C — attacking faithfulness directly

Both masks, same 300 pairs, same band. Divergence baseline: 0.1915 → 0.2135 = **12.4%**.

| off-face mask | area | mass before → after | concentration before → after | deletion AUC | → floor | vs divergence |
|---|---|---|---|---|---|---|
| frame | 67.3% | 0.239 → 0.614 | 0.36 → 0.91 | 0.1915 → 0.2270 | **19.9%** | 1.61×, p = 5.6e-09 |
| **ellipse** | **37.8%** | **0.053 → 0.215** | **0.14 → 0.57** | 0.1915 → 0.2188 | **15.4%** | 1.24×, p = 1.4e-02 |

Concentration ratio = mass share ÷ area share; 1.0 = no spatial preference.

**The tighter, more defensible ellipse mask made the attack harder, not easier.** The
unattacked explanation puts only 5.3% of its mass outside the ellipse, in 37.8% of the
image (ratio 0.14) — it strongly avoids background. The attack quadrupled that to 0.215 but
never reached the 0.378 no-preference line.

---

## 6. The two mechanisms

1. **Inside the face — redundancy.** Eyes, nose bridge, mouth and jawline all carry
   identity, so the attack relocates the ranking among *informative* pixels. Correlation
   collapses (0.236) while faithfulness barely moves (89% retained).
2. **Outside the face — invariance.** ArcFace-trained networks ignore background, so the
   input-gradient there is intrinsically near zero. A ±4/255 perturbation cannot manufacture
   sensitivity where the model has none.

**The saliency map's support region is a property of the model. An imperceptible
perturbation can shuffle mass within that support, but cannot move the support itself.**

---

## 7. What can and cannot be claimed

**Can be claimed.** An imperceptible perturbation (mean L∞ 2.67/255) to one image of a pair
leaves the decision preserved at 100% and the score essentially unchanged (+0.011), while
relocating 75% of the saliency map's hot-spots; the disruption is confined to gradient-based
methods (xFace retains r = 0.72); and a faithfulness-based evaluation shows the relocated
map remains ~89% as faithful — which a correlation-only evaluation would have missed.

**Cannot be claimed.** That the explanation was "destroyed", or that a reviewer is left with
a demonstrably false justification. The evidence supports *different*, not *wrong*.

**The methodological point.** With correlation alone, this would have been written up as a
success on r = 0.236 — exactly as notebook 04 was on r = 0.06. The insertion/deletion
evaluation is what separated "changed" from "became wrong". The supervisor's feedback
changed the conclusion.

---

## 8. Limitations

- White-box, full gradient access; no black-box or cross-model transfer tested
- Digital only — no JPEG re-compression, resizing or print-scan robustness
- One model, one dataset; LFW is near-saturated (whole error budget is 9 pairs per side)
- xSSAB is our implementation from the WACV 2024 paper, not the official code (no licence in the repo)
- Insertion/deletion carries a masking confound, bounded but not removed by the random control
- Insertion is a weak discriminator here (0.08 headroom vs 0.19 for deletion) — judge on deletion
- Only accepted genuine pairs were attacked; the counteractor/rejected-pair case is untouched

---

## 9. Known issues in the executed notebooks

- **`11-attack-diagnostics-c-fix.ipynb`**: only the mask cell and Part C were re-run. Parts
  A, B and D show stale output from the earlier frame-mask run, and the Part C comparison
  table mixes masks on the `border mass before` row (divergence 0.239 = frame, target 0.053
  = ellipse). The deletion-AUC rows are valid — they do not depend on the mask. All correct
  values are tabulated above.
- Notebook 09's summary table quotes the 8/255 row for decision attacks (it selects the
  largest ε) while notebook 08's headline budget is 4/255. Both are 100% success.
