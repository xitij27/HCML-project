# Next experiments

Revised after notebook 10. The write-up is coherent and defensible as it stands — nothing
here is required. But two claims in the conclusion are currently **asserted rather than
demonstrated**, and both can be closed cheaply.

## Post-notebook-10 verdict

The conclusion says the attack changes the explanation without making it wrong, and
attributes that to an **objective/metric mismatch** — the attack minimises correlation, so
it relocates saliency onto *other genuinely informative* pixels rather than uninformative
ones. Two gaps in that argument:

| Claim in the conclusion | Evidence we have | Evidence we lack |
|---|---|---|
| Saliency relocated onto other *informative* pixels (redundancy) | per-pair corr(map movement, faithfulness lost) = −0.347; the 200 most-disrupted pairs lose ~3× more | **where the saliency mass actually went** |
| The limit is the *objective*, not the perturbation budget | — | **what happens when the band is loosened.** 49.9% of pairs were shrunk below 3/255, which argues the *budget* is binding |

The second is an internal tension: the conclusion blames the objective while the data shows
the constraint biting on half the pairs. A reviewer will spot it. Experiment A closes it in
about 12 minutes.

**Recommended minimum: A + B + C below, ~30 minutes of GPU total, all at subset scale.**
None of them touches the existing results; each sharpens a claim already being made. Then
decide on the full target-mode run from C's outcome.

---

## A. Loosen the score band — is it the objective or the budget? (~12 min)

**Why first.** It resolves the internal tension above, and it answers the obvious question
*"would the attack have worked if you had allowed more score drift?"* — which cannot
currently be answered at all.

**Design.** ε = 4/255, gradient map only, ~300 accepted genuine pairs, sweep

```
SIM_TOL ∈ {0.02, 0.05, 0.10, 0.20}
```

Report per setting: mean |Δsim|, decision preserved, mean L∞ actually used, fraction of
pairs shrunk, map correlation, deletion AUC.

**How to read it.**

* Deletion AUC **stays flat** as the band loosens → the limit is the *objective*. The
  conclusion's central claim is confirmed, and confirmed strongly, because more freedom did
  not help.
* Deletion AUC **climbs** with the band → the limit is the *budget*, the conclusion needs
  rewording, and notebook 04's r = 0.06 sits on the same curve rather than being an
  incomparable anecdote.

Either outcome is worth having, and it also produces the trade-off figure (score stability
versus explanation disruption) that the report currently lacks.

---

## B. Where did the saliency actually go? (~4 min)

**Why.** The redundancy explanation is the load-bearing part of the conclusion and is
currently inferred, not shown. This measures it directly and needs no new attack machinery.

**Design.** ~300 pairs, reuse the existing attack, compute the gradient map before and
after, and record:

```python
BORDER = build_target_mask(border_frac=0.22)          # off-face frame
border_mass = (m * BORDER).sum() / m.sum()            # before and after
com_shift   = |centre_of_mass(m_after) - centre_of_mass(m_before)|   # pixels
top10_inside_face = fraction of the NEW top-10% pixels inside the face interior
```

**How to read it.**

* Border mass roughly unchanged (e.g. 0.23 → 0.26), centre of mass shifting a few pixels,
  new hot-spots still inside the face → **redundancy confirmed**: the mass moved from one
  informative region to another. This is the sentence the conclusion wants to be able to
  write — "the hot-spot moved from the eyes to the cheek, and both carry identity".
* Border mass rising sharply while faithfulness barely moves → the redundancy story is
  wrong and something more interesting is happening; investigate before writing up.

---

## C. Target-mode attack — attack faithfulness directly (~4 min subset)

**Why.** The one experiment that could change the headline from *"a partial success plus a
methodological finding"* to *"here is how you actually break an explanation, and here is why
the naive objective does not."* It is the controlled counterpart to the current attack:
same constraint, same evaluation, different objective.

**Design.** Replace the correlation term with an off-face mass term. Notebook 04 had this
mode; it was not carried into 09.

```python
def build_target_mask(H=112, W=112, border_frac=0.22):
    b = int(min(H, W) * border_frac)
    m = np.zeros((H, W), dtype=np.float32)
    m[:b, :] = 1; m[-b:, :] = 1; m[:, :b] = 1; m[:, -b:] = 1
    return m

TARGET_MASK = torch.tensor(build_target_mask(), device=DEVICE)   # (H, W)

# inside explanation_attack(), replacing the correlation term:
mass = ((att_map * TARGET_MASK).flatten(1).sum(1)
        / (att_map.flatten(1).sum(1) + 1e-8))          # (B,)
expl_loss = -mass          # minimise -> drive mass INTO the border
```

Everything else — the two-sided band, the hard projection, the evaluation — stays identical,
so the two modes are directly comparable. Also record `target_mass` before and after; it is
the attack's own objective and belongs in the results table.

**Subset first:** `MAX_PAIRS = 300`, `TRANSFER_METHODS = ["gradient"]`.

* Deletion AUC moves **> 30%** toward 0.370 (i.e. reaches ~0.237+) → run the full 2,991
  pairs with all four methods (~105 min). The paired result — *decorrelation preserves
  faithfulness, targeting destroys it* — is considerably stronger than either half alone.
* Moves **< 15%** → report as a second negative result. Combined with A, that would say the
  binding constraint is the perturbation budget under a decision constraint, which is a
  clean and defensible conclusion in its own right.

---

## Deferred — and why the outcome changed my view

**Robustness (JPEG, resizing) and SmoothGrad as a defence.** Both were sensible when the
attack looked strong. They are now **premature**: there is little point measuring whether a
partially-working attack survives compression, or building a defence against an attack that
already fails to make explanations unfaithful. Revisit only if C succeeds.

**Multi-method attack (defeat the "use another family" defence).** Expensive — a
differentiable occlusion surrogate needs its own forward passes inside the attack loop — and
it does not change the headline. The transfer staircase is already clean and honestly
reported. Out of scope for a bonus project; the natural next paper, not the next week.

**Patch-LFW.** Needs the patch dataset generated from the `patches/` scripts in the xSSAB
repository. Worth mentioning in the write-up as the closest published relative of our
evaluation, not worth building now.

**Counteractor attack** (corrupt the dissimilarity explanation of a rejected pair). Well
posed and untouched, but it is a new experiment rather than a strengthening of this one.

**A harder benchmark (IJB-C).** Out of scope — LFW is the dataset provided.

---

## Zero-compute, do it anyway

**Write to Marco** with two things:

1. Ask for the official xSSAB implementation. Ours is a reimplementation from the WACV 2024
   paper (his repo carries no licence, so nothing was vendored); `saliency_xssab` is a single
   self-contained function and his code swaps in directly.
2. Tell him the **actor map matches the raw input-gradient map on faithfulness** (deletion
   0.180, insertion 0.561 — identical) **while the combined actor−counteractor map is
   clearly worse** (0.198 / 0.533). Notebook 07 ranked xSSAB last because it scored the
   combined map; that ranking understates his method and the correction is in his favour.

---

## Settled — do not redo

- **Dataset layout.** 6,000 pairs, alternating 300 genuine / 300 imposter blocks, verified
  at Cohen's *d* = 8.32 with both alternatives rejected, holding in all 10 folds.
- **Threshold.** EER = 0.300% at 0.2136. The old 0.30 corresponds to FMR 0.033% and was
  near accuracy-optimal — the objection was methodological, not empirical.
- **Model correctness.** LFW 10-fold accuracy 99.80% ± 0.25%, matching the published
  ElasticFace-Arc figure.
- **Attack directions.** All 16 configurations correct; 100% success both ways at 4/255.
  The low-budget asymmetry (impersonation easier) is explained: obfuscation displaces the
  score 1.18–1.73× more per unit of perturbation but has 2.41× further to travel.
- **Insertion is a weak metric here.** Only 0.08 of headroom versus 0.19 for deletion,
  because random insertion already recovers most of the similarity. Judge attacks on
  deletion.
- **xSSAB variant.** Use `xssab_actor`, not the combined map.
