# QAM-Seg quality-aware SSL

Status: **MQE and QGC are implemented in code and covered by synthetic unit
tests. They are not experimentally validated.** No ISIC, HAM10000, or PH2
run is part of this change, and no Dice or HD95 number here is a result.
This does not change the supervised loss used by `python train.py`.

QAM-Seg (Quality-Aware Mamba Segmentation) keeps the released MambaLiteUNet
code as a supervised backbone and adds a quality-aware semi-supervised
objective on top. AMF, LGFM, and CGA stay upstream modules. See
`THIRD_PARTY.md`. CVPR metrics for MambaLiteUNet are not QAM-Seg results.

The supervised entry point remains:

```bash
python train.py
```

## CLI

Omitting every flag leaves the supervised path as it is. `--ssl` is wired.
The three ablation flags are still refused, before the device is touched
and before any epoch runs.

| Flag | Meaning | Wired? |
| --- | --- | --- |
| `--ssl` | MQE + QGC. Default process is a short CPU smoke on synthetic blobs | Yes (code, not a benchmark) |
| `--no-amf` | Ablation: bypass Adaptive Multi-branch Mamba Feature Fusion | No, exit 2 |
| `--no-lgfm` | Ablation: bypass Local-Global Feature Mixing | No, exit 2 |
| `--no-cga` | Ablation: bypass Cross-Gated Attention | No, exit 2 |

```bash
python train.py
python train.py --ssl
python -m qamseg.train_ssl
```

`python train.py --ssl` does not build MambaLiteUNet, does not read
`isic_loader`, and does not download data. It runs `qamseg.train_ssl` with
a tiny conv student so the loss, the gate, and the log line can execute
without a GPU. `--ssl-steps` defaults to 2 for that reason. Passing
`--no-amf`, `--no-lgfm`, or `--no-cga` with or without `--ssl` still exits
2. Those bypasses are not wired; turning them on would change the
supervised baseline without a matching forward.

`--help` exits before training. Unknown arguments are ignored so a typo
cannot silently select a different backbone.

Unit tests, from the repository root, with no GPU and no medical data:

```bash
python -m unittest discover -s tests -v
```

## Labeled path (already in the repo)

Unchanged:

- Data: `loader.py` / `dataprepare/`
- Model: `models/MambaLiteUNet.py`
- Loss: `BceDiceLoss` in `utils.py`
- Loop: `engine.train_one_epoch`, `val_one_epoch`, `test_one_epoch`

Validation and test already log pixel-pooled accuracy, sensitivity,
specificity, Dice (`f1_or_dsc`), and mIoU. HD95 is an extra log field in
pixels at network resolution. It does not replace those metrics. See the
README ERRATA before comparing any of this with the CVPR paper.

## MQE — multi-cue quality estimator (implemented, not validated)

Code: `qamseg/mqe.py`. Tests: `tests/test_mqe.py`.

MQE scores a pseudo-mask. It is not a second segmentation architecture and
it is not a claim about AMF.

Signals, computed from probability maps:

1. **Predictive entropy** of the sigmoid mask (Bernoulli entropy / `ln 2`).
2. **Teacher–student disagreement** on one view (`|p_teacher - p_student|`),
   and **view agreement**: hard Dice between the weak-view pseudo-mask and
   the strong-view pseudo-mask. Low agreement means low quality.
3. **Boundary uncertainty**: mean entropy on a band around the 0.5 contour,
   so a confident interior cannot hide a noisy edge. An empty band falls
   back to the image mean and does not divide by zero.

The estimator returns:

- an image-level score `q` in `[0, 1]`
- a pixel-level weight map in `[0, 1]`

Default fusion is a weighted average of the active cues. `fusion="sigmoid"`
applies the design form `σ(a · cues)`. Cue weights are placeholders.
`QualityWeights.learned` defaults to 0.

`LearnedQualityHead` is a 1×1 stub. `--use-learned-quality-head` does not
train it and does not turn on `L_Q` (`beta = 0`). Reliability targets
`r = 1[correct] · (1 - H)` are available as a pure function for a later
labeled-only fit. They are not computed from pseudo-labels.

Optional BAQ (`--ssl-baq`, off by default) down-weights pixels where the
image gradient is high and quality is low. It is not a new backbone block.

## QGC — quality-gated consistency (implemented, not validated)

Code: `qamseg/qgc.py`, `qamseg/train_ssl.py`. Tests: `tests/test_qgc.py`,
`tests/test_ssl_smoke.py`.

For each unlabeled image:

1. The EMA teacher sees a weak view and produces a pseudo-mask (stop-grad).
2. MQE returns `q` and a pixel map. Disagreement uses the student on that
   same weak view (also stop-grad). Dice uses the strong view.
3. **Gate.** `hard` keeps a pixel or image only when its score is at least
   `tau`. `soft` uses `score ** gamma`. `gated` (the default, and the step
   in the plan) drops an image when `q < tau` and otherwise weights the
   loss by `q ** gamma` and the pixel map.
4. The student on the strong view is trained with weighted BCE + Dice
   against the stopped teacher pseudo-label. The labeled term is
   `BceDiceLoss` at full weight.
5. `lambda_u(t) = lambda_max * exp(-5 (1 - t/T)^2)` and a linear `tau`
   ramp are helpers. The numbers are placeholders, not a tuned schedule.

Empty gates are safe: if every weight is zero, the unsupervised term is
zero, stays on the student graph, and does not divide by the weight sum.

This is consistency under a quality gate. It is not a reimplementation of
AMF, LGFM, or CGA, and it does not retune those blocks.

The smoke augmentations are a horizontal flip (weak) and a vertical flip
plus a corner cutout (strong). They are not the full weak/strong recipe.
The EMA decay defaults to 0.99 and is a placeholder.

## Data

`qamseg/data.py` builds a labeled loader and an unlabeled loader.
`loader.isic_loader` is unchanged and is not imported by that module.
The default datasets are synthetic blobs. A local `.npy` may be passed
with `--ssl-unlabeled-npy` or `--ssl-labeled-npy` plus `--ssl-mask-npy`.
A missing file raises `FileNotFoundError`. Nothing in this path downloads
ISIC, HAM10000, or PH2.

Real unlabeled pools — the remainder of a train split, masks withheld,
test images never used as unlabeled — are still a TODO in
`qamseg/data.py`.

## Logs

Each SSL step prints supervised loss, unlabeled loss (`unsup`), `lambda_u`,
`gate_keep_fraction`, and mean `q`. After the steps, a labeled eval prints
loss, mIoU, Dice (`f1_or_dsc`), accuracy, specificity, sensitivity, and
`hd95`. Those names match `engine.py`. HD95 still uses `utils.summarize_hd95`.
On the smoke path the numbers describe synthetic blobs. They are not
dataset scores and they do not replace the supervised log.

## What is still open

- Train MQE/QGC on a real labeled fraction with MambaLiteUNet as the
  student and teacher. This entry point does not construct that backbone.
- Prepare unlabeled splits. Do not use test images as unlabeled data.
- Wire `--no-amf`, `--no-lgfm`, and `--no-cga` only together with a defined
  forward and a parameter count from `scripts/count_params_flops.py`.
- Region-level gating, a trained reliability head, and the full augmentation
  list.
- No table that presents upstream CVPR numbers as QAM-Seg scores.

## Explicit non-goals of this document

- No training recipe that claims a Dice or HD95 number.
- No statement that the released checkpoints match the paper's parameter or
  FLOP table. That mismatch is tracked in the README ERRATA and upstream
  issues, and this plan does not resolve it.
