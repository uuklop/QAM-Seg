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
| `--ssl-backbone tiny` | `TinySegStudent`, the CPU convolutional fallback. Does not import `mamba_ssm`. There is no separate ConvLiteUNet class | Yes |
| `--ssl-backbone mamba` | SSL student is `models.MambaLiteUNet` when that import works. A failed import exits non-zero and does not fall back, because the backbone was requested | Yes, optional |
| `--ssl-split PATH` | Slice a local train `.npy` pair with `fraction_5.json`, `fraction_10.json`, or `fraction_20.json` | Yes, local files only |
| `--ssl-gate ungated` | Uniform consistency weights. With reweight off, this is plain mean teacher | Yes |
| `--quality-reweight {off,sample,pixel}` | MQE reweight of the unlabeled term. `off` (default) keeps the previous QGC loss | Yes |
| `--ssl-lambda-qw` | Coefficient on that reweighted term. Default `1`. QG-SSL used `0.25`. Ignored when reweight is off | Yes |
| `--ssl-qw-ramp {gaussian,linear,none}` | Mix from uniform weights to the MQE score. Default `gaussian` | Yes |
| `--ssl-method {qam,mt,uamt,pl,supervised}` | `qam` honors the flags above. The other four force a shared-student baseline | Yes |
| `--ssl-out-dir` | Write `metrics.json` for the aggregation script | Yes |
| `--ssl-eval-npy` / `--ssl-eval-mask-npy` | Held-out eval only. Val/test names are allowed here and still refused as training pools | Yes |
| `--no-amf` | Ablation: bypass Adaptive Multi-branch Mamba Feature Fusion | No, exit 2 |
| `--no-lgfm` | Ablation: bypass Local-Global Feature Mixing | No, exit 2 |
| `--no-cga` | Ablation: bypass Cross-Gated Attention | No, exit 2 |

```bash
python train.py
python train.py --ssl
python -m qamseg.train_ssl
```

`python train.py --ssl` does not read `isic_loader` and does not download
data. It runs `qamseg.train_ssl`. The default student is a tiny conv net
so the loss, the gate, and the log line can execute without a GPU and
without `mamba_ssm`. `--ssl-steps` defaults to 2 for that reason.
`--ssl-backbone mamba` is the only switch that constructs
`MambaLiteUNet`, and only after `models.MambaLiteUNet` imports. If that
import fails, the process prints the reason and exits non-zero. It does
not silently keep the tiny student. Omitting the flag, or passing
`--ssl-backbone tiny`, never imports the backbone. Passing
`--ssl-backbone` without `--ssl` prints one line and leaves supervised
training unchanged.

Passing `--no-amf`, `--no-lgfm`, or `--no-cga` with or without `--ssl`
still exits 2. Those bypasses are not wired; turning them on would change
the supervised baseline without a matching forward.

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

## MQE reweight (PL-QW analogue)

Code: `qamseg/reweight.py`. Tests: `tests/test_reweight.py`.

QG-SSL weights a pseudolabel loss by a frozen external network,
`w_j = g_φ(x, ŷ)`. QAM-Seg uses the same weighting pattern with the
score QGC already computed. No third network is constructed.

```text
α(t) = exp(-5 (1 - t/T)^2)          # --ssl-qw-ramp gaussian; T = steps-1
w_j(t) = (1 - α(t)) + α(t) * q_j     # sample mode; q_j is detached MQE
L = L_sup + λ_u(t) * λ_qw * mean_j(w_j(t) * ℓ_j)
```

`pixel` mode uses the MQE pixel map in place of `q_j`. `--quality-reweight off`
does not take this path: the unlabeled term is the previous QGC loss,
including its weight normalization. `--ssl-lambda-qw` defaults to 1 so
an ablation compares weights rather than a second scale. Pass `0.25` to
use the coefficient published with QG-SSL.

When reweighting is on, a `hard` or `gated` gate contributes only the
binary keep (`q >= tau`). It does not multiply by `q` a second time.
BAQ still edits the pixel map before that keep. `soft` does stack with
the reweight; the MedIA recipes do not use that combination.

`qam-qw` in the protocol runner is ungated sample reweight: the direct
analogue. `qam-gated-baq-qw` is the requested stacked ablation
(gate + BAQ + sample reweight).

## Quality-aware regularization

Not implemented. QG-SSL's QAR backpropagates `1 - g_φ(x, f_θ(x))` through
a frozen quality network. MQE has no such network. Differentiating the
existing cues would minimize entropy and repeat the consistency loss,
which is the UA-MT baseline and QGC, and it would push contours to be
confident where BAQ is trying to be careful. `qamseg/qar.py` records that
skip and raises if called. There is no `--ssl-qar` flag.

## Baselines on the same student

`--ssl-method` selects a preset. `qam` is the only mode that keeps
`--ssl-gate`, `--ssl-baq`, and `--quality-reweight`.

| Method | Unlabeled term |
| --- | --- |
| `mt` | EMA teacher, ungated consistency. MQE is logged and not used in the loss |
| `uamt` | Same, with teacher predictive entropy as the weight. Pixels above `--ssl-uncertainty-tau` (default 0.75) are dropped. This is not Monte Carlo dropout inside MambaLiteUNet |
| `pl` | Hard pseudo-labels where `max(p, 1-p)` is at least `--ssl-pl-threshold` (default 0.95 when the preset is selected) |
| `supervised` | `λ_u = 0`. Unlabeled images are not used in the loss |

UA-MT and PL are the fair confidence baselines. They do not call MQE for weights.

## MedIA protocol

`scripts/run_media_protocol.py` lists seeds 0, 1, and 2, labeled fractions
5/10/20, and a supervised full job. Recipes: `supervised`, `mt`, `uamt`,
`pl`, `qam-ungated`, `qam-gated`, `qam-gated-baq`, `qam-gated-baq-qw`,
and `qam-qw`. `qam-qw-pixel` is registered and off by default.

```bash
python scripts/run_media_protocol.py --dry-run --out-dir results/media_protocol
python scripts/run_media_protocol.py --smoke --recipes qam-qw,mt --seeds 0 --fractions 10
```

A real run refuses to start without `--train-npy`, `--mask-npy`, and
`--steps`. It writes splits with the existing fraction tool and does not
download data. `--backbone mamba` is the student for that run. The default
`--backbone tiny` stays the CPU fallback.

`scripts/aggregate_media_results.py` reads each `metrics.json` and writes
`summary.csv` and `summary.md` with mean ± sample std (ddof=1) for Dice,
mIoU, HD95, and ASSD. One seed leaves the std blank. Synthetic rows are
not medical scores. HD95 and ASSD are pixels, not millimetres.

Pass `--ssl-eval-npy` (or the runner's `--eval-npy`) with a local
`data_val.npy` / `data_test.npy` when those arrays exist. Until then the
eval split recorded in `metrics.json` is the labeled train subset.

## Related work

**MambaLiteUNet** (Rahman, Jung, and Hammond, CVPR 2026) is the supervised
backbone this fork trains. AMF, LGFM, and CGA are upstream modules.
QAM-Seg does not claim that paper's Dice or parameter table.

**QG-SSL** (Abhishek and Hamarneh, MICCAI 2026, arXiv:2606.01753) is the
closest quality-aware SSL method. It trains a separate predictor `g_φ` on
variable-quality masks, then freezes it. QAR pushes the segmentor to raise
`g_φ(x, f_θ(x))`. PL-QW reweights pseudolabel losses by `g_φ(x, ŷ)` with
`λ_qw = 0.25`. The predictor is a second phase and a second network, and
the paper applies it on top of MT, UA-MT, ICT, PL-T, and PL-C.

QAM-Seg does not add `g_φ`. The quality score is computed inside the
mean-teacher step from entropy, disagreement, cross-view agreement, and
boundary-band entropy (MQE), then used to gate consistency (QGC),
optionally to edit pixel weights at strong image gradients (BAQ), and,
when `--quality-reweight` is on, to reweight the unlabeled term the way
PL-QW reweights with `g_φ`. QAR is omitted for the reason above.

Uncertainty-weighted consistency (UA-MT) and confidence-threshold
pseudo-labels are baselines in the protocol, not part of the QAM-Seg
objective. They use the teacher's own confidence. MQE does not.

The checklists for a Medical Image Analysis submission, including the
runs that still need local data, are in
[MEDIA_NOTES.md](MEDIA_NOTES.md).

The smoke augmentations are a horizontal flip (weak) and a vertical flip
plus a corner cutout (strong). They are not the full weak/strong recipe.
The EMA decay defaults to 0.99 and is a placeholder.

## Data

`qamseg/data.py` builds a labeled loader and an unlabeled loader.
`loader.isic_loader` is unchanged and is not imported by that module.
The default datasets are synthetic blobs. A local `.npy` may be passed
with `--ssl-unlabeled-npy` or `--ssl-labeled-npy` plus `--ssl-mask-npy`.
A missing file raises `FileNotFoundError` and names the split tool.
Nothing in this path downloads ISIC, HAM10000, or PH2. Paths whose
basename is `data_val.npy`, `data_test.npy`, `mask_val.npy`, or
`mask_test.npy` are refused.

## Labeled fractions (5% / 10% / 20%)

`qamseg/splits.py` and `scripts/make_labeled_fraction_splits.py` write a
deterministic partition of one **train** id list. The seed, the RNG
(`numpy.random.Generator` over `PCG64`), and the count rule are stored in
every JSON file, along with the NumPy version that supplied PCG64:

```text
n_labeled = min(n, max(1, floor(fraction * n + 0.5)))
```

One Fisher-Yates permutation is drawn from `PCG64(seed).random_raw` with unbiased uint64 rejection. It does not call `Generator.permutation`. The 5% labeled ids are a prefix of the 10%
prefix of the 20% prefix. The unlabeled pool at each fraction is the
complement inside that same train list. Validation and test ids are not
sampled. If `--heldout-ids` overlaps the train list, the tool exits 2
and writes nothing.

Give the tool exactly one train source:

```bash
python scripts/make_labeled_fraction_splits.py \
  --train-npy ./data/ISIC2018/data_train.npy \
  --name isic2018 \
  --seed 0 \
  --out-dir splits

python scripts/make_labeled_fraction_splits.py \
  --ids-file ./local/isic2018_train_ids.txt \
  --name isic2018 \
  --seed 0

python scripts/make_labeled_fraction_splits.py \
  --train-dir ./local/train_images_only \
  --pattern '*.jpg' \
  --name custom \
  --seed 0
```

Files land in `splits/<name>/seed<seed>/`:

| File | Role |
| --- | --- |
| `manifest.json` | Seed, count rule, source, and file index |
| `fraction_5.json`, `fraction_10.json`, `fraction_20.json` | Labeled and unlabeled ids plus row indices |
| `labeled_*.csv`, `unlabeled_*.csv` | Same ids as CSV (`id`, `row`, `role`) |
| `membership.csv` | One row per train id, flags for each fraction |

`--train-npy` uses row ids `"0" .. "N-1"` and does not copy pixels. Pass
`data_train.npy` only. `data_val.npy`, `data_test.npy`, and `mask_*.npy`
are refused. `--ids-file` is a text file (one id per line), a JSON list,
or a CSV. Row `i` in that file is row `i` of the train array.

`--train-dir` sorts file stems lexicographically. That is **not** the
unsorted `glob` in `dataprepare/prepare_ISIC2017.py` and
`prepare_ISIC2018.py`. Those scripts read the whole challenge input
folder, then slice train/val/test by position, so that folder still
contains the images that become `data_val.npy` and `data_test.npy`. Do
not pass it as `--train-dir`. Use `--train-npy` on the train array, or an
id file that lists train ids only.

A missing path exits 2 with the path in the message. The tool does not
download a dataset to fill the gap.

## Pointing the SSL loader at a split

Once the local train arrays exist, one fraction is enough. The loader
keeps masks for `labeled_rows` and drops masks for `unlabeled_rows`. It
checks that the two sets partition the train array and that `n_train`
matches the leading dimension. A missing JSON or `.npy` raises
`FileNotFoundError` / exit 2. No download runs.

```bash
python train.py --ssl \
  --ssl-split splits/isic2018/seed0/fraction_10.json \
  --ssl-labeled-npy ./data/ISIC2018/data_train.npy \
  --ssl-mask-npy ./data/ISIC2018/mask_train.npy \
  --ssl-batch-size 2 \
  --ssl-steps 2
```

Add `--ssl-backbone mamba` on a machine where `MambaLiteUNet` imports.
The default without that flag stays `TinySegStudent`, including when the
split flags are set. Eval printed after these steps is the labeled
subset used for the run. It is not a held-out test score and it is not a
Dice or HD95 result on ISIC, HAM10000, or PH2.

`python train.py` with no flags does not read `splits/` and does not
build the SSL student.

## Logs

Each SSL step prints supervised loss, unlabeled loss (`unsup`), `lambda_u`,
`gate_keep_fraction`, mean `q`, `reweight`, and `qw_alpha`. `qw_alpha` is
0 when `--quality-reweight` is off. After the steps, a labeled eval prints
loss, mIoU, Dice (`f1_or_dsc`), accuracy, specificity, sensitivity,
`hd95`, and `assd`. Those names match `engine.py`. HD95 and ASSD use
`utils.summarize_hd95` and `utils.summarize_assd`.
On the smoke path the numbers describe synthetic blobs. They are not
dataset scores and they do not replace the supervised log.

## What is still open

- Run the MedIA protocol on a prepared labeled fraction with
  `--backbone mamba`, three seeds, and a held-out `--eval-npy`. This
  repository still does not run that training, and it still does not
  publish a medical Dice, mIoU, HD95, or ASSD. See
  [MEDIA_NOTES.md](MEDIA_NOTES.md).
- HAM10000 and PH2 still have no prepare script in this tree. Do not use
  test images as unlabeled data.
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
