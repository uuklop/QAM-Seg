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
| `--ssl-backbone tiny` | Default CPU conv student. Does not import `mamba_ssm` | Yes |
| `--ssl-backbone mamba` | SSL student is `models.MambaLiteUNet` when that import works. A failed import exits non-zero and does not fall back, because the backbone was requested | Yes, optional |
| `--ssl-split PATH` | Slice a local train `.npy` pair with `fraction_5.json`, `fraction_10.json`, or `fraction_20.json` | Yes, local files only |
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
`gate_keep_fraction`, and mean `q`. After the steps, a labeled eval prints
loss, mIoU, Dice (`f1_or_dsc`), accuracy, specificity, sensitivity, and
`hd95`. Those names match `engine.py`. HD95 still uses `utils.summarize_hd95`.
On the smoke path the numbers describe synthetic blobs. They are not
dataset scores and they do not replace the supervised log.

## What is still open

- Run MQE/QGC on a prepared labeled fraction with the optional Mamba
  student. The hook can construct `MambaLiteUNet` after a successful
  import. This repository still does not run that training, and it still
  does not publish a medical Dice or HD95.
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
