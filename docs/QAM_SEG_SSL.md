# QAM-Seg quality-aware SSL plan

Status: **plan only**. This slice does not train a semi-supervised model,
does not download data, and does not change the supervised loss.

QAM-Seg (Quality-Aware Mamba Segmentation) keeps the released MambaLiteUNet
code as a supervised backbone and adds a quality-aware semi-supervised
objective on top. AMF, LGFM, and CGA stay upstream modules. See
`THIRD_PARTY.md`. CVPR metrics for MambaLiteUNet are not QAM-Seg results.

The supervised entry point remains:

```bash
python train.py
```

## Reserved CLI

`train.py` accepts four flags and then **stops**, before the device is
touched and before any epoch runs. Omitting them leaves the supervised path
as it is.

| Flag | Reserved meaning | Wired? |
| --- | --- | --- |
| `--ssl` | Turn on the unlabeled branch below | No |
| `--no-amf` | Ablation: bypass Adaptive Multi-branch Mamba Feature Fusion | No |
| `--no-lgfm` | Ablation: bypass Local-Global Feature Mixing | No |
| `--no-cga` | Ablation: bypass Cross-Gated Attention | No |

```bash
python train.py --ssl
# exits with status 2 and points back at this document
```

`--help` also exits before training. Unknown arguments are ignored so a typo
cannot silently select a different backbone.

The ablation flags are documented here and refused in `train.py` on purpose.
Wiring them halfway would change the supervised baseline without a matching
experiment.

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

## MQE — mask quality estimator (not implemented)

MQE scores a pseudo-mask. It is not a second segmentation architecture and
it is not a claim about AMF.

Planned signals, all computed from tensors the backbone already produces:

1. **Predictive entropy** of the sigmoid mask, averaged over pixels.
2. **View agreement**: Dice between the pseudo-mask under a weak view and a
   strong view of the same unlabeled image. Low agreement means low quality.
3. **Boundary uncertainty**: entropy restricted to a band around the 0.5
   contour, so a confident interior cannot hide a noisy edge.

The estimator returns:

- an image-level score in `[0, 1]`
- an optional pixel-level weight map in `[0, 1]`

No learned MQE head is in this slice. The first implementation should be
the three statistics above, logged on a labeled hold-out where a true mask
exists, so the score can be compared with true Dice before it is trusted on
unlabeled data.

## QGC — quality-gated consistency (not implemented)

QGC uses MQE to decide whether an unlabeled pseudo-mask may supervise
another view.

Planned step, for each unlabeled image:

1. Teacher (or weak view) produces a pseudo-mask.
2. MQE returns an image score `q` and an optional pixel map.
3. **Gate.** If `q` is below a fixed threshold, drop the image for that
   step. Otherwise weight the unsupervised loss by `q` (and by the pixel
   map, if present).
4. Student (or strong view) is trained with the same `BceDiceLoss` against
   the gated pseudo-mask.
5. Ramp the unsupervised coefficient from 0 to a cap over the first part of
   training. The labeled `BceDiceLoss` stays at full weight the whole time.

Empty gates must be safe: a batch where every pseudo-mask is rejected adds
no unsupervised term and does not divide by zero.

This is consistency under a quality gate. It is not a reimplementation of
AMF, LGFM, or CGA, and it does not retune those blocks.

## What a later patch has to include

- A dataloader that can yield labeled and unlabeled batches without
  breaking `isic_loader` for `python train.py`.
- MQE as a pure function with a unit test on synthetic masks.
- QGC behind `--ssl`, default off.
- Ablations `--no-amf`, `--no-lgfm`, `--no-cga` only once each bypass has a
  defined forward and a parameter count from `scripts/count_params_flops.py`.
- Logs that keep the current metrics and add the unlabeled loss, the
  fraction of masks kept by the gate, and HD95.
- No table that presents upstream CVPR numbers as QAM-Seg scores.

## Explicit non-goals of this document

- No training recipe that claims a Dice or HD95 number.
- No statement that the released checkpoints match the paper's parameter or
  FLOP table. That mismatch is tracked in the README ERRATA and upstream
  issues, and this plan does not resolve it.
