# MedIA notes for QAM-Seg

Status: protocol and training hooks are in the tree. No medical Dice,
mIoU, HD95, or ASSD in this repository is a result. Nothing here
downloaded ISIC, HAM10000, PH2, or a colonoscopy set.

MambaLiteUNet is the backbone (Rahman, Jung, and Hammond, CVPR 2026).
It is not the contribution. QAM-Seg does not claim that paper's tables.

## What this round added

- MQE reweight of the unlabeled consistency loss (`--quality-reweight sample|pixel`), the PL-QW pattern without a new network. Default `off` keeps the previous QGC objective. Gaussian ramp by default. `--ssl-lambda-qw` defaults to 1 (QG-SSL published 0.25).
- Fair baselines on the same student and the same splits: plain mean teacher, entropy UA-MT, confidence-threshold pseudo-labels, and supervised (`λ_u = 0`).
- Multi-seed runner and mean±std aggregation for Dice, mIoU, HD95, and ASSD.
- QAR is **skipped**. See below.

The CPU fallback remains `TinySegStudent` (`--ssl-backbone tiny`). There is no ConvLiteUNet class. `--ssl-backbone mamba` still exits if `MambaLiteUNet` does not import. `--no-amf`, `--no-lgfm`, and `--no-cga` are still refused.

## Claimed contribution (do not inflate)

QAM-Seg is quality-aware semi-supervised segmentation on a mean teacher:

1. MQE scores a pseudo-mask from entropy, teacher–student disagreement, cross-view agreement, and boundary-band entropy.
2. QGC gates the consistency loss with that score.
3. BAQ is an optional spatial prior (high image gradient and low quality).
4. Sample or pixel reweighting uses the same score as a PL-QW weight, with a ramp, and without training `g_φ`.

Not claimed: a new backbone, a quality-predictor network, SAM / CLIP / YOLO / text modules, or a finished multi-dataset MedIA table.

## Why this is not QG-SSL

QG-SSL (Abhishek and Hamarneh, MICCAI 2026, arXiv:2606.01753, code `sfu-mial/QG-SSL`) trains `g_φ` on synthetic corruptions plus imperfect mid-training masks, freezes it, and then either

- QAR: minimize `1 - g_φ(x, f_θ(x))` so gradients flow through the quality net into the segmentor (`λ_qar ≈ 0.01`), or
- PL-QW: multiply each pseudolabel loss by `w_j = g_φ(x, ŷ)` with `g_φ` detached (`λ_qw ≈ 0.25`).

They report that pair as a drop-in on MT, UA-MT, ICT, PL-T, and PL-C, over three seeds, as Dice and IoU mean±std.

QAM-Seg does not train or freeze a second network. The score is an in-loop function of the teacher mask. Reweighting is the PL-QW half of that idea. QAR is omitted on purpose: a gradient through MQE is entropy minimization plus consistency. That duplicates QGC and the UA-MT baseline, and it pushes boundary confidence in the place BAQ down-weights. `qamseg/qar.py` raises if something tries to call it.

## Flags

| Flag | Default | Role |
| --- | --- | --- |
| `--quality-reweight` | `off` | `sample` or `pixel` MQE weights. `off` is the old QGC loss |
| `--ssl-lambda-qw` | `1` | Scale on the reweighted term only. Set `0.25` to match QG-SSL |
| `--ssl-qw-ramp` | `gaussian` | `linear` or `none`. Last step reaches a pure quality weight for gaussian and linear |
| `--ssl-gate` | `gated` | `ungated` is uniform consistency |
| `--ssl-baq` | off | Boundary-aware pixel prior |
| `--ssl-method` | `qam` | `mt`, `uamt`, `pl`, `supervised` override gate, BAQ, and reweight |
| `--ssl-uncertainty` | `off` | `entropy` is set by `--ssl-method uamt` |
| `--ssl-uncertainty-tau` | `0.75` | Drop high-entropy teacher pixels. Placeholder |
| `--ssl-pl-threshold` | `0` (preset `pl` uses `0.95`) | Confidence keep. Placeholder |
| `--ssl-out-dir` | empty | Writes `metrics.json` |
| `--ssl-eval-npy` | empty | Held-out images. Allowed to be `data_val.npy` or `data_test.npy` |
| `--ssl-recipe` / `--ssl-fraction` / `--ssl-group` | empty | Labels stored in `metrics.json`. They do not change the loss |

`python train.py` with no flags is still the supervised GPU entry. `python train.py --ssl` is still the short CPU smoke.

Ablation names used by the runner:

| Recipe | Flags |
| --- | --- |
| `qam-ungated` | gate ungated, reweight off, BAQ off |
| `qam-gated` | gate gated, reweight off, BAQ off |
| `qam-gated-baq` | gate gated, BAQ on, reweight off |
| `qam-gated-baq-qw` | gate gated, BAQ on, sample reweight |
| `qam-qw` | gate ungated, sample reweight (clean PL-QW analogue) |
| `qam-qw-pixel` | registered, not in the default matrix |

Baselines in the default matrix: `supervised` (5/10/20 and full), `mt`, `uamt`, `pl`.

## Commands

Dry run (no training, no data):

```bash
python scripts/run_media_protocol.py --dry-run --out-dir results/media_protocol
```

Synthetic wiring check:

```bash
python scripts/run_media_protocol.py --smoke --seeds 0 --fractions 10 \
  --recipes qam-qw,qam-gated-baq-qw,mt,uamt --out-dir /tmp/qam-smoke
python scripts/aggregate_media_results.py /tmp/qam-smoke
```

Real run, after the arrays exist locally. `--steps` is required and is not a tuned budget. The runner writes `splits/` under the output directory for seeds 0, 1, and 2.

```bash
python scripts/run_media_protocol.py \
  --train-npy ./data/ISIC2018/data_train.npy \
  --mask-npy ./data/ISIC2018/mask_train.npy \
  --eval-npy ./data/ISIC2018/data_test.npy \
  --eval-mask-npy ./data/ISIC2018/mask_test.npy \
  --backbone mamba \
  --device cuda \
  --steps 20000 \
  --out-dir results/media_isic2018

python scripts/aggregate_media_results.py results/media_isic2018
```

Missing arrays exit 2. The script does not download a substitute. Do not point `--train-npy` at `data_val.npy` or `data_test.npy`. Those names are valid only as `--eval-npy`.

## Tables the paper still needs

Aggregation writes one row per recipe, fraction, backbone, data tag, and eval split. A MedIA draft wants at least:

1. Main comparison at 5%, 10%, and 20% labeled, seeds 0/1/2, mean±std Dice, mIoU, HD95, ASSD: supervised-on-fraction, MT, UA-MT, PL, and the QAM recipe you actually claim.
2. Supervised full (100% of the train split) on the same test set.
3. Ablation: ungated, gated, gated+BAQ, gated+BAQ+sample-reweight. Report `qam-qw` next to that if the claim is the reweight rather than the hard gate.
4. Optional: `qam-qw-pixel` versus `qam-qw`, and `--ssl-lambda-qw 0.25` versus the default 1.
5. Same backbone for every column. `tiny` is a smoke student. The paper student is MambaLiteUNet, cited as the backbone.
6. Held-out test eval. `eval_split=labeled_train_subset` in `metrics.json` means the number is not a test score. Do not put those rows in a table.
7. More than one dataset before a journal submission. This tree can prepare ISIC 2017 and ISIC 2018 only. HAM10000 and PH2 still have no prepare script. Colonoscopy is not wired. No new dataset was added in this change.

Hyperparameters that are still placeholders: EMA 0.99, `tau` ramp 0.4→0.8, `λ_u` max 1, uncertainty tau 0.75, PL threshold 0.95, MQE cue weights, and the step budget. Say so, or retune on validation only.

## Checklist for James

- [ ] Prepare ISIC 2018 (and 2017 if it is in the paper) with `dataprepare/`. The prepare scripts still need the old SciPy `imread` / `imresize` note in the README. Move the six `.npy` files to `./data/ISIC2018/` or `./data/ISIC2017/`.
- [ ] Do not use validation or test images as the unlabeled pool.
- [ ] Confirm `mask_train.npy` lines up with `data_train.npy` row order. The fraction tool uses row ids `"0" .. "N-1"`.
- [ ] Run the protocol with `--backbone mamba` on a GPU, seeds 0, 1, 2, fractions 5/10/20, and `--eval-npy` / `--eval-mask-npy` pointed at the test arrays.
- [ ] Aggregate with `scripts/aggregate_media_results.py`. Drop any row whose `data` is `synthetic` or whose `eval_split` is not `held_out`.
- [ ] Decide whether the claimed system is gated+BAQ, gated+BAQ+reweight, or ungated reweight (`qam-qw`), from the validation split, then freeze that choice before touching the test set.
- [ ] If a second dataset is in the paper, add a prepare path first. Do not download it from this agent workflow, and do not stack SAM, CLIP, YOLO, or a text tower to get there.
- [ ] Keep the CVPR MambaLiteUNet numbers out of the QAM-Seg tables.
- [ ] QAR stays out unless a design appears that is not entropy minimization, not a second consistency term, and not a fight with BAQ. That design is not in this tree.
