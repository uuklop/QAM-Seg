# QAM-Seg

Quality-aware semi-supervised medical image segmentation.

QAM-Seg (Quality-Aware Mamba Segmentation) is a PhD project that keeps a
supervised Mamba segmentation network as its backbone and adds a
quality-aware semi-supervised objective (mask quality estimation and
quality-gated consistency). This repository is a fork of MambaLiteUNet.
MQE and QGC are implemented under `qamseg/` and specified in
[docs/QAM_SEG_SSL.md](docs/QAM_SEG_SSL.md). They are **not experimentally
validated**. No ISIC, HAM10000, or PH2 result is reported here.

This fork does not claim coauthorship of the upstream modules AMF, LGFM, or
CGA, and it does not claim the CVPR 2026 MambaLiteUNet metrics.

## Upstream prior

The supervised backbone, training loop, data loaders, and checkpoints in
this fork come from:

**MambaLiteUNet: Cross-Gated Adaptive Feature Fusion for Robust Skin Lesion Segmentation**
Md Maklachur Rahman, Soon Ki Jung, and Tracy Hammond.
CVPR 2026, pp. 8556–8565.
arXiv: [2604.20286](https://arxiv.org/abs/2604.20286).
Code: [maklachur/MambaLiteUNet](https://github.com/maklachur/MambaLiteUNet).
Paper HTML: [CVPR 2026 open access](https://openaccess.thecvf.com/content/CVPR2026/html/Rahman_MambaLiteUNet_Cross-Gated_Adaptive_Feature_Fusion_for_Robust_Skin_Lesion_Segmentation_CVPR_2026_paper.html).
Slides: [CVPR 2026 slides](https://cvpr.thecvf.com/media/cvpr-2026/Slides/36828_2CEf7jM.pdf).
Poster: [CVPR 2026 poster](https://cvpr.thecvf.com/media/PosterPDFs/CVPR%202026/36828.png?t=1779053789.472832).

Those links, the poster, and any Dice / parameter numbers in that paper
belong to the upstream work. QAM-Seg has not re-run that evaluation and
does not report those numbers as its own. Upstream issues that discuss
paper-versus-code differences are cited in the ERRATA below; the numbers
in those issues are the reporters' numbers, not measurements made by this
fork.

Full attribution and the license gap are in [THIRD_PARTY.md](THIRD_PARTY.md)
and [LICENSE](LICENSE).

## What this fork adds

- MIT license for the QAM-Seg additions, plus an explicit note that upstream
  ships no license.
- This README, rebranded for QAM-Seg. The GitHub "About" description is not
  changed from here.
- [docs/QAM_SEG_SSL.md](docs/QAM_SEG_SSL.md): MQE and QGC. Implemented in
  code, unit-tested on synthetic tensors, not experimentally validated.
- `--ssl` runs that path (`python train.py --ssl` or
  `python -m qamseg.train_ssl`). The default is a short CPU smoke on
  synthetic blobs with a tiny student, not MambaLiteUNet and not a
  benchmark. `--no-amf`, `--no-lgfm`, and `--no-cga` are still refused.
- [scripts/count_params_flops.py](scripts/count_params_flops.py): parameter
  count, and FLOPs when `thop` or `fvcore` can run a forward, at 256×256.
- HD95 (pixels) on the existing validation and test logs, without replacing
  accuracy, sensitivity, specificity, Dice, or mIoU.

No dataset is downloaded and no training run is part of this change.

## ERRATA

Known mismatches between the CVPR paper, as described by readers, and the
code in this fork. Code facts below were checked in this tree. Complexity
and Dice figures were **not** re-measured here; do not treat them as
QAM-Seg results. Use `scripts/count_params_flops.py` on a machine with the
training dependencies if you need a count of this code.

### SS2D versus 1D Mamba

`models/MambaLiteUNet.py` imports `Mamba` from `mamba_ssm` and runs it on a
flattened `(B, H*W, C)` token sequence (`MambaSSMLayer`). There is no SS2D
or VMamba 2D selective-scan class in this repository.

Upstream issue
[#3](https://github.com/maklachur/MambaLiteUNet/issues/3) states that the
paper describes the block as derived from VMamba/SS2D, while the released
code uses `mamba_ssm.Mamba` after that flatten.

### Parameter and FLOP counts

`configs/config_setting.py` sets `c_list` to `[16, 32, 48, 64, 96, 128]`.
That is the configuration this fork builds. This slice does not publish a
parameter or FLOP number for it.

Issue [#3](https://github.com/maklachur/MambaLiteUNet/issues/3) reports that
the paper lists this C3 configuration, with 4 branches, at **0.494M**
parameters and **0.326** GFLOPs at 256×256, and that a count of the
released code came out near **0.911M** trainable parameters. Issue
[#4](https://github.com/maklachur/MambaLiteUNet/issues/4) reports a
follow-up: the same `c_list` has been in the upstream tree since its first
commit, and the shipped checkpoints are the larger model rather than a
0.494M model. Both issues were open when this fork was prepared. QAM-Seg
does not adopt either the paper table or those issue measurements.

### Residual scale and the attention gate

In `AdaptiveMultiBranchMambaFeatureFusion`, `skip_scale` is
`nn.Parameter(torch.ones(1))`, so the residual scale starts at 1. Issue
[#3](https://github.com/maklachur/MambaLiteUNet/issues/3) states that the
paper initializes the AMF residual scale α to 0.

`CrossGatedAttentionGate` in this tree includes per-branch `branch_proj`
and a `combine` convolution. Issue #3 calls those extra stages out as
another paper-versus-code difference. This fork does not remove them.

### Phantom dataset scripts

The upstream README tells readers to run `prepare_HAM10000.py` and
`prepare_PH2.py`. **Those files are not in this repository.** Do not expect
them. Issue #4 also notes their absence. HAM10000 and PH2 are not prepared
by any script here.

Scripts that do exist, under `dataprepare/` (not the repository root):

| Script | In repo? |
| --- | --- |
| `dataprepare/prepare_ISIC2017.py` | yes |
| `dataprepare/prepare_ISIC2018.py` | yes |
| `dataprepare/prepare_custom_dataset.py` | yes |
| `prepare_HAM10000.py` | **no** |
| `prepare_PH2.py` | **no** |

The ISIC prepare scripts call `scipy.misc.imread` and `scipy.misc.imresize`,
which are not in current SciPy. The upstream README's separate Python 3.7
environment with `scipy==1.2.1` is still the note that applies to those
files. They also read images from `./ISIC2017/` or `./ISIC2018/` relative
to the working directory and write `data_*.npy` / `mask_*.npy` into the
working directory. `loader.py` reads those arrays from `config.data_path`
(`./data/ISIC2017/` or `./data/ISIC2018/`). Move the six `.npy` files into
that directory after preparation. The old README paths `/data/ISIC2017/`
and a root-level `python prepare_ISIC2017.py` do not match this tree.

## 1. Environment setup (supervised backbone)

The supervised stack is unchanged. Versions below are the ones pinned by
the upstream `requirements.txt`.

1. Create a Conda environment with Python 3.9:

   ```bash
   conda create -n mambaliteunet python=3.9 -y
   conda activate mambaliteunet
   ```

2. Upgrade pip (optional):

   ```bash
   pip install --upgrade pip
   ```

3. Install PyTorch with CUDA 11.8:

   ```bash
   pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```

   Or:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. Install the remaining pins:

   ```bash
   pip install -r requirements.txt
   ```

   If `causal_conv1d` or `mamba_ssm` fails to build:

   ```bash
   pip install --no-cache-dir causal_conv1d mamba_ssm
   ```

5. Check CUDA visibility:

   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```

`train.py` (with no flags) and `test.py` call `.cuda()` and
`torch.cuda.empty_cache()`. They are the upstream GPU training path.
`--ssl` is a separate CPU smoke loop and does not replace that path.

## 2. Dataset preparation

Do not download datasets as part of reading this README. The steps are the
upstream ones, with the paths corrected to files that exist.

### ISIC 2017

1. Download from the [ISIC 2017 challenge](https://challenge.isic-archive.com/data).
2. Place the training images and ground truth where
   `dataprepare/prepare_ISIC2017.py` looks: `./ISIC2017/` relative to the
   working directory (see the `Dataset_add` variable in that file).
3. Run:

   ```bash
   python dataprepare/prepare_ISIC2017.py
   ```

4. Move `data_train.npy`, `data_val.npy`, `data_test.npy`, `mask_train.npy`,
   `mask_val.npy`, and `mask_test.npy` to `./data/ISIC2017/`.
5. In `configs/config_setting.py`, set `datasets = 'ISIC2017'`.

### ISIC 2018

Same pattern. The script reads `./ISIC2018/` and
`configs/config_setting.py` defaults `datasets` to `'ISIC2018'` with
`data_path = './data/ISIC2018/'`.

```bash
python dataprepare/prepare_ISIC2018.py
```

### HAM10000 and PH2

No prepare script is shipped. See ERRATA. Nothing in this fork reproduces
those splits.

### Custom data

`dataprepare/prepare_custom_dataset.py` expects `images/*.png` and
`masks/*.png` beside the process working directory and writes the same six
`.npy` names. The split sizes are the variables at the top of that file.
Point `config.data_path` at the directory that holds the arrays. The config
object currently raises unless `datasets` is `ISIC2017` or `ISIC2018`; a
custom folder still needs that branch edited before `train.py` will open it.

## 3. Supervised training and testing

### 3.1 Train

```bash
python train.py
```

- Logs and checkpoints go under `./results/`.
- The loop runs `test_one_epoch` after the last epoch when `best.pth`
  exists. That call is in `train.py`.
- Default optimizer, loss, input size (256×256), and `c_list` are in
  `configs/config_setting.py`.

Ablation flags are refused with exit code 2 before the model is built:

```bash
python train.py --no-amf
python train.py --no-lgfm
python train.py --no-cga
```

`python train.py --help` prints the flag text and does not train.
`python train.py --ssl` does not enter the supervised loop; see section 4.

### 3.2 Test

`test.py` loads an upstream checkpoint and runs `test_one_epoch`.
The default path is:

```text
pre_trained_models/ISIC2018/best-epoch53-loss0.2218.pth
```

An ISIC 2017 checkpoint is commented next to it:

```text
pre_trained_models/ISIC2017/best-epoch51-loss0.1905.pth
```

These files are the weights shipped with the upstream repository. QAM-Seg
does not report a score for them.

```bash
python test.py
```

Outputs go under `./results/`.

### 3.3 Metrics

On validation epochs where `epoch % config.val_interval == 0`, and on every
test pass, the log still reports loss, mIoU, Dice (`f1_or_dsc`), accuracy,
specificity, sensitivity, and the confusion matrix, pooled over pixels the
way `engine.py` already did.

HD95 is an extra field, `hd95:`, in pixels at the mask resolution. It is the
symmetric 95th percentile of Euclidean surface distances (SciPy
`distance_transform_edt`). A pair of empty masks is distance 0. A pair with
only one empty mask is skipped and counted as undefined. If SciPy cannot be
imported, or the distance raises, the field is `n/a (...)` and the other
metrics are unchanged. There is no spacing metadata in the loader, so this
is not millimetres, and it is not a number from the CVPR paper.

### 3.4 Params and FLOPs

```bash
python scripts/count_params_flops.py
```

Default input is `1×3×256×256` on CPU. The script reads `c_list` from
`configs/config_setting.py` without launching training.

- If PyTorch or `mamba_ssm` is missing it prints `status: stub` and the
  install commands. That is expected on a machine without the backbone
  dependencies.
- When the class constructs, it prints `params_total` and
  `params_trainable` without a forward.
- FLOPs are printed only if `thop` or `fvcore` is installed and the forward
  succeeds. `thop` numbers are labeled as multiply-accumulates. `fvcore`
  numbers are that library's FLOP count. Neither is converted into the
  paper table. A CUDA-only `mamba_ssm` forward should be retried with
  `--device cuda` on a GPU machine.

Optional install, not pinned in `requirements.txt`:

```bash
pip install thop
# or
pip install fvcore
```

## 4. Quality-aware SSL

MQE (`qamseg/mqe.py`) and QGC (`qamseg/qgc.py`) are implemented.
`python train.py` with no flags does not run them. `python train.py --ssl`
and `python -m qamseg.train_ssl` do. The default student is a tiny
convolution, the default images are synthetic blobs, and the default length
is two optimizer steps on CPU. That checks the wiring. It is not a
training result, and it does not claim a Dice or HD95 score.

```bash
python train.py --ssl
python -m unittest discover -s tests -v
```

The step log adds unlabeled loss and `gate_keep_fraction`. The eval log
keeps loss, mIoU, Dice (`f1_or_dsc`), accuracy, specificity, sensitivity,
and HD95. Ablations `--no-amf`, `--no-lgfm`, and `--no-cga` still exit 2.
AMF, LGFM, and CGA are unchanged upstream modules. Details and the
unlabeled-data TODO are in [docs/QAM_SEG_SSL.md](docs/QAM_SEG_SSL.md).

## 5. Citation

Cite the upstream paper for the backbone. This fork does not add a QAM-Seg
paper citation.

```bibtex
@inproceedings{rahman2026mambaliteunet,
  title={MambaLiteUNet: Cross-Gated Adaptive Feature Fusion for Robust Skin Lesion Segmentation},
  author={Rahman, Md Maklachur and Jung, Soon Ki and Hammond, Tracy},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8556--8565},
  year={2026}
}
```

## 6. Acknowledgements

The ISIC 2017 and ISIC 2018 challenges, and the releases of HAM10000 and PH2,
are the datasets the upstream work credits. Dataset access is unchanged by
this fork. The upstream README also acknowledges
[VMamba](https://github.com/MzeroMiko/VMamba),
[Vision Mamba](https://github.com/hustvl/Vim),
[VM-UNet](https://github.com/JCruan519/VM-UNet), and
[UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet).

## 7. Issues

GitHub issue tracking for a fork is a repository setting. This tree does not
turn that setting on. Questions about AMF, LGFM, CGA, or the CVPR tables
belong with the upstream authors; the open upstream threads are
[issue 3](https://github.com/maklachur/MambaLiteUNet/issues/3) and
[issue 4](https://github.com/maklachur/MambaLiteUNet/issues/4).
