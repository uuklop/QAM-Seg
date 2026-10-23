"""CLI for supervised training and the quality-aware SSL path.

``python train.py`` with no flags does not enter SSL. ``--ssl`` dispatches
to ``qamseg.train_ssl`` before the supervised stack is imported.

``--no-amf``, ``--no-lgfm``, and ``--no-cga`` stay refused. Those bypasses
are not wired, and turning them on would change the supervised baseline
without a matching forward.
"""

import argparse


RESERVED_FLAGS = (
    ('no_amf', '--no-amf', 'AMF ablation is not wired.'),
    ('no_lgfm', '--no-lgfm', 'LGFM ablation is not wired.'),
    ('no_cga', '--no-cga', 'CGA ablation is not wired.'),
)


def build_parser():
    parser = argparse.ArgumentParser(
        prog='train.py',
        description=(
            'Supervised MambaLiteUNet training used as the QAM-Seg backbone. '
            '--ssl runs the quality-aware SSL path (synthetic CPU smoke by default). '
            'Ablation flags are still refused.'
        ),
    )
    parser.add_argument(
        '--ssl',
        action='store_true',
        help=(
            'Enable MQE + QGC. Default is a short CPU run on synthetic blobs, '
            'not an ISIC/HAM/PH2 experiment. See docs/QAM_SEG_SSL.md.'
        ),
    )
    parser.add_argument(
        '--no-amf',
        action='store_true',
        help='Reserved ablation. Does not bypass Adaptive Multi-branch Mamba Feature Fusion.',
    )
    parser.add_argument(
        '--no-lgfm',
        action='store_true',
        help='Reserved ablation. Does not bypass Local-Global Feature Mixing.',
    )
    parser.add_argument(
        '--no-cga',
        action='store_true',
        help='Reserved ablation. Does not bypass Cross-Gated Attention.',
    )
    parser.add_argument('--ssl-steps', type=int, default=2, help='SSL optimizer steps. Default is a smoke length.')
    parser.add_argument('--ssl-device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--ssl-batch-size', type=int, default=2)
    parser.add_argument('--ssl-image-size', type=int, default=32)
    parser.add_argument(
        '--ssl-gate',
        default='gated',
        choices=('hard', 'soft', 'gated', 'ungated'),
        help=(
            'QGC mode. ungated is uniform consistency (plain mean teacher when '
            'quality reweight is also off). Default gated keeps the previous objective.'
        ),
    )
    parser.add_argument('--ssl-level', default='both', choices=('image', 'pixel', 'both'))
    parser.add_argument('--ssl-tau0', type=float, default=0.4)
    parser.add_argument('--ssl-tau1', type=float, default=0.8)
    parser.add_argument('--ssl-warm-steps', type=int, default=1)
    parser.add_argument('--ssl-lambda-max', type=float, default=1.0)
    parser.add_argument('--ssl-gamma', type=float, default=1.0)
    parser.add_argument('--ssl-seed', type=int, default=0)
    parser.add_argument(
        '--ssl-unlabeled-npy',
        default='',
        help='Optional local unlabeled image .npy. Missing paths error. No dataset download.',
    )
    parser.add_argument(
        '--ssl-labeled-npy',
        default='',
        help='Optional local labeled image .npy. Requires --ssl-mask-npy. No dataset download.',
    )
    parser.add_argument('--ssl-mask-npy', default='', help='Mask .npy paired with --ssl-labeled-npy.')
    parser.add_argument(
        '--ssl-split',
        default='',
        help=(
            'Fraction JSON from scripts/make_labeled_fraction_splits.py '
            '(fraction_5.json, fraction_10.json, or fraction_20.json). '
            'Requires local train .npy paths. Missing files error. No download.'
        ),
    )
    parser.add_argument(
        '--ssl-backbone',
        default='tiny',
        choices=('tiny', 'conv', 'mamba'),
        help=(
            'SSL student. tiny (default) is the CPU smoke stand-in. '
            'conv is models.ConvLiteUNet, a capacity-matched CNN fallback that does not '
            'import mamba_ssm and must not be reported as Mamba. '
            'mamba imports models.MambaLiteUNet and exits non-zero if that import fails. '
            'Ignored unless --ssl is set. Not a CVPR run.'
        ),
    )
    parser.add_argument(
        '--use-learned-quality-head',
        action='store_true',
        help='Acknowledge the MQE head stub. It is not trained and its fusion weight stays 0.',
    )
    parser.add_argument(
        '--ssl-baq',
        action='store_true',
        help='Multiply pixel weights by the optional boundary-aware prior. Off by default.',
    )
    parser.add_argument(
        '--quality-reweight',
        default='off',
        choices=('off', 'sample', 'pixel'),
        help=(
            'MQE reweight of the unlabeled term (QG-SSL PL-QW analogue, no g_phi). '
            'off (default) keeps QGC. sample weights each image by the ramped MQE score. '
            'pixel weights each pixel by the ramped MQE map. See docs/QAM_SEG_SSL.md.'
        ),
    )
    parser.add_argument(
        '--ssl-lambda-qw',
        type=float,
        default=1.0,
        help=(
            'Coefficient on the reweighted unlabeled term. Ignored when --quality-reweight off. '
            'Default 1 keeps the unlabeled scale aligned with --ssl-lambda-max. '
            'QG-SSL used 0.25; pass 0.25 to match that coefficient.'
        ),
    )
    parser.add_argument(
        '--ssl-qw-ramp',
        default='gaussian',
        choices=('gaussian', 'linear', 'none'),
        help=(
            'Ramp of the MQE weight from uniform (step 0) to the quality score (last step). '
            'gaussian matches the lambda_u shape. none uses the quality score immediately.'
        ),
    )
    parser.add_argument(
        '--ssl-method',
        default='qam',
        choices=('qam', 'mt', 'uamt', 'pl', 'supervised'),
        help=(
            'Recipe preset. qam (default) honors the gate, BAQ, and reweight flags. '
            'mt, uamt, pl, and supervised replace those with a fair baseline on the same student. '
            'uamt uses teacher entropy, not dropout inside MambaLiteUNet.'
        ),
    )
    parser.add_argument(
        '--ssl-uncertainty',
        default='off',
        choices=('off', 'entropy'),
        help='UA-MT weight from teacher predictive entropy. Off unless --ssl-method uamt sets it.',
    )
    parser.add_argument(
        '--ssl-uncertainty-tau',
        type=float,
        default=0.75,
        help='Drop pixels whose normalized entropy exceeds this value. 1 keeps every pixel. Placeholder.',
    )
    parser.add_argument(
        '--ssl-pl-threshold',
        type=float,
        default=0.0,
        help='Confidence-threshold pseudo-labels. 0 disables. --ssl-method pl uses 0.95 unless this is set.',
    )
    parser.add_argument(
        '--ssl-recipe',
        default='',
        help='Name stored in metrics.json (for example qam-gated-baq-qw). Does not change the loss.',
    )
    parser.add_argument(
        '--ssl-group',
        default='',
        help='Table group stored in metrics.json (ablation, baseline, or supervised). Does not change the loss.',
    )
    parser.add_argument(
        '--ssl-fraction',
        default='',
        help='Fraction token stored in metrics.json (5, 10, 20, or full). Does not select the split by itself.',
    )
    parser.add_argument(
        '--ssl-out-dir',
        default='',
        help='If set, write metrics.json (Dice, mIoU, HD95, ASSD) for scripts/aggregate_media_results.py.',
    )
    parser.add_argument(
        '--ssl-eval-npy',
        default='',
        help=(
            'Optional held-out image .npy used only for eval. data_val.npy and data_test.npy are allowed here. '
            'They are still refused as unlabeled training data. No download.'
        ),
    )
    parser.add_argument(
        '--ssl-eval-mask-npy',
        default='',
        help='Mask .npy paired with --ssl-eval-npy. Eval only.',
    )
    return parser


def parse_train_args(argv=None):
    """Return (args, unknown, reserved_names).

    `reserved_names` lists ablation flags that were actually passed.
    `unknown` is ignored so extra tokens do not change the supervised run.
    """
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    reserved = [flag for attr, flag, _reason in RESERVED_FLAGS if getattr(args, attr)]
    return args, unknown, reserved


def resolve_ssl_method(args):
    """Apply a baseline preset.

    ``qam`` leaves the flags alone. The other methods force a shared-student
    baseline so a gate or MQE reweight cannot leak into MT / UA-MT / PL /
    supervised. Hybrids use ``--ssl-method qam`` plus the low-level flags.
    """
    import copy

    method = str(args.ssl_method)
    if method == 'qam':
        return args
    resolved = copy.copy(args)
    resolved.ssl_gate = 'ungated'
    resolved.quality_reweight = 'off'
    resolved.ssl_baq = False
    resolved.ssl_uncertainty = 'off'
    resolved.ssl_pl_threshold = 0.0
    if method == 'mt':
        return resolved
    if method == 'uamt':
        resolved.ssl_uncertainty = 'entropy'
        resolved.ssl_uncertainty_tau = float(args.ssl_uncertainty_tau)
        return resolved
    if method == 'pl':
        threshold = float(args.ssl_pl_threshold)
        resolved.ssl_pl_threshold = 0.95 if threshold <= 0.0 else threshold
        return resolved
    if method == 'supervised':
        resolved.ssl_lambda_max = 0.0
        return resolved
    raise ValueError(f'Unknown SSL method: {method}')


def refuse_reserved_flags(reserved):
    """Exit 2 when an ablation flag was passed. No-op otherwise."""
    if not reserved:
        return
    reasons = {flag: reason for _attr, flag, reason in RESERVED_FLAGS}
    print('QAM-Seg: ablation flags are scaffolds only and are not wired into training.')
    print('The supervised MambaLiteUNet path is unchanged if you omit them.')
    print('--ssl is separate and does not enable these bypasses.')
    print('See docs/QAM_SEG_SSL.md.')
    for flag in reserved:
        print(f'  {flag}: {reasons[flag]}')
    print('Refusing to start. Requested: ' + ' '.join(reserved))
    raise SystemExit(2)
