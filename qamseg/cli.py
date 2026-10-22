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
    parser.add_argument('--ssl-gate', default='gated', choices=('hard', 'soft', 'gated'))
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
        choices=('tiny', 'mamba'),
        help=(
            'SSL student. tiny (default) is the CPU conv stand-in. '
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
