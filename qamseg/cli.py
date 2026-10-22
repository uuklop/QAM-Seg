"""Reserved QAM-Seg flags for train.py.

The flags are parsed and then refused. They do not alter the supervised
MambaLiteUNet run. See docs/QAM_SEG_SSL.md.
"""

import argparse


RESERVED_FLAGS = (
    ('ssl', '--ssl', 'Quality-aware SSL is not implemented.'),
    ('no_amf', '--no-amf', 'AMF ablation is not wired.'),
    ('no_lgfm', '--no-lgfm', 'LGFM ablation is not wired.'),
    ('no_cga', '--no-cga', 'CGA ablation is not wired.'),
)


def build_parser():
    parser = argparse.ArgumentParser(
        prog='train.py',
        description=(
            'Supervised MambaLiteUNet training used as the QAM-Seg backbone. '
            'Reserved SSL and ablation flags are refused so they cannot start '
            'a half-implemented run.'
        ),
    )
    parser.add_argument(
        '--ssl',
        action='store_true',
        help='Reserved. Quality-aware SSL is not implemented. See docs/QAM_SEG_SSL.md.',
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
    return parser


def parse_train_args(argv=None):
    """Return (args, unknown, reserved_names).

    `reserved_names` lists the flag strings that were actually passed.
    `unknown` is ignored by train.py so extra tokens do not change the run.
    """
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    reserved = [flag for attr, flag, _reason in RESERVED_FLAGS if getattr(args, attr)]
    return args, unknown, reserved


def refuse_reserved_flags(reserved):
    """Exit 2 when a reserved flag was passed. No-op otherwise."""
    if not reserved:
        return
    reasons = {flag: reason for _attr, flag, reason in RESERVED_FLAGS}
    print('QAM-Seg: reserved flags are scaffolds only and are not wired into training.')
    print('The supervised MambaLiteUNet path is unchanged if you omit them.')
    print('See docs/QAM_SEG_SSL.md.')
    for flag in reserved:
        print(f'  {flag}: {reasons[flag]}')
    print('Refusing to start. Requested: ' + ' '.join(reserved))
    raise SystemExit(2)
