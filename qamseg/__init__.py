"""QAM-Seg additions on top of the MambaLiteUNet supervised backbone.

MQE, QGC, MQE loss reweighting, labeled-fraction splits, and an optional
SSL backbone hook live in this package. ``--ssl-backbone conv`` selects
``models.ConvLiteUNet``, a CNN stand-in that is not Mamba and not a quality
predictor. They are covered by synthetic unit tests. They are not
experimentally validated. See docs/QAM_SEG_SSL.md and docs/MEDIA_NOTES.md.
There is no separate quality-predictor network.
"""
