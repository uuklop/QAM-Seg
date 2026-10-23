"""Why QAM-Seg does not add a QG-SSL quality-aware regularizer.

QG-SSL's QAR is ``mean(1 - g_φ(x, f_θ(x)))`` with gradients through a
frozen external quality network into the predicted mask. QAM-Seg has no
``g_φ``. MQE is a stop-grad function of entropy, disagreement, view
agreement, and boundary entropy.

Pushing that function up with gradients would:

- minimize predictive entropy (a confidence penalty, which is the UA-MT
  baseline, not a new quality model);
- pull the student toward the teacher and toward the other view
  (consistency, which is already QGC);
- sharpen the contour band while BAQ is trying to down-weight
  high-gradient, low-quality pixels.

Those terms duplicate QGC / the UA-MT baseline or fight BAQ. They are
not implemented. ``quality_aware_regularization`` raises if called.
"""

from __future__ import annotations

QAR_STATUS = "skipped"
QAR_REASON = (
    "QAM-Seg does not add a QAR loss. Without a separate g_phi, "
    "differentiating through MQE duplicates QGC and the entropy-weighted "
    "UA-MT baseline, and it conflicts with BAQ. "
    "See docs/MEDIA_NOTES.md."
)


def quality_aware_regularization(*_args, **_kwargs):
    """Refuse the QG-SSL QAR objective. No quality network is built."""
    raise RuntimeError(QAR_REASON)
