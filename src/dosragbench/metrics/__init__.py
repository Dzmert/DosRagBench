"""Evaluation metrics for DoSRAGBench."""

from dosragbench.metrics.metrics import (
    MetricsReport,
    QueryResult,
    compute_avi,
    compute_metrics,
)
from dosragbench.metrics.refusal import (
    RefusalType,
    SeverityLevel,
    classify_refusal,
    classify_severity,
)
from dosragbench.metrics.answer import answer_contains_gold, normalize_answer

__all__ = [
    "MetricsReport",
    "QueryResult",
    "compute_avi",
    "compute_metrics",
    "RefusalType",
    "SeverityLevel",
    "classify_refusal",
    "classify_severity",
    "answer_contains_gold",
    "normalize_answer",
]
