
from pathlib import Path
from datetime import datetime, timezone
import json
import os

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Default monitoring thresholds.
# These are initial project defaults, not universal fraud-industry standards.
DEFAULT_THRESHOLDS = {
    "max_rejection_rate": 0.05,
    "max_high_risk_rate": 0.20,
    "max_missing_cells": 0,
    "max_duplicate_transactions": 0,
}


def _safe_float(value):
    """Convert numeric values into JSON-safe floats."""
    if pd.isna(value):
        return None

    return float(value)


def _save_json_atomic(data, output_path):
    """Write JSON safely using a temporary file and atomic replacement."""
    output_path = Path(output_path)
    temp_path = output_path.with_name(
        f".{output_path.stem}.tmp"
    )

    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, output_path)

    finally:
        if temp_path.exists():
            temp_path.unlink()


def _save_histogram(series, title, output_path):
    """Save a histogram without displaying a GUI."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return

    figure, axis = plt.subplots(figsize=(9, 5))

    try:
        axis.hist(values, bins=25)
        axis.set_title(title)
        axis.set_xlabel("Value")
        axis.set_ylabel("Frequency")

        figure.tight_layout()
        figure.savefig(output_path, dpi=120)

    finally:
        plt.close(figure)


def monitor(
    df,
    rejected,
    scored,
    folder,
    thresholds=None,
):
    """
    Generate data quality and model monitoring reports.

    Parameters
    ----------
    df : pandas.DataFrame
        Accepted and validated transactions.
    rejected : pandas.DataFrame
        Rejected transactions.
    scored : pandas.DataFrame
        Transactions with model predictions.
    folder : str or Path
        Monitoring output directory.
    thresholds : dict, optional
        Custom monitoring thresholds.

    Returns
    -------
    dict
        Monitoring summary with metrics, alerts and status.
    """

    # Validate inputs.
    for name, frame in [
        ("df", df),
        ("rejected", rejected),
        ("scored", scored),
    ]:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")

    required_scoring_columns = {
        "risk_category",
        "risk_score",
    }

    missing_columns = (
        required_scoring_columns - set(scored.columns)
    )

    if missing_columns:
        raise ValueError(
            "Scoring output missing monitoring columns: "
            f"{sorted(missing_columns)}"
        )

    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)

    limits = {
        **DEFAULT_THRESHOLDS,
        **(thresholds or {}),
    }

    # --------------------------------------------------
    # 1. Data quality metrics
    # --------------------------------------------------

    accepted_count = len(df)
    rejected_count = len(rejected)
    total_count = accepted_count + rejected_count

    rejection_rate = (
        rejected_count / total_count
        if total_count
        else 0.0
    )

    missing_cells = int(
        df.isna().sum().sum()
        + rejected.isna().sum().sum()
    )

    duplicate_count = 0

    if "transaction_id" in df.columns:
        duplicate_count = int(
            df["transaction_id"].duplicated().sum()
        )

    # --------------------------------------------------
    # 2. Fraud and transaction metrics
    # --------------------------------------------------

    fraud_prevalence = None

    if "is_fraud" in df.columns and accepted_count:
        fraud_values = pd.to_numeric(
            df["is_fraud"],
            errors="coerce",
        ).dropna()

        if not fraud_values.empty:
            fraud_prevalence = _safe_float(
                fraud_values.mean()
            )

    amount_mean = None
    amount_median = None

    if "amount" in df.columns and accepted_count:
        amounts = pd.to_numeric(
            df["amount"],
            errors="coerce",
        ).dropna()

        if not amounts.empty:
            amount_mean = _safe_float(amounts.mean())
            amount_median = _safe_float(amounts.median())

    # --------------------------------------------------
    # 3. Model monitoring metrics
    # --------------------------------------------------

    scored_count = len(scored)

    categories = (
        scored["risk_category"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    high_risk_count = int(
        categories.eq("high").sum()
    )

    high_risk_rate = (
        high_risk_count / scored_count
        if scored_count
        else None
    )

    category_distribution = {
        str(key): int(value)
        for key, value in categories.value_counts(
            dropna=False
        ).items()
    }

    risk_scores = pd.to_numeric(
        scored["risk_score"],
        errors="coerce",
    ).dropna()

    risk_score_metrics = {
        "valid_score_count": int(len(risk_scores)),
        "missing_score_count": int(
            scored_count - len(risk_scores)
        ),
        "mean": (
            _safe_float(risk_scores.mean())
            if not risk_scores.empty
            else None
        ),
        "median": (
            _safe_float(risk_scores.median())
            if not risk_scores.empty
            else None
        ),
        "min": (
            _safe_float(risk_scores.min())
            if not risk_scores.empty
            else None
        ),
        "max": (
            _safe_float(risk_scores.max())
            if not risk_scores.empty
            else None
        ),
        "p95": (
            _safe_float(risk_scores.quantile(0.95))
            if not risk_scores.empty
            else None
        ),
    }

    # --------------------------------------------------
    # 4. Generate monitoring alerts
    # --------------------------------------------------

    alerts = []

    if rejection_rate > limits["max_rejection_rate"]:
        alerts.append({
            "severity": "warning",
            "metric": "rejection_rate",
            "observed": rejection_rate,
            "threshold": limits["max_rejection_rate"],
            "message": "Transaction rejection rate exceeds threshold.",
        })

    if (
        high_risk_rate is not None
        and high_risk_rate > limits["max_high_risk_rate"]
    ):
        alerts.append({
            "severity": "warning",
            "metric": "high_risk_rate",
            "observed": high_risk_rate,
            "threshold": limits["max_high_risk_rate"],
            "message": "High-risk prediction rate exceeds threshold.",
        })

    if missing_cells > limits["max_missing_cells"]:
        alerts.append({
            "severity": "warning",
            "metric": "missing_cells",
            "observed": missing_cells,
            "threshold": limits["max_missing_cells"],
            "message": "Missing values detected in transaction data.",
        })

    if duplicate_count > limits["max_duplicate_transactions"]:
        alerts.append({
            "severity": "warning",
            "metric": "duplicate_transactions",
            "observed": duplicate_count,
            "threshold": limits["max_duplicate_transactions"],
            "message": "Duplicate transaction IDs detected.",
        })

    if risk_score_metrics["missing_score_count"] > 0:
        alerts.append({
            "severity": "warning",
            "metric": "missing_risk_scores",
            "observed": risk_score_metrics["missing_score_count"],
            "threshold": 0,
            "message": "Some scored transactions have invalid or missing risk scores.",
        })

    if not scored.empty and risk_scores.empty:
        alerts.append({
            "severity": "warning",
            "metric": "invalid_risk_scores",
            "observed": scored_count,
            "threshold": 0,
            "message": "No valid numeric risk scores were produced.",
        })

    # --------------------------------------------------
    # 5. Build monitoring report
    # --------------------------------------------------

    summary = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "monitoring_status": (
            "warning" if alerts else "healthy"
        ),

        "batch_metrics": {
            "accepted_rows": accepted_count,
            "rejected_rows": rejected_count,
            "total_input_rows": total_count,
            "scored_rows": scored_count,
        },

        "data_quality": {
            "rejection_rate": float(rejection_rate),
            "missing_cells": missing_cells,
            "duplicate_transactions": duplicate_count,
        },

        "transaction_metrics": {
            "fraud_prevalence": fraud_prevalence,
            "mean_amount": amount_mean,
            "median_amount": amount_median,
        },

        "model_metrics": {
            "high_risk_count": high_risk_count,
            "high_risk_rate": (
                float(high_risk_rate)
                if high_risk_rate is not None
                else None
            ),
            "risk_category_distribution": category_distribution,
            "risk_score": risk_score_metrics,
        },

        "thresholds": limits,
        "alerts": alerts,
        "alert_count": len(alerts),
    }

    # --------------------------------------------------
    # 6. Save machine-readable monitoring report
    # --------------------------------------------------

    _save_json_atomic(
        summary,
        p / "monitoring_summary.json",
    )

    # --------------------------------------------------
    # 7. Generate monitoring charts
    # --------------------------------------------------

    if "amount" in df.columns:
        _save_histogram(
            df["amount"],
            "Transaction Amount Distribution",
            p / "amount_distribution.png",
        )

    _save_histogram(
        scored["risk_score"],
        "Model Prediction Score Distribution",
        p / "risk_scores.png",
    )

    return summary
