
import pandas as pd

from finguard.ingestion.reader import REQUIRED


def validate(df, rules):
    # 1. Validate the input structure
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"Duplicate column names detected: {duplicates}"
        )

    missing = sorted(set(REQUIRED) - set(df.columns))

    if missing:
        raise ValueError(
            f"Data contract violation. Missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError("Input batch is empty")

    # 2. Preserve existing validation logic
    x = df[REQUIRED].copy()
    why = pd.Series("", index=x.index, dtype="object")

    def bad(mask, reason):
        nonlocal why
        mask = mask.fillna(True)
        why.loc[mask] = why.loc[mask].apply(
            lambda v: (v + ";" if v else "") + reason
        )

    # 3. Transaction ID validation
    dup = (
        x.transaction_id.isna()
        | x.transaction_id.duplicated()
    )
    bad(dup, "duplicate_or_missing_transaction_id")

    # 4. Timestamp validation
    x["transaction_timestamp"] = pd.to_datetime(
        x.transaction_timestamp,
        errors="coerce",
        utc=True,
    )
    bad(
        x.transaction_timestamp.isna(),
        "invalid_timestamp",
    )

    # 5. Amount validation
    x["amount"] = pd.to_numeric(
        x.amount,
        errors="coerce",
    )
    bad(
        x.amount.isna()
        | (x.amount < float(rules["min_amount"]))
        | (x.amount > float(rules["max_amount"])),
        "invalid_amount",
    )

    # 6. Categorical validation
    for col, allowed in [
        ("currency", rules["supported_currencies"]),
        ("merchant_category", rules["allowed_categories"]),
        ("payment_method", rules["allowed_payment_methods"]),
    ]:
        bad(
            x[col].isna()
            | ~x[col].astype(str).isin(allowed),
            "invalid_" + col,
        )

    # 7. Required field validation
    for col in [
        "customer_id",
        "distance_from_home_km",
        "transactions_last_24h",
        "is_fraud",
    ]:
        bad(x[col].isna(), "missing_" + col)

    # 8. Numeric feature validation
    x["distance_from_home_km"] = pd.to_numeric(
        x.distance_from_home_km,
        errors="coerce",
    )

    x["transactions_last_24h"] = pd.to_numeric(
        x.transactions_last_24h,
        errors="coerce",
    )

    x["is_fraud"] = pd.to_numeric(
        x.is_fraud,
        errors="coerce",
    )

    bad(
        x.distance_from_home_km.isna()
        | (x.distance_from_home_km < 0),
        "invalid_distance",
    )

    bad(
        x.transactions_last_24h.isna()
        | (x.transactions_last_24h < 0),
        "invalid_velocity",
    )

    bad(
        ~x.is_fraud.isin([0, 1]),
        "invalid_target",
    )

    # 9. Separate accepted and rejected records
    mask = why.ne("")

    good = x.loc[~mask].copy()
    rejected = x.loc[mask].copy()
    rejected["rejection_reasons"] = why[mask]

    summary = {
        "input_count": len(x),
        "accepted_count": len(good),
        "rejected_count": len(rejected),
        "duplicate_count": int(dup.sum()),
        "rejection_rate": float(mask.mean()),
    }

    return good, rejected, summary
