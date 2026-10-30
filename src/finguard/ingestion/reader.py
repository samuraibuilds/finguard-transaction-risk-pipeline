import pandas as pd
REQUIRED=["transaction_id","customer_id","transaction_timestamp","amount","currency","merchant_category","payment_method","distance_from_home_km","transactions_last_24h","is_fraud"]
def read_csv(path):
    from pathlib import Path
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(f"Input CSV missing: {p}")
    d=pd.read_csv(p)
    missing=set(REQUIRED)-set(d.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    return d[REQUIRED].copy()
