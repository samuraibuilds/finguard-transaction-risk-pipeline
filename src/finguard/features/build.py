import pandas as pd
NUM=["amount","transaction_hour","distance_from_home_km","transactions_last_24h"]
CAT=["currency","merchant_category","payment_method"]
def features(df):
    x=df.copy(); x["transaction_hour"]=pd.to_datetime(x.transaction_timestamp,utc=True,errors="coerce").dt.hour
    for c in NUM: x[c]=pd.to_numeric(x[c],errors="coerce")
    return x[NUM+CAT],x["is_fraud"].astype(int) if "is_fraud" in x else None
