import pandas as pd,numpy as np
from finguard.inference.scorer import score
class Dummy:
    def predict_proba(self,x): return np.tile([.1,.9],(len(x),1))
def test_score_category():
    d=pd.DataFrame([{"transaction_timestamp":"2025-01-01","amount":2,"distance_from_home_km":1,"transactions_last_24h":1,"currency":"INR","merchant_category":"food","payment_method":"card"}])
    r=score(d,Dummy(),{"threshold":.5,"model_version":"test"},"b",.8); assert r.iloc[0].risk_category=="high"
def test_empty():
    d=pd.DataFrame(columns=["transaction_timestamp","amount","distance_from_home_km","transactions_last_24h","currency","merchant_category","payment_method"])
    assert score(d,Dummy(),{"threshold":.5,"model_version":"v"},"b").empty
