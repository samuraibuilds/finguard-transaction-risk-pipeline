import pandas as pd
from finguard.features.build import features
def test_target_not_feature():
    d=pd.DataFrame([{"transaction_timestamp":"2025-01-01T05:00Z","amount":3,"distance_from_home_km":1,"transactions_last_24h":2,"currency":"INR","merchant_category":"food","payment_method":"upi","is_fraud":1}])
    x,y=features(d); assert "is_fraud" not in x and y.iloc[0]==1 and x.iloc[0].transaction_hour==5
