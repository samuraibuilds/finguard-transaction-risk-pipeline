import pandas as pd
from finguard.validation.validator import validate
R={"supported_currencies":["INR"],"allowed_categories":["grocery"],"allowed_payment_methods":["card"],"min_amount":.01,"max_amount":10000}
def row(): return {"transaction_id":"t1","customer_id":"c1","transaction_timestamp":"2025-01-01T00:00:00Z","amount":10,"currency":"INR","merchant_category":"grocery","payment_method":"card","distance_from_home_km":2,"transactions_last_24h":1,"is_fraud":0}
def test_valid_row():
    g,b,s=validate(pd.DataFrame([row()]),R); assert len(g)==1 and len(b)==0
def test_invalid_amount_quarantined():
    x=row(); x["amount"]=-1; g,b,_=validate(pd.DataFrame([x]),R); assert g.empty and "invalid_amount" in b.iloc[0].rejection_reasons
def test_duplicate_quarantined():
    g,b,s=validate(pd.DataFrame([row(),row()]),R); assert len(g)==1 and len(b)==1 and s["duplicate_count"]==1
