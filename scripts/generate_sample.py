from pathlib import Path
import numpy as np,pandas as pd
rng=np.random.default_rng(20261030); n=2400
ts=pd.date_range("2025-01-01",periods=n,freq="30min",tz="UTC"); amount=rng.lognormal(5,1,n)
cat=rng.choice(["grocery","travel","electronics","fuel","food","health","retail","utilities"],n)
hour=ts.hour.to_numpy(); distance=rng.gamma(2,8,n); velocity=rng.poisson(2.2,n)
logit=-4+.42*(amount>500)+.55*(distance>35)+.35*(velocity>5)+.28*np.isin(hour,[0,1,2,3,4])+.25*np.isin(cat,["electronics","travel"])
fraud=rng.binomial(1,1/(1+np.exp(-logit)))
d=pd.DataFrame({"transaction_id":[f"TX{i:06}" for i in range(n)],"customer_id":[f"C{v:05}" for v in rng.integers(1,401,n)],"transaction_timestamp":ts.astype(str),"amount":amount.round(2),"currency":rng.choice(["INR","USD","EUR"],n,p=[.85,.1,.05]),"merchant_category":cat,"payment_method":rng.choice(["card","upi","bank_transfer","wallet"],n),"distance_from_home_km":distance.round(2),"transactions_last_24h":velocity,"is_fraud":fraud})
d.loc[5,"amount"]=-12; d.loc[11,"merchant_category"]="unclassified"; d.loc[19,"transaction_timestamp"]="bad-time"; d.loc[23,"transaction_id"]=d.loc[22,"transaction_id"]; d.loc[27,"customer_id"]=None
p=Path("data/raw/sample/transactions.csv"); p.parent.mkdir(parents=True,exist_ok=True); d.to_csv(p,index=False); print(f"Generated {n} synthetic rows, seed=20261030")
