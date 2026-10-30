from pathlib import Path
import json,joblib,pandas as pd
from finguard.features.build import features
def load_model(folder,version):
    p=Path(folder); m=p/f"{version}.joblib"; j=p/f"{version}.json"
    if not m.exists() or not j.exists(): raise FileNotFoundError(f"Model {version} missing; run training first")
    return joblib.load(m),json.loads(j.read_text())
def score(df,model,meta,batch,high=.7):
    if df.empty: return df.assign(risk_score=[],risk_category=[],model_version=[],batch_id=[],threshold=[])
    X,_=features(df); s=model.predict_proba(X)[:,1]; t=float(meta["threshold"])
    out=df.copy(); out["risk_score"]=s; out["risk_category"]=["high" if v>=high else "medium" if v>=t else "low" for v in s]
    out["model_version"]=meta["model_version"]; out["batch_id"]=batch; out["threshold"]=t
    return out
