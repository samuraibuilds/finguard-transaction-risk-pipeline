from contextlib import asynccontextmanager
import uuid,pandas as pd
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from finguard.common import config
from finguard.inference.scorer import load_model,score
class Transaction(BaseModel):
    transaction_id:str=Field(min_length=1); customer_id:str=Field(min_length=1); transaction_timestamp:str
    amount:float=Field(gt=0,le=1000000); currency:str; merchant_category:str; payment_method:str
    distance_from_home_km:float=Field(ge=0); transactions_last_24h:int=Field(ge=0)
class Request(BaseModel): transaction:Transaction
@asynccontextmanager
async def lifespan(app):
    c=config(); app.state.cfg=c; app.state.model=None; app.state.meta=None
    try:
        m,meta=load_model(c["paths"]["model_dir"],c["model"]["version"])
        app.state.model=m; app.state.meta=meta
    except FileNotFoundError:
        # Keep health/readiness diagnostics available; inference returns HTTP 503 until trained.
        pass
    yield
app=FastAPI(title="FinGuard Risk API",version="1.0.0",lifespan=lifespan)
@app.get("/health")
def health():
    ready = getattr(app.state,"model",None) is not None
    return {"status":"ok" if ready else "degraded","model_loaded":ready}
@app.get("/model-info")
def info():
    if app.state.meta is None: raise HTTPException(503,"Model artifacts unavailable; run the training pipeline")
    return {k:app.state.meta[k] for k in ["model_version","model_name","threshold"]}
@app.post("/predict")
def predict(req:Request):
    if app.state.model is None: raise HTTPException(503,"Model artifacts unavailable; run the training pipeline")
    row=req.transaction.model_dump()
    try:
        result=score(pd.DataFrame([row]),app.state.model,app.state.meta,"api-"+uuid.uuid4().hex[:8],app.state.cfg["model"]["high_risk_threshold"]).iloc[0]
        return {"request_id":row["transaction_id"],"risk_score":float(result.risk_score),"risk_category":str(result.risk_category),"model_version":str(result.model_version),"threshold":float(result.threshold),"message":"Risk score is not proof of fraud."}
    except Exception as e: raise HTTPException(422,f"Scoring failed: {e}")
