import argparse,json
from pathlib import Path
from finguard.orchestration.pipeline import run_pipeline
from finguard.common import config
from finguard.ingestion.reader import read_csv
from finguard.validation.validator import validate
from finguard.inference.scorer import load_model,score
def main():
    a=argparse.ArgumentParser(); s=a.add_subparsers(dest="cmd",required=True)
    r=s.add_parser("run"); r.add_argument("--config",default="config/config.yaml"); r.add_argument("--no-train",action="store_true")
    q=s.add_parser("score"); q.add_argument("--config",default="config/config.yaml")
    x=a.parse_args()
    if x.cmd=="run": print(json.dumps(run_pipeline(x.config,not x.no_train),indent=2))
    else:
        c=config(x.config); d=read_csv(c["paths"]["input_csv"]); good,_,_=validate(d,c["validation"])
        m,meta=load_model(c["paths"]["model_dir"],c["model"]["version"]); scored=score(good,m,meta,c["project"]["batch_id"],c["model"]["high_risk_threshold"])
        o=Path(c["paths"]["scored_dir"]); o.mkdir(parents=True,exist_ok=True); scored.to_csv(o/"scored_batch.csv",index=False)
        print(f"Scored {len(scored)} rows")
if __name__=="__main__": main()
