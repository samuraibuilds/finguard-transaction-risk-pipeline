import logging,yaml
from pathlib import Path
def config(path="config/config.yaml"):
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f"Config not found: {p}")
    c=yaml.safe_load(p.read_text())
    t=float(c["model"]["threshold"]); h=float(c["model"]["high_risk_threshold"])
    if not (0<t<1 and 0<h<1 and h>=t): raise ValueError("Invalid risk thresholds")
    if not 0<=float(c["validation"]["max_rejection_rate"])<=1: raise ValueError("Invalid rejection limit")
    return c
def logging_setup():
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s %(message)s",
      handlers=[logging.StreamHandler(),logging.FileHandler("logs/finguard.log",encoding="utf8")],force=True)
