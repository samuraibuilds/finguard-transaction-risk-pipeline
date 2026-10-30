import sqlite3
from pathlib import Path
def run_record(path,r):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(p) as c:
        c.execute("CREATE TABLE IF NOT EXISTS pipeline_runs(run_id TEXT PRIMARY KEY,batch_id TEXT,started_at TEXT,finished_at TEXT,status TEXT,input_count INTEGER,accepted_count INTEGER,rejected_count INTEGER,error TEXT)")
        c.execute("INSERT OR REPLACE INTO pipeline_runs VALUES(?,?,?,?,?,?,?,?,?)",tuple(r.get(k) for k in ["run_id","batch_id","started_at","finished_at","status","input_count","accepted_count","rejected_count","error"]))
