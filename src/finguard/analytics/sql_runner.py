import sqlite3
from pathlib import Path
import pandas as pd
QUERIES={
"daily":"SELECT date(transaction_timestamp) day,COUNT(*) transactions,SUM(amount) total_amount FROM transactions GROUP BY day ORDER BY day",
"merchant":"SELECT merchant_category,COUNT(*) transactions,SUM(amount) total_amount,AVG(is_fraud) observed_fraud_rate FROM transactions GROUP BY merchant_category",
"customer":"SELECT customer_id,COUNT(*) transactions,SUM(amount) total_amount,AVG(amount) avg_amount FROM transactions GROUP BY customer_id ORDER BY transactions DESC LIMIT 20",
"hourly":"SELECT CAST(strftime('%H',transaction_timestamp) AS INTEGER) hour,COUNT(*) transactions FROM transactions GROUP BY hour ORDER BY hour",
"high_value":"SELECT * FROM transactions WHERE amount >= (SELECT MAX(amount)*.9 FROM transactions) ORDER BY amount DESC",
"fraud_by_channel":"SELECT merchant_category,payment_method,COUNT(*) transactions,AVG(is_fraud) observed_fraud_rate FROM transactions GROUP BY merchant_category,payment_method"}
def run(df,folder):
    p=Path(folder); p.mkdir(parents=True,exist_ok=True); result={}
    with sqlite3.connect(":memory:") as c:
        df.to_sql("transactions",c,index=False)
        for name,q in QUERIES.items():
            result[name]=pd.read_sql_query(q,c); result[name].to_csv(p/f"{name}.csv",index=False)
    return result
