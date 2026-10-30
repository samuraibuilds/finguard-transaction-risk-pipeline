SELECT date(transaction_timestamp) day,COUNT(*) transactions,SUM(amount) total_amount FROM transactions GROUP BY day;
SELECT merchant_category,COUNT(*) transactions,AVG(is_fraud) observed_fraud_rate FROM transactions GROUP BY merchant_category;
SELECT customer_id,COUNT(*) transactions,SUM(amount) total_amount FROM transactions GROUP BY customer_id;
