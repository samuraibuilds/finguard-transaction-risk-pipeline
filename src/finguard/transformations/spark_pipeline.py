from pathlib import Path
import shutil

from pyspark.sql import SparkSession, functions as F


def run_spark(csv_path, out_dir, app="FinGuardBatch", partitions=2):

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(app)
        .config("spark.sql.shuffle.partitions", str(partitions))
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )

    try:
        # Read validated input data
        d = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(csv_path)
        )

        # Feature transformations
        d = (
            d.withColumn(
                "event_ts",
                F.to_timestamp("transaction_timestamp")
            )
            .withColumn(
                "transaction_date",
                F.to_date("event_ts")
            )
            .withColumn(
                "transaction_hour",
                F.hour("event_ts")
            )
            .withColumn(
                "amount_log1p",
                F.log1p("amount")
            )
            .withColumn(
                "is_weekend",
                F.dayofweek("event_ts").isin([1, 7]).cast("int")
            )
        )

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Remove only previous Spark output folders.
        # Never delete the parent directory containing the input CSV.
        for folder in (
            "transactions",
            "merchant_summary",
            "customer_summary",
        ):
            target = out / folder

            if target.exists():
                shutil.rmtree(target)

        # Write transaction-level Parquet data
        d.coalesce(1).write.mode("overwrite").parquet(
            str(out / "transactions")
        )

        # Merchant-level aggregation
        (
            d.groupBy("merchant_category")
            .agg(
                F.count("*").alias("count"),
                F.sum("amount").alias("amount_sum"),
                F.avg("is_fraud").alias("observed_fraud_rate"),
            )
            .coalesce(1)
            .write.mode("overwrite")
            .parquet(str(out / "merchant_summary"))
        )

        # Customer-level aggregation
        (
            d.groupBy("customer_id")
            .agg(
                F.count("*").alias("count"),
                F.sum("amount").alias("amount_sum"),
            )
            .coalesce(1)
            .write.mode("overwrite")
            .parquet(str(out / "customer_summary"))
        )

        return {
            "spark_rows": d.count(),
            "transactions": str(out / "transactions"),
        }

    finally:
        spark.stop()