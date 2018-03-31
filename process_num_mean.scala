val df = spark.read.format("csv").
        option("header", "true").
        option("inferSchema", "true").
        load("/opt/shared-data/kkbox-churn-prediction-challenge/user_logs.csv")

val result = df.groupBy("msno").
    agg(mean("num_25"),
        mean("num_50"),
        mean("num_75"),
        mean("num_985"),
        mean("num_100"),
        mean("num_unq"),
        mean("total_secs"),
        sum("num_25"),
        sum("num_50"),
        sum("num_75"),
        sum("num_985"),
        sum("num_100"),
        sum("num_unq"),
        sum("total_secs")
    )


result.repartition(1).write.
format("com.databricks.spark.csv").
option("header", "true").save("num_mean.csv")

// result.printSchema()