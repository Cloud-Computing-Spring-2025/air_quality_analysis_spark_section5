import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# 1. Start Spark Session
spark = SparkSession.builder \
    .appName("AirQualityML_CSV") \
    .getOrCreate()

# 2. Load the cleaned dataset
df = spark.read.option("header", "true").option("inferSchema", "true").csv("Output/output_section3_csv/*.csv")

# 3. Confirm available columns
print("Available columns:", df.columns)

# 4. Handle missing values using Imputer (mean replacement)
imputer = Imputer(
    inputCols=['temperature', 'humidity', 'lag1_pm25', 'rolling_avg_pm25', 'rate_of_change_pm25'],
    outputCols=['temperature', 'humidity', 'lag1_pm25', 'rolling_avg_pm25', 'rate_of_change_pm25']
)
df = imputer.fit(df).transform(df)

# 5. Assemble feature vector
feature_cols = ['temperature', 'humidity', 'lag1_pm25', 'rolling_avg_pm25', 'rate_of_change_pm25']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# 6. Train/Test split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# 7. Train Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="avg_pm25")
lr_model = lr.fit(train_data)

# 8. Make predictions
predictions = lr_model.transform(test_data)

# 9. Evaluate model
evaluator_rmse = RegressionEvaluator(labelCol="avg_pm25", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="avg_pm25", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print(f"Initial RMSE: {rmse}")
print(f"Initial R²: {r2}")

# 10. Hyperparameter tuning with cross-validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.1, 0.5])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .build())

crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator_rmse,
                          numFolds=3)

cv_model = crossval.fit(train_data)
best_model = cv_model.bestModel
predictions_best = best_model.transform(test_data)

# 11. Evaluate best model
best_rmse = evaluator_rmse.evaluate(predictions_best)
best_r2 = evaluator_r2.evaluate(predictions_best)

print(f"[Tuned] Best RMSE: {best_rmse}")
print(f"[Tuned] Best R²: {best_r2}")

# 12. Save the best model
best_model.save("Output/section4_model/air_quality_lr_model")

# 13. Save predictions for dashboard use (Section 5)
predictions_best.select("timestamp", "location", "avg_pm25", "prediction").write.mode("overwrite").csv("Output/section4_predictions/")

# 14. Real-time prediction plan (placeholder)
'''
Real-Time Prediction Plan:
- Ingest new records from the TCP stream (Section 1).
- Apply the same VectorAssembler on the streaming DataFrame.
- Load the trained model and apply .transform() to predict PM2.5.
- Append predictions to a sink (console, Kafka, database).
'''

# End
spark.stop()
