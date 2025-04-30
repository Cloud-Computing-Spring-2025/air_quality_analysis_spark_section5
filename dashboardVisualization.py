from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Start Spark session
spark = SparkSession.builder.appName("AirQualityEvaluation").getOrCreate()

# Load CSV WITHOUT header and manually assign column names
df = spark.read.option("header", "false").option("inferSchema", "true").csv("Output/section4_predictions/*.csv")
df = df.toDF("timestamp", "location", "actual", "predicted")

# Show schema to confirm
print("Renamed Schema:", df.columns)

# Convert to Pandas
pdf = df.select("actual", "predicted").dropna().toPandas()

# Evaluation metrics
rmse = mean_squared_error(pdf['actual'], pdf['predicted'], squared=False)
mae = mean_absolute_error(pdf['actual'], pdf['predicted'])
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Save metrics
with open("Output/section5_metrics.txt", "w") as f:
    f.write(f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(pdf['actual'].values, label='Actual PM2.5')
plt.plot(pdf['predicted'].values, label='Predicted PM2.5')
plt.xlabel("Index")
plt.ylabel("PM2.5")
plt.title("Actual vs Predicted PM2.5 Levels")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Output/section5_visual_comparison.png")
plt.show()