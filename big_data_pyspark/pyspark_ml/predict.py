from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from pyspark.ml import Pipeline, PipelineModel
import os
import sys

model_path = sys.argv[1]
test_path = sys.argv[2]
predict_path = sys.argv[3]

test = spark.read.json(test_path)
test = data.select('reviewText')

model = PipelineModel.load(model_path)

predictions = model.transform(test)

predictions.select("prediction").write.mode("overwrite").text(predict_path)
