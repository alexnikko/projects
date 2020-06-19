from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from model import pipeline
import os
import sys

path_to_data = sys.argv[1]
model_path = sys.argv[2]

# Считываем трейн
data = spark.read.json(path_to_data)
data = data.select('reviewText', 'overall')

# Обучаем модель
pipeline_model = pipeline.fit(data)

# Сохраняем модель
pipeline_model.write().overwrite().save(model_path)
