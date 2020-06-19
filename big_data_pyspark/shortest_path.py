from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')


from pyspark.sql.functions import *
from pyspark.sql.types import *
import os
import sys

def shortest_path(v_from, v_to, df, max_path_length=10):
    """
        v_from - исходная вершина
        v_to - целевая вершина
        df - Spark DataFrame с ребрами графа
        max_path_length - максимальная длина пути

        Возвращает: pyspark.sql.DataFrame, состоящий из одного столбца с найдеными путями
    """

    base_df = df.where(df['fromm'] == v_from)
    # Если на первом шаге сразу есть ребро из старотовой вершины в искомую
    # Краевой случай
    if base_df.where(base_df.to == v_to).count() > 0:
        base_df = base_df.where(df['to'] == v_to)
        base_df = base_df.select(concat_ws(',', base_df['fromm'], base_df['to']).alias('path'))
        return base_df

    # Сформируем базовый датафрейм
    base_df = base_df.select(base_df['to'].alias('how'), concat_ws(',', base_df['fromm'], base_df['to']).alias('path'))

    # Будем итеративно джоинить его

    for i in range(max_path_length):
        base_df = base_df.join(df, base_df.how == df.fromm)

        # Если на следующем шаге будет добавлена искомая вершина, то остановимся
        if base_df.where(base_df.to == v_to).count() > 0:
            base_df = base_df.where(base_df.to == v_to)
            base_df = base_df.select(concat_ws(',', base_df['path'], base_df['to']).alias('path'))
            return base_df

        base_df = base_df.select(base_df['to'].alias('how'), concat_ws(',', base_df['path'], base_df['to']).alias('path'))



v_from = sys.argv[1]
v_to = sys.argv[2]
read_from = sys.argv[3]
write_to = sys.argv[4]


twitter_schema = StructType(fields=[
    StructField("user_id", IntegerType()),
    StructField("follower_id", IntegerType())
])

twitter = spark.read.csv(read_from, sep="\t", schema=twitter_schema).cache()
twitter = twitter.selectExpr('follower_id as fromm', 'user_id as to')

max_path_length = twitter.select('fromm').distinct().count()

ans = shortest_path(v_from, v_to, twitter, max_path_length=max_path_length)

ans.select("path").write.mode("overwrite").text(write_to)
