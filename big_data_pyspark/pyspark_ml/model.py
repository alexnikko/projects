from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml import Pipeline

tokenizer = Tokenizer(inputCol='reviewText', outputCol='reviewWords')
stop_words_remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol='reviewWordsWithoutTrash')
vectorizer = CountVectorizer(inputCol=stop_words_remover.getOutputCol(), outputCol="word_vector", minDF=150)
lr = LinearRegression(featuresCol=vectorizer.getOutputCol(), labelCol='overall')

pipeline = Pipeline(stages=[tokenizer, stop_words_remover, vectorizer, lr])
