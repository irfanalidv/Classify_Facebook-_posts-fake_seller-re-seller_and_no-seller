# Databricks notebook source
#load data as dataframe
data = sqlContext.read.load('/FileStore/tables/Facebook_Post_Classifier/FB_User_Classification.csv', format='csv', header='true', inferSchema='true',sep='\t')
data.show()

# COMMAND ----------

data.printSchema()

# COMMAND ----------

data = data.select(['description','INDEX New'])
data.show()

# COMMAND ----------

data = data.withColumnRenamed('description','text').withColumnRenamed('INDEX New','class')
data.show()

# COMMAND ----------

data.printSchema()

# COMMAND ----------

#Cleaning and Preparing the Data
#Creating a new length feature:
from pyspark.sql.functions import length
data = data.withColumn('length',length(data['text']))
data.show()

# COMMAND ----------

data=data.na.drop(how = 'any')
data.printSchema()

# COMMAND ----------

data.groupby('class').mean().show()
# Pretty Clear Difference

# COMMAND ----------

#Feature Transformations
from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer

tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
idf = IDF(inputCol="c_vec", outputCol="tf_idf")
ham_spam_to_num = StringIndexer(inputCol='class',outputCol='label')

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector

clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')

# COMMAND ----------

#The Model NaiveBayes
from pyspark.ml.classification import NaiveBayes
# Use defaults
nb = NaiveBayes()

# COMMAND ----------

#Data Pipeline
from pyspark.ml import Pipeline
data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,count_vec,idf,clean_up])
cleaner = data_prep_pipe.fit(data)
clean_data = cleaner.transform(data)


# COMMAND ----------

#Training and Evaluation of model
clean_data = clean_data.select(['label','features'])
clean_data.show()


# COMMAND ----------

#splitting the data set in Test and Train
(training,testing) = clean_data.randomSplit([0.7,0.3])
spam_predictor = nb.fit(training)


# COMMAND ----------

test_results = spam_predictor.transform(testing)
test_results.show()


# COMMAND ----------

#Classification Evaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print("Accuracy of model at predicting the Post was: {}".format(acc))
