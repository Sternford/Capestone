from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import Row
import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

conf = SparkConf()
conf.set("spark.master", "local[*]")
conf = conf.setAppName('Pysparksvm')
sc = SparkContext(conf=conf)
sql = SQLContext(sc)

#load dataset
rdd = sc.textFile(r".\data\vehicles_train.csv")

header = rdd.first()
rdd = rdd.filter(lambda line:line != header)

#convert spark dataframe to pandas dataframe
df = rdd.map(lambda line: line.split(',')).toDF()
myData = df.toPandas()
#print(myData)

#connverting categorical data int digits, label ecoder
print("data preprocessing...")
le = LabelEncoder()
for col in myData.columns.values:

    # Encoding only categorical variables
    if myData[col].dtypes == 'object':

        # Using whole data to form an exhaustive list of levels

        data = myData[col].append(myData[col])
        le.fit(data)
        myData[col] = le.transform(myData[col])

#droping the labels column
streamingData = myData.values[:, 0:13]
testingTargets = myData.values[:, 13]

print(testingTargets)

#classification
svm_linear_estimator = svm.SVC()

print("Classifying using saved model....")
estimator = joblib.load("svm_model/svm_model.pkl")
result = estimator.predict(streamingData)

acc = metrics.accuracy_score(result, testingTargets)
print("Accuracy of the model is: ", acc)

results = le.inverse_transform(result)
print(results)
