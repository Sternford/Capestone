import pandas as pd
import sys
from sklearn import svm, metrics
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

#load data
trainingData = pd.read_csv(r".\data\vehicles_train.csv")

#data preprocessing
le = LabelEncoder()

for col in trainingData.columns.values:

    # Encoding only categorical variables
    if trainingData[col].dtypes == 'object':

        # Using whole data to form an exhaustive list of levels

        data = trainingData[col].append(trainingData[col])
        le.fit(data.values)
        trainingData[col] = le.transform(trainingData[col])

trainingTarget = trainingData.labelss
trainingData = trainingData.drop(['labelss'], axis=1)

X = trainingData
y= trainingTarget

#model builiding and training
model = svm.SVC()
model.fit(X,y)

#saving model state
joblib.dump(model, 'svm_model/svm_model.pkl')
print("model saved successful!")

sys.exit()
