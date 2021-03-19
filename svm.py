import pandas as pd
import numpy as np
import pickle
import time
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype('U')

# train Data
trainData = pd.read_csv("data/train.csv", encoding="latin-1")
trainData = clean_dataset(trainData)

# test Data
testData = pd.read_csv("data/test.csv", encoding="latin-1")
testData = clean_dataset(testData)

# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5, max_df = 0.8, sublinear_tf = True, use_idf = True)

train_vectors = vectorizer.fit_transform(trainData['review'])
test_vectors = vectorizer.transform(testData['review'])

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, trainData['rating'])
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

# results
print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(testData['rating'], prediction_linear, output_dict=True)
print('1: ', report['1'])
print('2: ', report['2'])
print('3: ', report['3'])
print('4: ', report['4'])
print('5: ', report['5'])

# Dump models
pickle.dump(vectorizer, open('models/vectorizer.sav', 'wb'))
pickle.dump(classifier_linear, open('models/classifier.sav', 'wb'))
