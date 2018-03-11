import sys
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import confusion_matrix
import pickle
#import coremltools

df = pd.read_csv("fake_clean.csv")
df=df.dropna()

train, test = train_test_split(df, test_size=0.2)

#converting text using tf-idf vectorizer
tfidf1 = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,lowercase=True,ngram_range=(1,3),max_df=0.9, min_df=0.1)
tfidf2 = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,lowercase=True,ngram_range=(1,3),max_df=0.9, min_df=0.1)

train=pd.concat([train,pd.DataFrame(tfidf1.fit_transform(train['text'].values.astype('U')).toarray().tolist())],axis=1)
train=pd.concat([train,pd.DataFrame(tfidf2.fit_transform(train['title'].values.astype('U')).toarray().tolist())],axis=1)

test=pd.concat([test,pd.DataFrame(tfidf1.transform(test['text'].values.astype('U')).toarray().tolist())],axis=1)
test=pd.concat([test,pd.DataFrame(tfidf2.transform(test['title'].values.astype('U')).toarray().tolist())],axis=1)

train=train.fillna(0)
test=test.fillna(0)

y_train=train.status
y_test=test.status

train.drop(['text','title','status'],inplace=True,axis=1)
test.drop(['text','title','status'],inplace=True,axis=1)

#Classification model to learn the news features
model = RandomForestClassifier(n_estimators=200, oob_score='TRUE', n_jobs=-1, random_state=50, max_features="auto",min_samples_leaf=1)
model.fit(train, y_train)
y_pred=model.predict(test)
output = pd.DataFrame()
output['status']=y_pred
# calculatig the importance of features
features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = model.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
# plotting the importance
features.set_index('feature', inplace=True)
#features.plot(kind='barh', figsize=(20, 20))
print("Random Forest score is:",model.score(test,y_test))
print("accuracy of Random Forest:",accuracy_score(y_pred,y_test))
print("confusion matrix of Random Forest:",confusion_matrix(y_test, y_pred))

#json to be given to app developer
out = output.to_json(orient='records')[1:-1].replace('},{', '} {')
with open('output.txt', 'w') as f:
    f.write(out)
