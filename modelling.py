import sys
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# reload(sys)
# sys.setdefaultencoding('utf8')


df = pd.read_csv("fake_clean.csv")
y = df.spam_score
X = df.drop('spam_score', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

tfidf1 = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,lowercase=True,ngram_range=(1,3),max_df=0.9, min_df=0.1)
tfidf2 = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,lowercase=True,ngram_range=(1,3),max_df=0.9, min_df=0.1)

X_train.append(pd.DataFrame(tfidf1.fit_transform(X_train['text'].values.astype('U')).toarray().tolist()))
X_train.append(pd.DataFrame(tfidf1.fit_transform(X_train['title'].values.astype('U')).toarray().tolist()))

X_test.append(pd.DataFrame(tfidf1.fit_transform(X_test['text'].values.astype('U')).toarray().tolist()))
X_test.append(pd.DataFrame(tfidf1.fit_transform(X_test['title'].values.astype('U')).toarray().tolist()))

X_train.drop(['text','title'],inplace=True,axis=1)

X_test.drop(['text','title'],inplace=True,axis=1)

model = RandomForestRegressor(n_estimators=200, oob_score='TRUE', n_jobs=-1, random_state=50, max_features="auto",min_samples_leaf=1)
model.fit(X_train, y_train)
# calculatig the importance of features
features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = model.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
# plotting the importance
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(20, 20))

#Linear regression and SVM both come out to be bad models for this dataset

lr = linear_model.LinearRegression()
parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
grid = GridSearchCV(lr, parameters)
grid.fit(X_train, y_train)
print ("r2 / variance : ", grid.best_score_)
y_pred = grid.predict(X_test)
residuals = y_test - y_pred
print("Residual sum of squares: %.2f"% np.mean(residuals**2))

SSResiduals = (residuals ** 2).sum()
SSTotal = ((y_test - y_test.mean()) ** 2).sum()

print("Linear Regression R-squared and mean_squared_error: \n")
print("R-squared=", 1 - (SSResiduals / SSTotal))
print("Mean squared error=", sqrt(mean_squared_error(y_test, y_pred)))

sv=SVR()
sv.fit(X_train, y_train)
y_pred=sv.predict(X_test)
print("SVM score is:",sv.score(X_test,y_test))


