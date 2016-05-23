# -*- coding: utf-8 -*-
#This version focus on Feature Engineering
#Level 1 Word matching DONE
#Level 2 Word block matching DONE
#Level 3 Material matching
#Level 4 Number / dimension matching [IN PROGRESS]
#Level 5 Add extra words: eg. 1) R-19 -> add r, 19, r19; 2) 2-Light -> add 2, light, 2light; 3) 6ft -> 6, ft, 6-ft;
#Level 5B After adding, if match, then delete from search query, to avoid duplicates
#Level 6 Attribute Info [IN PROGRESS]
import time
start_time = time.time()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
import _pickle as pickle

#from sklearn import pipeline, model_selection
from sklearn import pipeline, grid_search
#from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
#from nltk.metrics import edit_distance
from nltk.stem.porter import *
stemmer = PorterStemmer()
#from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
#stemmer = SnowballStemmer('english')
import re

import xgboost as xgb


# Load the data into DataFrames
Dir = '/Users/zhenyuz/Documents/Projects/Kaggle/HomeDepot/data'

### Load
pickle_file = '%s/df_v5A_5.pickle'%(Dir)
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  df_all = save['df_all']
  del save  # hint to help gc free up memory
  print('df_all', df_all.shape)
##########################

df_train = pd.read_csv('%s/train.csv'%(Dir), encoding="ISO-8859-1")
df_test = pd.read_csv('%s/test.csv'%(Dir), encoding="ISO-8859-1")
num_train = df_train.shape[0]

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)


df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']
y_train = df_train['relevance'].values
#y_train = pd.DataFrame(df_train['relevance'].values,columns=['relevance'])
X_train =df_train[:]
X_test = df_test[:]
print("--- Features Set: %s minutes ---" % round(((time.time() - start_time)/60),2))



########### Model Fitting

rfr = RandomForestRegressor(n_estimators = 300, n_jobs = -1, random_state = 2009, verbose = 1)

#best: 35, 20
param_grid = {'max_features': [30,35,45],'max_depth': [20,25,30]}

d_col_drops=['id','relevance']
X_train2 = X_train.drop(d_col_drops,axis=1).values
model = grid_search.GridSearchCV(estimator = rfr, param_grid = param_grid, n_jobs=-1, cv = 2, verbose = 20, refit=False)
model.fit(X_train2, y_train)

print("Grid Scores:")
print(model.grid_scores_)
print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

#Extracting scores
scores = [x[1] for x in model.grid_scores_]
scores = np.array(scores).reshape(len(param_grid['xgb_model__n_estimators']), len(param_grid['xgb_model__learning_rate']))



#Training error
y_train_pred = model.predict(X_train2)
TMSE=fmean_squared_error(y_train, y_train_pred)
print(TMSE)

f1=open('%s/Grid_score.txt'%(Dir), 'w+')
f1.write(str(model.grid_scores_))

#Prediction
model = xgb.XGBRegressor(n_estimators=800,learning_rate=0.025,max_depth=6,silent=True, nthread=-1, gamma=0.000001, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0, missing=None)
model.fit(X_train2, y_train)


X_test2 = X_test.drop(d_col_drops,axis=1).values
y_pred = model.predict(X_test2)
y_pred=[max(1.,min(x,3.)) for x in y_pred]
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('%s/submission_v5_xgboost.csv'%(Dir),index=False)
print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time)/60),2))




###########################   Feature Importance

rfr = RandomForestRegressor(n_estimators = 1000, n_jobs = -1, random_state = 2009, verbose = 1,max_features=25,max_depth=20)
rfr.fit(X_train2, y_train)

#Plot Feature Importance

forest=rfr
n_feature=forest.n_features_
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

# Feature Names
feature_names=X_train.drop(d_col_drops,axis=1).columns


# Print the feature ranking
print("Feature ranking:")

feature_name_sort=[]
for f in range(n_feature):
    print("%d. feature %d %s (%f)" % (f + 1, indices[f], feature_names[indices[f]],importances[indices[f]]))
    feature_name_sort.append(feature_names[indices[f]])

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(n_feature), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(n_feature),feature_name_sort,rotation=90,fontsize=14)
plt.xlim([-1, n_feature])
plt.show()



for f in range(n_feature):
    print("%d. feature %d %s (%f)" % (f + 1, indices[f], feature[indices[f]], importances[indices[f]]))













