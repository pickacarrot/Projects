import pandas as pd
import numpy as np
import csv as csv


Dir='/Users/PullingCarrot/Documents/Projects/Kaggle/Titanic/data'

import os
os.chdir(Dir)


# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe

# convert all strings to integer classifiers.
# fill in the missing values of the data and make it complete.

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# All the ages with no data -> make the median of all Ages
# ceil all ages
train_df.Age = ceil(train_df.Age)
min_age = train_df.Age.dropna().min()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = min_age

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Fare'], axis=1) 


# TEST DATA
test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


# All the ages with no data -> make the median of all Ages
test_df.Age = ceil(test_df.Age)
mini_age = test_df.Age.dropna().min()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = mini_age

# All the missing Fares -> assume median of their respective class
#if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
#   median_fare = np.zeros(3)
#    for f in range(0,3):                                              # loop 0 to 2
#        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
#    for f in range(0,3):                                              # loop 0 to 2
#        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Fare'], axis=1) 

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values

#****************
###Models
#0 Original Random Forests
from sklearn.ensemble import RandomForestClassifier
model0 = RandomForestClassifier(n_estimators=100)

#1 Boosting
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

models=[model0]

for i in range(10):
    modeli=GradientBoostingClassifier(n_estimators=200, learning_rate=0.1*i+0.1, max_depth=1, random_state=0)
    models.append(modeli)


#models=[model0,model1,model2,model3,model4]

# Cross Validation
nFold=10
cv_index=np.random.random_integers(low=0, high=nFold-1, size=len(train_data))

X_all=train_data[0::,1::]
Y_all=train_data[0::,0]

modelN=len(models)

cv_mse=np.zeros(modelN)

for i in range(nFold):
    X_train=X_all[cv_index!=i]
    y_train=Y_all[cv_index!=i]
    X_test=X_all[cv_index==i]
    y_test=Y_all[cv_index==i]
    
    for j in range(modelN):
        models[j].fit(X_train, y_train)
        y_pred=models[j].predict(X_test)
        
        cv_mse[j]+=np.sum(abs(y_test-y_pred))/len(y_test)/nFold    
    
print cv_mse

#****************
model_optimal=GradientBoostingClassifier(n_estimators=200, learning_rate=0.7, max_depth=1, random_state=0)

model_optimal = model_optimal.fit( train_data[0::,1::], train_data[0::,0] )

output = model_optimal.predict(test_data).astype(int)

predictions_file = open("myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()

