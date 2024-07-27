import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#Step2

        train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()

#Step3
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()

#Step 4
women = train_data.loc[train_data.Sex == 'female']['Survived']
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)
#Step 4 rresult : % of women who survived: 0.7420382165605095

#Step5 
men = train_data.loc[train_data.Sex == 'male']['Survived']
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men)

#Step5 result: % of men who survived: 0.18890814558058924

#Step 6
from sklearn.ensemble import RandomForestClassifier
#import the random forest algorithm
y = train_data["Survived"]
# fom the train data csv we create the variable y for the survived
features = ["Pclass", "Sex", "SibSp", "Parch"]
# we add attributes from the csv 
#these will be the attributes that will be analised 
#to generate a survival rate.
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
#the above features for train data and test data

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
#we use 100 trees to do estimations 
model.fit(X, y)
predictions = model.predict(X_test)


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
#concatenate data to show prediction based on passenger id.
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved! ")

