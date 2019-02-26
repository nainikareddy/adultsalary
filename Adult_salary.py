# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 20:11:09 2018

@author: Nayanika Reddy
"""

import pandas as pd
#Train data
salary = pd.read_csv("C:/Downloads/Classification_problem.csv",encoding='utf-8')
#Test data
salarytest = pd.read_csv("C:/Downloads/Classification_problem_test.csv",encoding='utf-8')
#removing common ? values in train data for WorkClass,Occupation,Native-Country for train data
salary = salary[(salary.WorkClass != ' ?') | (salary.Occupation != ' ?') | (salary['Native-Country'] != ' ?')]
salary.head()
salary.shape
"""
(32534, 15)
"""

#removing common ? values in train data for WorkClass,Occupation,Native-Country for test data
salarytest = salarytest[(salarytest.WorkClass != '?') | (salarytest.Occupation != '?') | (salarytest['Native-Country'] != '?')]
salarytest.head()
salarytest.shape
"""
(16262, 15)"""

#replacing missing values with most occuring values for train data
salary.WorkClass = salary.WorkClass.replace(' ?',' Private')
salary.Occupation = salary.Occupation.replace(' ?',' Prof-specialty')
salary['Native-Country']= salary['Native-Country'].replace(' ?',' United-States')
salary.WorkClass.value_counts()
"""
 Private             24505
 Self-emp-not-inc     2541
 Local-gov            2093
 State-gov            1298
 Self-emp-inc         1116
 Federal-gov           960
 Without-pay            14
 Never-worked            7
Name: WorkClass, dtype: int64 
"""
#replacing missing values with most occuring values for test data
salarytest.WorkClass = salarytest.WorkClass.replace('?','Private')
salarytest.Occupation = salarytest.Occupation.replace('?','Prof-specialty')
salarytest['Native-Country']= salarytest['Native-Country'].replace('?','United-States')
salarytest.WorkClass.value_counts()
"""
Private             12154
Self-emp-not-inc     1321
Local-gov            1043
State-gov             683
Self-emp-inc          579
Federal-gov           472
Without-pay             7
Never-worked            3
Name: WorkClass, dtype: int64 

"""

#label encoding for Salary and Sex for train data
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
salary.Sex = encoder.fit_transform(salary.Sex)

salary.Salary=salary.Salary.map({' <=50K':0, ' >50K':1})

#label encoding for Salary and Sex for test data

salarytest.Sex = encoder.fit_transform(salarytest.Sex)

salarytest.Salary=salarytest.Salary.map({'<=50K.':0, '>50K.':1})

salary = pd.get_dummies(salary,columns=['WorkClass','Education','Marital-Status','Occupation','Relationship', 'Race', 'Native-Country'],drop_first=True)

#converting categorical variables to numeric for test data
salarytest = pd.get_dummies(salarytest,columns=['WorkClass','Education','Marital-Status','Occupation','Relationship', 'Race', 'Native-Country'],drop_first=True)
salary = salary.drop('Native-Country_ Holand-Netherlands',axis=1)


features = salary.loc[:, salary.columns != 'Salary']
target = pd.DataFrame(salary['Salary'])

features_test = salarytest.loc[:, salarytest.columns != 'Salary']
target_test = pd.DataFrame(salarytest['Salary'])

from imblearn.over_sampling import SMOTE
x_train = features
y_train = target
smt = SMOTE()
x_train, y_train = smt.fit_sample(x_train, y_train)

x_test = features_test
y_test = target_test

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

classifier1 = LogisticRegression(C=0.001)
logreg = classifier1.fit(x_train,y_train)

print("Training accuracy: ",logreg.score(x_train,y_train))

predicted = logreg.predict(x_test)
print("Testing Accuracy score: ",accuracy_score(y_test,predicted))
print("Confusion Matrix: ")
print(confusion_matrix(y_test,predicted))

print("\nClassification Report: ")
print(classification_report(y_test, predicted))

"""
Training accuracy:  0.864153540916
Testing Accuracy score:  0.818534005657
Confusion Matrix: 
[[10564  1854]
 [ 1097  2747]]

Classification Report: 
              precision    recall  f1-score   support

           0       0.91      0.85      0.88     12418
           1       0.60      0.71      0.65      3844

   micro avg       0.82      0.82      0.82     16262
   macro avg       0.75      0.78      0.76     16262
weighted avg       0.83      0.82      0.82     16262
"""

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted, pos_label=1)
print("AUC :",metrics.auc(fpr, tpr))
"""
AUC : 0.782660391607
"""
from sklearn.svm import LinearSVC

classifier2 = LinearSVC(penalty='l2',max_iter=250,C=5)
linsvm = classifier2.fit(x_train,y_train)

print("Training accuracy: ",linsvm.score(x_train,y_train))

predicted = linsvm.predict(x_test)
print("Testing Accuracy score: ",accuracy_score(y_test,predicted))
print("Confusion Matrix: ")
print(confusion_matrix(y_test,predicted))

print("\nClassification Report: ")
print(classification_report(y_test, predicted))
"""
Training accuracy:  0.565675993036
Testing Accuracy score:  0.787357028656
Confusion Matrix: 
[[12259   159]
 [ 3299   545]]

Classification Report: 
              precision    recall  f1-score   support

           0       0.79      0.99      0.88     12418
           1       0.77      0.14      0.24      3844

   micro avg       0.79      0.79      0.79     16262
   macro avg       0.78      0.56      0.56     16262
weighted avg       0.78      0.79      0.73     16262
"""
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted, pos_label=1)
print("AUC :",metrics.auc(fpr, tpr))
"""
AUC : 0.56448770113
"""
from sklearn.tree import DecisionTreeClassifier


classifier3=DecisionTreeClassifier(criterion='entropy', random_state=90,max_depth=8)
DTC=classifier3.fit(x_train,y_train)

print("Training accuracy: ",DTC.score(x_train,y_train))

predicted = DTC.predict(x_test)
print("Testing Accuracy score: ",accuracy_score(y_test,predicted))
print("Confusion Matrix: ")
print(confusion_matrix(y_test,predicted))

print("\nClassification Report: ")
print(classification_report(y_test, predicted))
"""
Training accuracy:  0.863465198202
Testing Accuracy score:  0.838457754274
Confusion Matrix: 
[[11622   796]
 [ 1831  2013]]

Classification Report: 
              precision    recall  f1-score   support

           0       0.86      0.94      0.90     12418
           1       0.72      0.52      0.61      3844

   micro avg       0.84      0.84      0.84     16262
   macro avg       0.79      0.73      0.75     16262
weighted avg       0.83      0.84      0.83     16262
"""
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted, pos_label=1)
print("AUC :",metrics.auc(fpr, tpr))
"""
AUC : 0.729786378874
"""
from sklearn.ensemble import RandomForestClassifier


classifier4=RandomForestClassifier(random_state=90,max_depth=10,n_estimators=100)
RF=classifier4.fit(x_train,y_train)

print("Training accuracy: ",RF.score(x_train,y_train))

predicted = RF.predict(x_test)
print("Testing Accuracy score: ",accuracy_score(y_test,predicted))
print("Confusion Matrix: ")
print(confusion_matrix(y_test,predicted))

print("\nClassification Report: ")
print(classification_report(y_test, predicted))
"""
Training accuracy:  0.889136332348
Testing Accuracy score:  0.836121018325
Confusion Matrix: 
[[10707  1711]
 [  954  2890]]

Classification Report: 
              precision    recall  f1-score   support

           0       0.92      0.86      0.89     12418
           1       0.63      0.75      0.68      3844

   micro avg       0.84      0.84      0.84     16262
   macro avg       0.77      0.81      0.79     16262
weighted avg       0.85      0.84      0.84     16262

"""


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted, pos_label=1)
print("AUC :",metrics.auc(fpr, tpr))
"""
AUC : 0.803154835157
"""
from sklearn.linear_model import Perceptron


classifier5=Perceptron(eta0=0.5,max_iter=75)
mlp=classifier5.fit(x_train,y_train)

print("Training accuracy: ",mlp.score(x_train,y_train))

predicted = mlp.predict(x_test)
print("Testing Accuracy score: ",accuracy_score(y_test,predicted))
print("Confusion Matrix: ")
print(confusion_matrix(y_test,predicted))

print("\nClassification Report: ")
print(classification_report(y_test, predicted))

"""
Training accuracy:  0.5
Testing Accuracy score:  0.23637928914
Confusion Matrix: 
[[    0 12418]
 [    0  3844]]

Classification Report: 
              precision    recall  f1-score   support

           0       0.00      0.00      0.00     12418
           1       0.24      1.00      0.38      3844

   micro avg       0.24      0.24      0.24     16262
   macro avg       0.12      0.50      0.19     16262
weighted avg       0.06      0.24      0.09     16262
"""
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted, pos_label=1)
print("AUC :",metrics.auc(fpr, tpr))
"""
AUC : 0.599705577433

"""