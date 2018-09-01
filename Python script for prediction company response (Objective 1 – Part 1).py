# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:12:54 2018

@author: rudas
"""

# Importing the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve
from sklearn import tree
import pydotplus
import graphviz
from IPython.display import Image 
# In[20]:
# Importing the dataset
url = 'C:\\Users\\rudas\\Dropbox\\CapstoneProject\\DataSet\\TaxComplaint.csv'
complaint = pd.read_csv(url, parse_dates=['Date received', 'Date sent to company'])
complaint.info()
# In[21]
# Renaming some of the columns
complaint = complaint.rename(columns = {'Date received':'DateReceived', 'Sub-product':'SubProduct', 
                                        'Company public response':'CompanyPublicResponse',
                                        'Consumer consent provided?':'ConsumerConsentProvided', 
                                        'Submitted via':'SubmittedVia', 
                                        'Date sent to company':'DateSentToCompany',
                                        'Company response to consumer':'CompanyResponseToConsumer', 
                                        'Timely response?':'TimelyResponse',
                                        'Consumer disputed?':'ConsumerDisputed'})
# In[22]:
# Drop the columns with null values
complaint = complaint.drop(columns=['CompanyPublicResponse','ConsumerConsentProvided'])
# In[7]:
# check the number of features for each class in our dataset
pd.value_counts(complaint['SubProduct'], sort=True)
# In[6]:
pd.value_counts(complaint['Issue'], sort=True)
# In[7]:
pd.value_counts(complaint['Company'], sort=True)
# In[23]:
# Just taking top 5 companies with most complaints for the analysis
complaint = complaint[(complaint['Company']=='BANK OF AMERICA, NATIONAL ASSOCIATION')|
        (complaint['Company']=='WELLS FARGO & COMPANY')|
        (complaint['Company']=='OCWEN LOAN SERVICING LLC')|
        (complaint['Company']=='JPMORGAN CHASE & CO.')|
        (complaint['Company']=='NATIONSTAR MORTGAGE')]
# In[9]:
pd.value_counts(complaint['TimelyResponse'], sort=True)
# In[9]:
pd.value_counts(complaint['CompanyResponseToConsumer'], sort=True)
# In[11]:
pd.value_counts(complaint['ConsumerDisputed'], sort=True)
# In[24]:
# Replacing the dataset categorical variables into numeric codes
complaint['month'] = complaint['DateReceived'].dt.month
# In[25]:
complaint['Issue'] = complaint['Issue'].map({'Loan modification,collection,foreclosure': 1, 
         'Loan servicing, payments, escrow account': 2, 'Application, originator, mortgage broker':3, 
         'Settlement process and costs':4,'Credit decision / Underwriting':5,'Other':6})
# In[26]:
complaint['SubProduct'] = complaint['SubProduct'].map({'Second mortgage': 1, 
         'Reverse mortgage': 2, 'VA mortgage':3, 
         'FHA mortgage':4,'Conventional adjustable mortgage (ARM)':5,'Conventional fixed mortgage':6,
         'Home equity loan or line of credit':7,'Other mortgage':8})
# In[27]:
complaint['Company'] = complaint['Company'].map({'BANK OF AMERICA, NATIONAL ASSOCIATION':1,
         'WELLS FARGO & COMPANY':2,'OCWEN LOAN SERVICING LLC':3, 'JPMORGAN CHASE & CO.':4,
         'NATIONSTAR MORTGAGE':5})
# In[28]:
complaint['TimelyResponse'] = complaint['TimelyResponse'].map({'Yes':1, 'No':0})
# In[29]:
complaint['CompanyResponseToConsumer'] = complaint['CompanyResponseToConsumer'].map({'Closed with explanation':1,
                                                                                     'Closed with non-monetary relief':2,
                                                                                     'Closed without relief':3,
                                                                                     'Closed':4,
                                                                                     'Closed with monetary relief':5,
                                                                                     'Closed with relief':6})
# In[30]:
complaint['ConsumerDisputed'] = complaint['ConsumerDisputed'].map({'Yes':1, 'No':0})
# In[37]:
# Drop redundant columns
complaint = complaint.drop(columns=['State','DateReceived','DateSentToCompany','SubmittedVia'])
complaint_wo_na = complaint[np.isfinite(complaint['CompanyResponseToConsumer'])]
complaint = complaint.drop(columns=['CompanyResponseToConsumer'])
# In[32]:
# Creating the required functions
def evaluateModel(estimator, X_test, y_test, name):
    """This function takes the tuned classifier and produces the accuracy, RSME, Confusion Matrix, and 
    Classification Report for the classifier supplied"""
    print("Classifer: ",name)
    print("\nAccuracy: {0:.4f}".format(estimator.score(X_test, y_test)))
    # Make prediction and print the confusion matrix
    y_pred = estimator.predict(X_test)
    # RSME value
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("\nRoot Mean Squared Error: {0:.4f}".format(rmse))
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, y_pred)) 
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
# The following function is from:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects
    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt    
# ## Predicting Issues
# Create arrays for the predictors and the target variable
y = complaint['Issue'].values
X = preprocessing.scale(complaint.drop('Issue', axis=1).values)
# In[22]:
# Split into training and test set : we keep 80% of our data as training set and 20% as test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123, stratify=y)
# In[23]:
# KFold Cross-Validation
fold = KFold(len(X_train), n_folds=5, shuffle=True)
# In[24]:
# Decision Tree Classifier
# Create the classifier
decTree = DecisionTreeClassifier()
# Tuning hyper-parameters
params = {'criterion': ['gini', 'entropy']}
decTreeEst = GridSearchCV(estimator=decTree, cv=fold, param_grid=params)
# In[39]:
# Fit the classifer to the training data
decTreeEst.fit(X_train, y_train)
# In[26]:
# Creating the classifier with tuned hyper-parameter
bestDecTreeModel = DecisionTreeClassifier(criterion=decTreeEst.best_estimator_.criterion)
# In[27]:
# Evaluating the model
evaluateModel(decTreeEst,X_test, y_test, 'Decision Tree Classifier')
plot_learning_curve(bestDecTreeModel, 'Decision Tree Classifier', X_test, y_test)
# In[44]:
## Visualizing the Decision Tree
# Fit the best model
bestDecTreeModel.fit(X_train, y_train)
# Create DOT Data
dot_data = tree.export_graphviz(bestDecTreeModel, out_file=None, 
                                feature_names=colName,
                                class_names=list(['1','2','3','4','5','6']))
# Create GraphViz object
graph = pydotplus.graph_from_dot_data(dot_data) 
# Draw graph
Image(graph.create_png())
# In[59]:
# Score the best Decision Tree model
colName = complaint.columns
colName = colName[np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18])]
pd.DataFrame(bestDecTreeModel.feature_importances_, index=colName, 
             columns = ['Importance']).sort_values('Importance', ascending=False)
# In[33]:
# Random Forest Classifier
# Create the classifier
rForest = RandomForestClassifier()
# In[34]
# Tuning hyper-parameters
params = {'n_estimators': [10, 50, 100, 200, 500], 'criterion': ['gini', 'entropy']}
rForestEst = GridSearchCV(estimator=rForest, cv=fold, param_grid=params)
# In[35]:
# Fit the classifer to the training data
rForestEst.fit(X_train, y_train)
# In[62]:
# Creating the classifier with tuned hyper-parameter
bestRandForestModel = RandomForestClassifier(n_estimators=rForestEst.best_estimator_.n_estimators, 
                                             criterion=rForestEst.best_estimator_.criterion)
# In[63]:
# Score the best Random Forest model
bestRandForestModel.fit(X_train, y_train)
colName = complaint.columns
colName = colName[np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18])]
pd.DataFrame(bestRandForestModel.feature_importances_, 
             index=colName, columns = ['Importance']).sort_values('Importance', ascending=False)
# In[37]:
# Evaluating the model
evaluateModel(rForestEst,X_test, y_test, 'Random Forest Classifier')
plot_learning_curve(bestRandForestModel, 'Random Forest Classifier', X_test, y_test)
# ## Predicting Company Response to Consumer
# In[39]:
# Create arrays for the predictors and the target variable
y = complaint_wo_na['CompanyResponseToConsumer'].values
X = preprocessing.scale(complaint_wo_na.drop('CompanyResponseToConsumer', axis=1).values)
# In[45]:
# Split into training and test set : we keep 80% of our data as training set and 20% as test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123, stratify=y)
# KFold Cross-Validation
fold = KFold(len(X_train), n_folds=5, shuffle=True)
# In[48]:
# Decision Tree Classifier
# Create the classifier
decTree = DecisionTreeClassifier()
# Tuning hyper-parameters
params = {'criterion': ['gini', 'entropy']}
decTreeEst = GridSearchCV(estimator=decTree, cv=fold, param_grid=params)
# Fit the classifer to the training data
decTreeEst.fit(X_train, y_train)
# Creating the classifier with tuned hyper-parameter
bestDecTreeModel = DecisionTreeClassifier(criterion=decTreeEst.best_estimator_.criterion)
# Predict with the best model
bestDecTreeModel.fit(X_train, y_train)
# In[50]:
# Evaluating the model: Decision Tree Classification
evaluateModel(decTreeEst,X_test, y_test, 'Decision Tree Classification')
plot_learning_curve(bestDecTreeModel, 'Decision Tree Classification', X_test, y_test)
# Score the best Random Forest model
colName = complaint_wo_na.columns
colName = colName[np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19])]
pd.DataFrame(bestDecTreeModel.feature_importances_, 
             index=colName, columns = ['Importance']).sort_values('Importance', ascending=False)
# In[51]:
# Random Forest Classifier
# Create the classifier
rForest = RandomForestClassifier()
# Tuning hyper-parameters
params = {'n_estimators': [10, 50, 100, 200, 500], 'criterion': ['gini', 'entropy']}
rForestEst = GridSearchCV(estimator=rForest, cv=fold, param_grid=params)
# Fit the classifer to teh training data
rForestEst.fit(X_train, y_train)
# Creating the classifier with tuned hyper-parameter
bestRForestModel = RandomForestClassifier(n_estimators=rForestEst.best_estimator_.n_estimators, 
                                             criterion=rForestEst.best_estimator_.criterion)
# Predict with the best model
bestRForestModel.fit(X_train, y_train)
# In[53]:
# Evaluating the model: Random Forest Classification
evaluateModel(rForestEst,X_test, y_test, 'Random Forest Classification')
plot_learning_curve(bestRForestModel, 'Random Forest Classification', X_test, y_test)
# Score the best Random Forest model
colName = complaint_wo_na.columns
colName = colName[np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19])]
pd.DataFrame(bestRForestModel.feature_importances_, 
             index=colName, columns = ['Importance']).sort_values('Importance', ascending=False)
