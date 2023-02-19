# =============================================================================
# Quelco 
# Credit Risk ALGORITHM
# For ANY questions/problems/help, email me: ferdinantos.kottas.2019@mumail.ie
# =============================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.feature_selection import f_classif
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import chi2_contingency
import plotly.offline as py 
py.init_notebook_mode(connected=True) # this code, allow us to work with offline plotly version
import plotly.graph_objs as go # it's like "plt" of matplot
import plotly.tools as tls # It's useful to we get some tools of plotly
import warnings # This library will be used to ignore some warnings
from collections import Counter # To do counter of some features
from optbinning import OptimalBinning
from sklearn import model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
import scikitplot as skplt
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector
import seaborn as sn
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector

#create usefull functions


def describex(data):
    data = pd.DataFrame(data)
    stats = data.describe()
    skewness = data.skew()
    kurtosis = data.kurtosis()
    skewness_df = pd.DataFrame({'skewness':skewness}).T
    kurtosis_df = pd.DataFrame({'kurtosis':kurtosis}).T
    return stats.append([kurtosis_df,skewness_df])

# import dataset

df = pd.read_csv("C:/Users/Ferdinantos Kottas/Desktop/ergasies aristotelio/dataset.csv")
data = df

# correlation matrix
subset = df.loc[:,['bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4',
       'bill_amt5', 'bill_amt6', 'pay_amt1', 'pay_amt2', 'pay_amt3',
       'pay_amt4', 'pay_amt5', 'pay_amt6']]
corr_matrix = subset.corr()
sn.heatmap(corr_matrix, annot=True)
plt.show()

#descriptive statistics

T1 = describex(df)
T1

# inconsistent  data (negative prices in )

df = df[(df.bill_amt1 >= 0) | (df.bill_amt2 >= 0) | (df.bill_amt3 >= 0)| 
        (df.bill_amt6 >= 0) | (df.bill_amt4 >= 0 )| (df.bill_amt5 >= 0)] 

#create 4 variables

df["bill_amtL3M"] = df.bill_amt1 + df.bill_amt2 + df.bill_amt3 
df["bill_amtL6M"] = df.bill_amt1 + df.bill_amt2 + df.bill_amt3 + df.bill_amt4 + df.bill_amt5 + df.bill_amt6

df["pay_amtL3M"] = df.pay_amt1 + df.pay_amt2 + df.pay_amt3 
df["pay_amtL6M"] = df.pay_amt1 + df.pay_amt2 + df.pay_amt3 + df.pay_amt4 + df.pay_amt5 + df.pay_amt6

# correlation matrix
subset = df.loc[:,['bill_amtL3M', 'bill_amtL6M', 
'pay_amtL3M', 'pay_amtL6M']]
corr_matrix = subset.corr()
sn.heatmap(corr_matrix, annot=True)
plt.show()
#plots 

list_col = ['age', 'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5','pay_6'] # few columns to have plots

for x in list_col: # df.columns.drop('default.payment.next.month') if you want everything to be ploted 

    df_good = df[df['default.payment.next.month'] == 0]
    df_bad = df[df['default.payment.next.month'] == 1]
    fig, ax = plt.subplots(nrows=2, figsize=(12,8))
    plt.subplots_adjust(hspace = 0.4, top = 0.8)
    g1 = sns.distplot(df_good[x], ax=ax[0], 
             color="g")
    g1 = sns.distplot(df_bad[x], ax=ax[0], 
             color='r')
    g1.set_title(str(x) + " Distribuition", fontsize=15)
    g1.set_xlabel(str(x))
    g1.set_xlabel("Frequency")
    g2 = sns.countplot(x=x,data=df, 
              palette="hls", ax=ax[1], 
              hue = "default.payment.next.month")
    g2.set_title(str(x) + " Counting by Risk", fontsize=15)
    g2.set_xlabel(str(x))
    g2.set_xlabel("Count")
    plt.show()


#outliers as nan

list_outliers = [ 'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4',
       'bill_amt5', 'bill_amt6', 'pay_amt1', 'pay_amt2', 'pay_amt3',
       'pay_amt4', 'pay_amt5', 'pay_amt6', 'bill_amtL3M', 'bill_amtL6M', 
       'pay_amtL3M', 'pay_amtL6M']

for x in list_outliers:
    q75,q25 = np.percentile(df.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(1.5*intr_qr)
    #min = q25-(1.5*intr_qr)
 
    #df.loc[df[x] < min,x] = np.nan
    df.loc[df[x] > max,x] = np.nan

df.isna().sum()
(df.isna().sum()/25796)*100

# From myData object, read included features and target variable.
# You can select a specific row/column of the dataframe object by using 'iloc',
# Make sure that, after selecting (/accessing) the desired rows and columns,
# you extract their 'values', which is what you want to store in variables 'X' and 'y'.
y = df.loc[:,'default.payment.next.month']
X = df.loc[:,df.columns.drop('default.payment.next.month')]

# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# 'random_state' parameter should be fixed to a constant integer (i.e. 0) so that the same split
# will occur every time this script is run.
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.25, random_state = 0)


# discretization of the continues variables (woe)
my_dictionary = {}


for x in df.columns.drop('default.payment.next.month'):
    optb = OptimalBinning(name=x, dtype="numerical", solver="cp")
    optb.fit(x_train[x], y_train)
    print(optb.status)
    binning_table = optb.binning_table
    tab = binning_table.build()
    my_dictionary.update({x:tab})
    binning_table.plot(metric="woe")
    binning_table.plot(metric="event_rate")
    x_train[str(x)+"_WOE"] = optb.transform(x_train[x], metric="woe")
    pd.Series(x_train[str(x)+"_WOE"]).value_counts()
    x_test[str(x)+"_WOE"] = optb.transform(x_test[x], metric="woe")
    df[str(x)+"_WOE"] = optb.transform(df[x], metric="woe")
    x_train = x_train.drop([x],axis=1)
    x_test = x_test.drop([x],axis=1) # if you want to drop the other variables use x_train.drop(x)

t3 = x_train.isna().sum()
t4 = x_test.isna().sum()
t5 = df.isna().sum()
#models
# Create your models for different combinations of parameters
# Train the model and predict. Save the performance metrics.
# =============================================================================

# Create logistic regression object
#tunning
logistic = LogisticRegression()

#feature elimination
# Perform stepwise regression for feature selection
sfs = SequentialFeatureSelector(logistic,
                                k_features=30,
                                #tol= 0.8,
                                forward=True,
                                scoring='accuracy',
                                cv=None)
selected_features = sfs.fit(x_train,y_train)
selected_features_names = np.array(selected_features.k_feature_names_)
selected_features_names

#selected features
x_train_log = x_train[selected_features_names]
x_test_log = x_test[selected_features_names]

# Create a list of all of the different penalty values that you want to test and save them to a variable called 'penalty'
penalty = ['l1', 'l2']# Create a list of all of the different C values that you want to test and save them to a variable called 'C'
C = [0.0001, 0.001, 0.01, 1, 100]# Now that you have two lists each holding the different values that you want test, use the dict() function to combine them into a dictionary. # Save your new dictionary to the variable 'hyperparameters'
hyperparameters = dict(C=C, penalty=penalty)# Fit your model using gridsearch
clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
best_model = clf.fit(x_train_log, y_train)#Print all the Parameters that gave the best results:
print('Best Parameters',clf.best_params_)# You can also print the best penalty and C value individually from best_model.best_estimator_.get_params()
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
clf.feature_names_in_ # array(['age_WOE', 'pay_0_WOE', 'pay_2_WOE', 'pay_5_WOE', 'pay_6_WOE',
     #   'bill_amt3_WOE', 'bill_amt4_WOE', 'pay_amt1_WOE',
     #  'education:0_WOE', 'education:1_WOE', 'education:4_WOE',
     #  'education:5_WOE', 'education:6_WOE', 'marriage:0_WOE',
     #  'marriage:3_WOE'], dtype=object)
# evaluate model on the hold out dataset
yhat = best_model.predict(x_test_log)
# evaluate the model
acc = accuracy_score(y_test, yhat)
rec = recall_score(y_test, yhat)
pre = precision_score(y_test, yhat)
f1 = f1_score(y_test, yhat)

y_pred_proba = clf.predict_proba(x_test_log)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)

#confusion matrix
confusion_matrix = confusion_matrix(y_test, yhat)

cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show() 

#ks plot
#skplt.metrics.plot_ks_statistic(y_test, y_pred_proba)
#plt.show()

#create ROC curve
log_disp = RocCurveDisplay.from_estimator(clf, x_test_log, y_test)
plt.show()

plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#save mtrics
metric_log = [accuracy_score(y_test, yhat),
                  recall_score(y_test, yhat),
                  precision_score(y_test, yhat),
                  f1_score(y_test, yhat)]


#random forest
# enumerate splits
outer_results = list()
# configure the cross-validation procedure
cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
# define the model
model = RandomForestClassifier(random_state=1)
#feature elimination
model.fit(x_train,y_train)
f_i = list(zip(x_train.columns,model.feature_importances_))
f_i.sort(key = lambda x : x[1])
plt.barh([x[0] for x in f_i],[x[1] for x in f_i])

plt.show()

#feature
rfe = RFECV(model, min_features_to_select= 5 ,scoring = "accuracy")
rfe.fit(x_train,y_train)
selected_features = np.array(x_train.columns)[rfe.get_support()]
selected_features 
# selected feautres ['limit_bal_WOE', 'age_WOE', 'pay_0_WOE', 'pay_2_WOE', 'pay_3_WOE',
#      'pay_4_WOE', 'pay_5_WOE', 'pay_6_WOE', 'bill_amt1_WOE',
#      'bill_amt2_WOE', 'bill_amt3_WOE', 'bill_amt4_WOE', 'bill_amt5_WOE',
#      'bill_amt6_WOE', 'pay_amt1_WOE', 'pay_amt2_WOE', 'pay_amt3_WOE',
#      'pay_amt4_WOE', 'pay_amt5_WOE', 'pay_amt6_WOE', 'sex:1_WOE',
#      'sex:2_WOE', 'education:1_WOE', 'education:2_WOE',
#      'education:3_WOE', 'education:4_WOE', 'education:5_WOE',
#      'marriage:0_WOE', 'marriage:1_WOE', 'marriage:2_WOE',
#      'bill_amtL3M_WOE', 'bill_amtL6M_WOE', 'pay_amtL3M_WOE',
#      'pay_amtL6M_WOE']
    
x_train_rf = x_train[selected_features]
x_test_rf = x_test[selected_features]

#tunning model
# define search space
space = dict()
space['n_estimators'] = [10, 100, 500]
space['max_features'] = [3, 4, 5, 6, 8, 10]
# define search
search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
# execute search
result = search.fit(x_train_rf, y_train)
# get the best performing model fit on the whole training set
best_model1 = result.best_estimator_
# evaluate model on the hold out dataset
yhat = best_model1.predict(x_test_rf)

y_pred_proba1=best_model1.predict_proba(x_test_rf)

result.feature_names_in_ # ['limit_bal_WOE', 'age_WOE', 'pay_0_WOE', 'pay_2_WOE', 'pay_3_WOE',
       #'pay_4_WOE', 'pay_5_WOE', 'pay_6_WOE', 'bill_amt1_WOE',
       #'bill_amt2_WOE', 'bill_amt3_WOE', 'bill_amt4_WOE', 'bill_amt5_WOE',
       #'bill_amt6_WOE', 'pay_amt1_WOE', 'pay_amt2_WOE', 'pay_amt3_WOE',
       #'pay_amt4_WOE', 'pay_amt5_WOE', 'pay_amt6_WOE', 'sex:1_WOE',
       #'sex:2_WOE', 'education:1_WOE', 'education:2_WOE',
       #'education:3_WOE', 'education:4_WOE', 'education:5_WOE',
       #'marriage:0_WOE', 'marriage:1_WOE', 'marriage:2_WOE',
       #'bill_amtL3M_WOE', 'bill_amtL6M_WOE', 'pay_amtL3M_WOE',
       #'pay_amtL6M_WOE']

#confusion matrix
confusion_matrix = confusion_matrix(y_test, yhat)

cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show() 

#ks plot
skplt.metrics.plot_ks_statistic(y_test, y_pred_proba1)
plt.show()

# evaluate the model
acc1 = accuracy_score(y_test, yhat)
rec1 = recall_score(y_test, yhat)
pre1 = precision_score(y_test, yhat)
f11 = f1_score(y_test, yhat)

#plot roc
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(best_model1, x_test_rf, y_test, ax=ax, alpha=0.8)
log_disp.plot(ax=ax, alpha=0.8)
plt.show()

# report progress
print('>acc=%.3f,>recall=%.3f,>precision=%.3f, >f1=%.3f, est=%.3f, cfg=%s' % (acc1,rec1,pre1, f11, result.best_score_, result.best_params_))

metric_rf = [accuracy_score(y_test, yhat),
                  recall_score(y_test, yhat),
                  precision_score(y_test, yhat),
                  f1_score(y_test, yhat)]


#evaluation models plots

# CREATE MODELS AND PLOTS HERE

labels = ['Accuracy', 'Recall', 'Precision', 'F1']

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, metric_rf, width, label='RF')
rects2 = ax.bar(x + width/2, metric_log, width, label='Logistic')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Metrics by ML models')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()

#add predicted results

df["predicted results"] = best_model1.predict(df.loc[:,np.array(result.feature_names_in_)])

