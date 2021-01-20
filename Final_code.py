#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
sns.set_style("whitegrid")

# Read input file
df1 = pd.read_csv('final.csv')
df1.head()
df1.describe()
df1

##import data file for google colab
#from google.colab import files
#df1=files.upload()

## to read excel files
#!pip install -q xlrd
#import pandas as pd
#df1 = pd.read_excel('Final.xlsx')
#df1
#df1.head()
#df1.describe()

# Prediction of people living and dead after 1 year calculation of death percentage
live = df1[df1['Risk1Yr'] == 0]
death = df1[df1['Risk1Yr'] == 1]

cond = ['PRE4', 'PRE5', 'PRE6', 'PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11',        'PRE14', 'PRE17', 'PRE19', 'PRE25', 'PRE30', 'PRE32', 'AGE']

l = [np.mean(live[c]) for c in cond]
d = [np.mean(death[c]) for c in cond]

ld = pd.DataFrame(data={'Attribute': cond, 'Live 1yr Mean': l, 'Death 1yr Mean': d})
ld = ld.set_index('Attribute')

print('Death: {:d}, Live: {:d}'.format(len(death), len(live)))
print("1 year death: {:.2f}% out of 1269 patients".format(np.mean(df1.Risk1Yr)*100))
ld

# Percentage difference in means of live vs death patients
d = np.array(d)
l = np.array(l)

p_diff = (d-l)/l*100

fig, axes = plt.subplots(2,1,figsize=(12,18))

axes[0].bar(cond, p_diff)
axes[0].set_title('Mean Difference % between Dead and Live 1yr', fontsize=18)
axes[0].set_xticks(cond)
axes[0].set_xticklabels(cond, rotation=90)
axes[0].set_ylabel('Percent', fontsize=13)

# Count plot of true/false condition columns

tf_col = ['PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 'PRE17', 'PRE19', 'PRE25', 'PRE30', 'PRE32']
tf_sum = [df1[col].sum()/1269 for col in tf_col]

axes[1].bar(tf_col, tf_sum)
axes[1].set_xticks(tf_col)
axes[1].set_xticklabels(tf_col, rotation=90)
axes[1].set_ylabel('Proportion of Total Patients', fontsize=13)
axes[1].set_title('Proportion of Patient Conditions before Surgery', fontsize=18)

plt.tight_layout()
plt.show()

# Count plots of Diagnosis, Tumor_Size, Performance with difference of live and death data

fig, axes = plt.subplots(3,1,figsize=(10,15))

sns.countplot(x='DGN', hue='Risk1Yr', data=df1, palette='Blues_d', ax=axes[0]).set_title('Diagnosis', fontsize=18)
sns.countplot(x='PRE14', hue='Risk1Yr', data=df1, palette='Blues_d', ax=axes[1]).set_title('Tumor_Size', fontsize=18)
sns.countplot(x='PRE6', hue='Risk1Yr', data=df1, palette='Blues_d', ax=axes[2]).set_title('Performance', fontsize=18)

plt.tight_layout()

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""
    data = np.concatenate((data1, data2))
    permuted_data = np.random.permutation(data)
    
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""
    perm_replicates = np.empty(size)
    
    for i in range(size):
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""
    diff = np.mean(data_1) - np.mean(data_2)
    return diff

# Hypothesis testing with Permutations of data
condition = ['PRE4', 'PRE5', 'PRE6', 'PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11',        'PRE14', 'PRE17', 'PRE19', 'PRE25', 'PRE30', 'PRE32', 'AGE']
p_val = []

for c in condition:
    empirical_diff_means = diff_of_means(death[c], live[c])
    perm_replicates = draw_perm_reps(death[c], live[c], diff_of_means, size=10000)
    if empirical_diff_means > 0:
        p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)
        p_val.append(p)
    else:
        p = np.sum(perm_replicates <= empirical_diff_means) / len(perm_replicates)
        p_val.append(p)

print(list(zip(condition, p_val)))

# Scatter plot for FVC, FEV1, Age columns

fig, axes = plt.subplots(1,2,figsize=(13,5))
axes[0].plot(df1.PRE4, df1.PRE5, linestyle='none', marker='.')

axes[0].set_xlabel('FVC', fontsize=13)
axes[0].set_ylabel('FEV1', fontsize=13)
axes[0].set_title('FVC vs FEV1', fontsize=16)

axes[1].plot(df1.AGE, df1.PRE5, linestyle='none', marker='.', label='FEV1')
axes[1].plot(df1.AGE, df1.PRE5, linestyle='none', marker='.', label='FVC')
axes[1].set_xlabel('Age', fontsize=13)
axes[1].set_ylabel('FEV1, FVC', fontsize=13)
axes[1].legend()
axes[1].set_title('Age vs FEV1, FVC', fontsize=16)

plt.tight_layout()


# Correlation coefficients for FVC and FEV1
np.corrcoef(df1.PRE4, df1.PRE5)[0,1]


# Correlation coefficients for Age and FVC
np.corrcoef(df1.AGE, df1.PRE4)[0,1]

# Calculating Optimal Number of Features
X=df1.iloc[:,2:-1]
y=df1["Risk1Yr"].iloc[:]
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=None, step=1)
rfe.fit(X, y)
print("Optimal number of features : %d" % rfe.n_features_)
print(rfe.support_)
print(rfe.ranking_)


#Selecting Best Features using kBest
modelLogit= LogisticRegression()
skb=SelectKBest(score_func=f_classif,k=8)
skbfit=skb.fit(X,y)
print(skbfit.get_support())
best_features=skbfit.transform(X)
print("Scores",skbfit.scores_)
print(best_features.shape)

#
df=pd.concat(df1['PRE4','PRE6','PRE8','PRE9','PRE10','PRE14','AGE'])

df = df1.loc[:, ['PRE4','PRE6','PRE8','PRE9','PRE10','PRE14','AGE','Risk1Yr']]

# Display Features selected
X=df.iloc[:,:-1]
y=df["Risk1Yr"].iloc[:]
X

#Comparison of Accuracies using Algorithms Random Forest, SVM and XGBoost

#Calculating Rough Accuracy
#Splitting Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3, random_state=0)

# Feature Scaling Data
from sklearn.preprocessing import StandardScaler 
sc_X =StandardScaler() 
X_train =sc_X.fit_transform(X_train)
X_test =sc_X.fit_transform(X_test)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm3 = confusion_matrix(y_test, y_pred)
print("Random Forest accuracy: {}%".format(accuracy_score(y_pred,y_test)* 100))

# SVM Accuracy
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("SVM: {}%".format(accuracy_score(y_pred,y_test)* 100))


#Implement XGBoost

import sys
get_ipython().system('{sys.executable} -m pip install xgboost')
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3}  # the number of classes that exist in this datset
num_round = 20  # the number of training iterations

bst = xgb.train(param, dtrain, num_round)
bst.dump_model('dump.raw.txt')
preds = bst.predict(dtest)
preds

best_preds = np.asarray([np.argmax(line) for line in preds])
best_preds

from sklearn.metrics import precision_score
print (precision_score(y_test, best_preds, average='macro')*100)

from sklearn.externals import joblib
joblib.dump(bst, 'bst_model.pkl', compress=True)
# bst = joblib.load('bst_model.pkl') # load it later