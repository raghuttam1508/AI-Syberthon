#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import seaborn as sns
import pickle

from pandas.plotting import scatter_matrix
import matplotlib.lines as mlines
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE


# In[2]:


df = pd.read_csv('C:/Users/raghu/Downloads/PS_20174392719_1491204439457_log.csv') 
print(df.shape)


# In[3]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[4]:


df.head()


# In[5]:


df.isna().sum()


# In[6]:


occ = df['isFraud'].value_counts()
occ


# In[7]:


fraud_ratio = occ/len(df.index)
fraud_ratio


# In[8]:


occ_1= df['isFlaggedFraud'].value_counts()
occ_1 


# In[9]:


fraud_ratio = occ_1/len(df.index)
fraud_ratio


# In[10]:


df.head(2)


# In[11]:


fraudby_type = df.groupby("type")["isFraud"].count()
fraudby_type


# In[12]:


FlaggedFraud = df.loc[(df.isFlaggedFraud == 1) & (df.type == 'TRANSFER')]
print("The no. of Flagged Fraudulent Transactions :", len(FlaggedFraud))

print("Minimum Transaction :", df.loc[df.isFlaggedFraud == 1].amount.min())
print("Maximum Transaction :", df.loc[df.isFlaggedFraud == 1].amount.max())


# In[13]:


dataTransfer = df.loc[df['type'] == 'TRANSFER']
dataTransfer = pd.DataFrame(dataTransfer)
dataTransfer.head(10)


# In[14]:


dataTransfer.loc[(dataTransfer.isFlaggedFraud == 1) & (dataTransfer.oldbalanceOrg == dataTransfer.newbalanceOrig)].sort_values(by = 'oldbalanceOrg').head(10)


# In[15]:


dataTransfer.loc[(dataTransfer.isFlaggedFraud == 1) & (dataTransfer.oldbalanceOrg == dataTransfer.newbalanceOrig)].sort_values(by = 'oldbalanceOrg').tail(10)


# In[16]:


dataFlagged = df.loc[df.isFlaggedFraud == 1]

print('Minimum Balance of oldBalanceOrig for FlaggedFraud and Transfer mode :', dataFlagged.oldbalanceOrg.min())
print('Maximum Balance of oldbalanceOrig for FlaggedFraud and Transfer mode :', dataFlagged.oldbalanceOrg.max())


# In[17]:


X = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]
X.shape


# In[18]:


X.head()


# In[19]:


sns.boxplot(x='isFraud',y='step',data=X)


# In[20]:


plt.rcParams['figure.figsize'] = (10, 7)
df['amount'].value_counts().head(20).plot.bar()
plt.title('Most Common Transaction amounts')
plt.xlabel('Amounts')
plt.ylabel('Count')
plt.show()


# In[21]:


X['isFraud'].value_counts()


# In[22]:


plt.rcParams['figure.figsize'] =(14, 7)
sns.distplot(df.step, kde = False)
plt.title('Distribution Plot for steps')
plt.xlabel('Step')
plt.show();


# In[23]:


Y = X['isFraud'] # Target Variable

X = X.drop(['isFraud'], axis = 1) # REmoving target variable
print("Shape of x: ", X.shape)
print("Shape of y: ", Y.shape)


# In[24]:


ax = df.groupby(['type', 'isFraud'])['amount'].size().plot(kind='line', color='r')
ax.set_title("Transactions which are conf. Fraud")
ax.set_ylabel("Count of Transactions")
for q in ax.patches:
    ax.annotate(str(format(int(q.get_height()))),(q.get_x(), q.get_height()))


# In[25]:


X.head()


# In[26]:


ax = df.groupby(['type', 'isFlaggedFraud'])['amount'].size().plot(kind='line')
ax.set_title("Transactions which are Flagged Fraud by old sys")
ax.set_ylabel("Count of Transactions")


# In[27]:


Y.head()


# In[28]:


ax = df.groupby(['type', 'isFraud'])['amount'].size().plot(kind='line', color='r')
ax = df.groupby(['type', 'isFlaggedFraud'])['amount'].size().plot(kind='line')
ax.set_title("Comparison btw isFraud and isFlaggedFraud")
ax.set_ylabel("Count of Transactions")


# In[29]:


ax = df.groupby(['type', 'isFlaggedFraud'])['amount'].sum().plot(kind='bar')
ax.set_title("Sum of money lost to fraud as FlaggedFraud")
ax.set_ylabel("Count of Transactions")
for q in ax.patches:
    ax.annotate(str(format(int(q.get_height()))),(q.get_x(), q.get_height()))


# In[30]:


ax = df.groupby(['type', 'isFraud'])['amount'].sum().plot(kind='bar', color='r')
ax.set_title("Sum of money lost to fraud as Fraud")
ax.set_ylabel("Count of Transactions")
for q in ax.patches:
    ax.annotate(str(format(int(q.get_height()))),(q.get_x(), q.get_height()))


# In[31]:


from statsmodels.tools import categorical

tmp = df.loc[(df['type'].isin(['TRANSFER', 'CASH_OUT'])),:]
tmp.drop(['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)
tmp = tmp.reset_index(drop=True)
a = np.array(tmp['type'])
b = categorical(a, drop=True)
tmp['type_num'] = b.argmax(1)


# In[32]:


ax = pd.value_counts(tmp['isFraud'], sort = True).sort_index().plot(kind='bar', title="Fraud transaction count", figsize=(8, 4))
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()))   
ax.set_title("# of transactions break by actual fraud")
ax.set_xlabel("(0 = Not Actual Fraud | 1 = Actual Fraud)")
ax.set_ylabel("Count of transactions")
plt.show()


# In[33]:


from imblearn.under_sampling import RandomUnderSampler

X = df.drop(['isFraud', 'type', 'nameOrig', 'nameDest'], axis = 1)
y = df.isFraud
rus = RandomUnderSampler(sampling_strategy=0.8)
X_res, y_res = rus.fit_resample(X, y)
print(X_res.shape, y_res.shape)
print(pd.value_counts(y_res))


# In[47]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.2, random_state = 121)


# In[48]:


print("Shape of x_train: ", x_train.shape)
print("Shape of y_train: ", y_train.shape)

print("Shape of x_test: ", x_test.shape)
print("Shape of y_test: ", y_test.shape)


# In[49]:


model = LogisticRegression()
model.fit(x_train, y_train)
predictions= model.predict(x_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[50]:


y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print('Accuracy', accuracy_score(y_test, y_pred))


# In[51]:


y_pred = model.predict(X)
print(classification_report(y, y_pred))
print('Accuracy:', accuracy_score(y, y_pred))


# In[52]:


models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))


# In[53]:


cart=DecisionTreeClassifier()
cart.fit(x_train, y_train)
predictions=cart.predict(x_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[55]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=15)
if True:
    probabilities = clf.fit(x_train, y_train.values.ravel()).predict(x_test)


# In[57]:


from sklearn.metrics import average_precision_score
if True:
    print(average_precision_score(y_test,probabilities))


# In[ ]:




