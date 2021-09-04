import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

dataset = pd.read_csv('C:/Users/yashb/Downloads/PS_20174392719_1491204439457_log.csv') 
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
dataset.isnull().values.any()
steps = dataset['step'].value_counts().nunique()
print(steps)
plt.rcParams['figure.figsize'] = (14, 7)
sns.distplot(dataset.step, kde = False)
plt.title('Distribution Plot for steps')
plt.xlabel('Step')
plt.show();
plt.rcParams['figure.figsize'] = (10, 7)
dataset['amount'].value_counts().head(20).plot.bar()
plt.title('Most Common Transaction amounts')
plt.xlabel('Amounts')
plt.ylabel('Count')
plt.show()
oc_isFraud = dataset['isFraud'].value_counts()
oc_isFraud
oc_isFlaggedFraud= dataset['isFlaggedFraud'].value_counts()
oc_isFlaggedFraud 
fraud_type= dataset.groupby("type")["isFraud"].count()
fraud_type
#isFraud (conf)
ax = dataset.groupby(['type', 'isFraud'])['amount'].size().plot(kind='line', color='r')
ax.set_title("Transactions which are conf. Fraud")
ax.set_ylabel("Count of Transactions")

#isFlaggedFraud (conf)
ax = dataset.groupby(['type', 'isFlaggedFraud'])['amount'].size().plot(kind='line')
ax.set_title("Transactions which are Flagged Fraud by old sys")
ax.set_ylabel("Count of Transactions")

#comparison btw isFraud and isFlaggedFraud
ax = dataset.groupby(['type', 'isFraud'])['amount'].size().plot(kind='line', color='r')
ax = dataset.groupby(['type', 'isFlaggedFraud'])['amount'].size().plot(kind='line')
ax.set_title("Comparison btw isFraud and isFlaggedFraud")
ax.set_ylabel("Count of Transactions")

#Sum of money lost to fraud as FlaggedFraud
ax = dataset.groupby(['type', 'isFlaggedFraud'])['amount'].sum().plot(kind='bar')
ax.set_title("Sum of money lost to fraud as FlaggedFraud")
ax.set_ylabel("Count of Transactions")
for q in ax.patches:
    ax.annotate(str(format(int(q.get_height()))),(q.get_x(), q.get_height()))
    
#Sum of money lost to fraud as Fraud
ax = dataset.groupby(['type', 'isFraud'])['amount'].sum().plot(kind='bar', color='r')
ax.set_title("Sum of money lost to fraud as Fraud")
ax.set_ylabel("Count of Transactions")
for q in ax.patches:
    ax.annotate(str(format(int(q.get_height()))),(q.get_x(), q.get_height()))
    
print("Minimum Transaction :", dataset.loc[dataset.isFlaggedFraud == 1].amount.min())
print("Maximum Transaction :", dataset.loc[dataset.isFlaggedFraud == 1].amount.max())

dataTransfer = dataset.loc[dataset['type'] == 'TRANSFER']
dataTransfer = pd.DataFrame(dataTransfer)
dataTransfer.head(20)
