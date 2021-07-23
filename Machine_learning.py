#!/usr/bin/env python
# coding: utf-8

# # Task 2.2: Machine Learning

# In this task a machine learining model which will simulate an A/B test will be created.After we get this model it will be put to the test and will be checked for its accuracy.

# Import much needed **libraries**.

# In[74]:


import os
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import math as mt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
sns.set()
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression


# In[13]:


sns.set()
import warnings
warnings.filterwarnings('ignore')
path='C:/Users/chuna/Downloads/week4/AdSmartABdata.csv'
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


# Define a function that plots **Tested** data points vs **Pridicted** ones.

# In[43]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

def plot_preds(y_test, y_preds, model_name):
    N = len(y_test)
    plt.figure(figsize=(20,15))
    original = plt.scatter(np.arange(1, N+1), y_test, c='blue')
    prediction = plt.scatter(np.arange(1, N+1), y_preds, c='red')
    plt.xticks(np.arange(1, N+1))
    plt.xlabel('# Oberservation')
    plt.ylabel('Enrollments')
    title = 'True labels vs. Predicted Labels ({})'.format(model_name)
    plt.title(title)
    plt.legend((original, prediction), ('Original', 'Prediction'))
    plt.show()


# Define a function that calculates the **metrics** of accuracy.

# In[44]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def calculate_metrics(y_test, y_preds):
    rmse = np.sqrt(mean_squared_error(y_test, y_preds))
    r_sq = r2_score(y_test, y_preds)
    mae = mean_absolute_error(y_test, y_preds)

    print('RMSE Score: {}'.format(rmse))
    print('R2_Squared: {}'.format(r_sq))
    print('MAE Score: {}'.format(mae))


# **Import** data as csv

# In[45]:


auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
downloaded = drive.CreateFile({'id':"1YSn01vvlHKQaAIBtwIXRNd-oTaTuDN09"}) 
downloaded.GetContentFile('ABAdRecall.csv')
import pandas as pd
data=pd.read_csv('ABAdRecall.csv')
data.head()
df=data


# In[19]:


df_grouped = df.groupby('experiment').agg(yes=('yes', 'sum'),no=('no','sum'))

df_grouped['total']=df_grouped['yes']+df_grouped['no']#################remove33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
df_grouped


# In[46]:


df_control=df.loc[df['experiment'] == 'control']
df_exposed=df.loc[df['experiment'] == 'exposed']
df_control


# In[64]:


df_exposed_clean=df_exposed.drop(df_exposed[(df_exposed.yes == 0) & (df_exposed.no == 0)].index)
df_control_clean=df_control.drop(df_control[(df_control.yes == 0) & (df_control.no == 0)].index)
df_exposed_clean


# Check for any **NaN** values

# In[65]:


df_control_clean.isnull().values.any()


# In[ ]:


df_control_clean.isna().sum()


# In[49]:


df_exposed_clean.isna().sum()


# # Data Wrangling

# In[68]:


data_total = pd.concat([df_exposed, df_control])
data_total


# In[69]:


data_total=data_total.drop(data_total[(data_total.yes == 0) & (data_total.no == 0)].index)
data_total


# Add day as a column(**DOW**).

# In[70]:


data_total["experiment"].replace({"exposed": 1, "control": 0}, inplace=True)
data_total['date'] = pd.to_datetime(data_total['date']) 
data_total['DOW'] = data_total['date'].dt.day_name() 
  


data_total.sample(10)


# In[52]:


data_total.info()


# Remove `Date` and any missing data.Then shuffle the data set.

# In[71]:


import numpy as np
np.random.seed(7)
import sklearn.utils

# Remove missing data
data_total.dropna(inplace=True)

# Remove Date columns
del data_total['date']

# Shuffle the data
data_total = sklearn.utils.shuffle(data_total)


# In[72]:


data_total


# Change all independent variables to numerical values.They will be **labelled** a new number as per their previous entry.

# In[73]:


data_total = data_total[['auction_id', 'hour', 'device_make', 'platform_os', 'browser', 'yes', 'no','experiment']]
data_total['row_id'] = data_total.index
data_total=data_total.drop(['auction_id'],axis=1)
data_total=data_total.drop(['device_make'],axis=1)
lb = LabelEncoder()
data_total['browser'] = lb.fit_transform(data_total['browser'])

data_total


# # Split data into Training ,Testing and Validatting

# Since we are going to pridict for **yes**,it is separated from the rest.Also **`row_id`** and **`no`** columns are because they are irrelevant for our prediction.**Row_id** is randomly assigned and shouldnt stir our prediction.

# On the other hand the **`no`** column is removed because had we known its value this all prediction would be to no avail.

# In[77]:



survived =  data_total['yes']                                       #data_total.iloc[:, [5]]
titanic = data_total.drop(['yes'],axis=1)
titanic = titanic.drop(['row_id'],axis=1)
titanic = titanic.drop(['no'],axis=1)

x_train, x_test, y_train, y_test = train_test_split(titanic, survived, test_size = 0.2, random_state=42)
titanic


# In[78]:


survived


# Next we are going to create different models and take the best one using **GridSearch.**

# In[82]:


# Create logistic regression object
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

logistic = linear_model.LogisticRegression()
# Create a list of all of the different penalty values that you want to test and save them to a variable called 'penalty'
penalty = ['l1', 'l2']
# Create a list of all of the different C values that you want to test and save them to a variable called 'C'
C = [0.0001, 0.001, 0.01, 1, 100]
# Now that you have two lists each holding the different values that you want test, use the dict() function to combine them into a dictionary. 
# Save your new dictionary to the variable 'hyperparameters'
hyperparameters = dict(C=C, penalty=penalty)
# Fit your model using gridsearch
clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
best_model = clf.fit(X, Y)
#Print all the Parameters that gave the best results:
print('Best Parameters',clf.best_params_)
# You can also print the best penalty and C value individually from best_model.best_estimator_.get_params()
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
best_model


# After we have the best model we will predict with it.We can seehow many we missed using **Confusion matrix.**

# In[83]:


from sklearn.metrics import confusion_matrix
y_preds = best_model.predict(x_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_preds)

print(confusion_matrix)


# Our **accuracy** is follows

# In[84]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_preds, normalize=True)


# The raw **Pridicted** data is shown below.

# In[ ]:


y_test=np.array(y_test)
y_test


# In[85]:


calculate_metrics(y_test, y_preds)


# Plot of **predicted** data and **test** data.

# In[86]:


plot_preds(y_test, y_preds, 'Logistic Regression')


# # Model 02: Decision Tree

# In[86]:


For decision tree we will use pre determined parameters which consider K-fold as 5.


# In[111]:


from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
  
    # Performing training 
clf_gini.fit(x_train, y_train) 
y_preds=clf_gini.predict(x_test)


# We also calculate here accuracy and its lower than **Logistic regression.**

# In[112]:


accuracy_score(y_test, y_preds, normalize=True)


# In[110]:


plot_preds(y_test, y_preds, 'Decision Tree')


# In[113]:


calculate_metrics(y_test, y_preds)


# # XGBoost

# In[114]:


import xgboost as xgb


# Lets refresh our memory

# In[115]:


x_train


# In[116]:


y_train


# In[94]:


x_train.info()


# In[95]:


DM_train = xgb.DMatrix(data=x_train,label=y_train)
DM_test = xgb.DMatrix(data=x_test,label=y_test)


# The below parameters are considering k fold-5.

# In[117]:


parameters = {
    'max_depth': 6,
    'objective': 'reg:linear',
    'booster': 'gblinear',
    'n_estimators': 1000,
    'learning_rate': 0.2,
    'gamma': 0.01,
    'random_state': 7,
    'subsample': 1.
}


# In[119]:


from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_train, y_train)
print(model)
# make predictions for test data
y_preds = model.predict(x_test)


# In[120]:


accuracy_score(y_test, y_preds, normalize=True)


# In[121]:


plot_preds(y_test, y_preds, 'XGBoost')


# # Feature importance

# Which feature had the highest impact on our **"yes"** values. 

# In[122]:


from xgboost import XGBClassifier, plot_importance
plot_importance(model)


# In[ ]:


Discussion about usage of A/B testing, k-fold and Machine learning approach are discused on the paper.
 


# # Cheers!!!
