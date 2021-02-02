#!/usr/bin/env python
# coding: utf-8

# # Artificial Intelligence and Machine Learning | Live Project

# ## Topic: Breast Cancer Prediction 

# ### by Anurag Sen | as5864@srmist.edu.in 

# In[91]:


import warnings
warnings.filterwarnings('ignore')


# In[92]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[93]:


df = pd.read_csv("https://raw.githubusercontent.com/ingledarshan/AIML-B2/main/data.csv")


# In[94]:


df.head(10)


# In[95]:


df.info()


# In[96]:


df['Unnamed: 32']


# In[97]:


df.drop("Unnamed: 32", axis = 1, inplace = True)  #use axis = 1 for columns and axis = 0 for rows


# In[98]:


df


# In[99]:


df.columns


# In[100]:


df.drop("id", axis = 1 , inplace = True) #as id number is not involved in disease prediction


# In[101]:


df


# In[102]:


l = list(df.columns)
print(l)


# In[103]:


#to sort out the mean, se and worst data

features_mean = l[1:11]
features_se = l[11:21]
features_worst = l[21:]


# In[104]:


print(features_mean)


# In[105]:


print(features_se)


# In[106]:


print(features_worst)


# In[107]:


df['diagnosis'].unique()  #so the Diagnosis column has 2 types of cancers. M =  Malignant , B = Benign


# In[108]:


sns.countplot(df['diagnosis']) #analysis of Malignant and Benign cases


# In[109]:


df['diagnosis'].value_counts()  #357 cases of Benign cancer and 212 cases of Malignant cancer


# ## Exploring the given data  

# In[110]:


df.describe()   #for the radius_mean, the mean and median(50%) value are near around. thus this data point is very effective to predict.
#Standard deviation of the data is the spread of the data from the mean.


# In[111]:


len(df.columns)


# In[112]:


#Correlation plot

corr = df.corr()
corr


# In[113]:


plt.figure(figsize=(8,8))
sns.heatmap(corr)   #The more the brighter the plot, the higher is the correlation of the quantities


# In[114]:


df.head()


# In[115]:


df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})


# In[119]:


df


# In[121]:


df['diagnosis'].unique()


# In[125]:


#diagnosis is the prediction of the disease, so it is the result of the process. 
#the input data will be everything apart from the diagnosis
X = df.drop('diagnosis', axis = 1)
X


# In[127]:


y = df['diagnosis']
y


# In[130]:


#splitting the data into Train and Test, we know X = input, Y = output. we keep 70% of the data for training and the rest for testing.

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)


# In[132]:


#Scaling the margins of the data to prevent mimatching of the boundaries. We tend to compact the data near around some value.
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)   #fit_transform as it will learn the behaviour of the data
X_test = ss.transform(X_test)         #transform as it will directly perform on a new data


# ## Machine Learning Models 

# ### 1. Logistic Regression 

# In[232]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[233]:


y_pred = lr.predict(X_test)


# In[234]:


y_pred


# In[235]:


y_test


# In[236]:


from sklearn.metrics import accuracy_score
print("Accuracy of logistic regression is: " , accuracy_score(y_test, y_pred))


# In[237]:


lr_acc = accuracy_score(y_test, y_pred)


# In[238]:


results = pd.DataFrame()   #creating an empty data frame
results


# In[239]:


tempResults = pd.DataFrame({'Algorithm':['Logistic Regression Method'], 'Accuracy':[lr_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# ### 2. Decision Tree Classifier 

# In[240]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)


# In[241]:


y_pred = dtc.predict(X_test)
y_pred


# In[242]:


from sklearn.metrics import accuracy_score
print("Accuracy of Decision Tree Classifier is: ", accuracy_score(y_test, y_pred))


# In[243]:


dtc_acc = accuracy_score(y_test, y_pred)


# In[244]:


tempResults = pd.DataFrame({'Algorithm':['Decision tree Classifier Method'], 'Accuracy':[dtc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# ### 3. Random Forest Classifier

# In[245]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)


# In[246]:


y_pred = rfc.predict(X_test)
y_pred


# In[247]:


from sklearn.metrics import accuracy_score
print("Accuracy of Random Forest Classifier is: ",accuracy_score(y_test, y_pred))


# In[248]:


rfc_acc = accuracy_score(y_test, y_pred)


# In[249]:


tempResults = pd.DataFrame({'Algorithm':['Random Forest Classifier Method'], 'Accuracy':[rfc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# ### 4. Support Vector Machines 

# In[250]:


from sklearn import svm
svc = svm.SVC()
svc.fit(X_train, y_train)


# In[251]:


y_pred = svc.predict(X_test)
y_pred


# In[252]:


from sklearn.metrics import accuracy_score
print("Accuracy of SVM is: ",accuracy_score(y_test, y_pred))


# In[253]:


svc_acc = accuracy_score(y_test, y_pred)


# In[254]:


tempResults = pd.DataFrame({'Algorithm':['Support Vector Classifier Method'], 'Accuracy':[svc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]


# In[255]:


results


# ### Conclusion
# Thus, we employed 4 Machine Learning Algorithms on the same data set and obtained the result that Logistic Regression method and Support Vector Classifier method has the highest accuracy.

# In[ ]:




