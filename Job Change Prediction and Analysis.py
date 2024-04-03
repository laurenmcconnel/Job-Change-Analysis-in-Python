#!/usr/bin/env python
# coding: utf-8

# # Job Change Prediction and Analysis
# The Dataset which I chose to analyse was resourced from the following link
# 'https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists/data'

#To start the analysis I imported some essential libraries
import pandas as pd
import numpy as np

# In[95]:
#Then, I imported the Dataset 
inPath = '/Users/laure/OneDrive/Documents/Year 4/'
inpDfRaw = pd.read_csv(inPath + 'Job_Change_Prediction.csv', 
                    sep=',',header = 0, index_col=0)
inpDfRaw

# In[96]:
#Next, I counted the null values per column in the Dataset
np.nan
inpDfRaw.isna().sum(axis = 0)

# There was quite a high number of NaN values, particularly in the columns of 'gender', 'company size' and 'company type'.
# The decrepency in the 'company' related columns is likely due to the employees with 'no relevent experience' having no values for these values.
# As a result I decided to remove the company size and type columns.
# I also decided to remove the gender column due to the large amount of NaN values which removed a lot of data. 

# In[119]:
#Below I removed the stated columns
inpDf = inpDfRaw.drop(['company_size','company_type','gender'], axis=1)
inpDf


# Next, I removed all rows with null values, within the other columns, this allowed me to keep way more rows for a bigger sample size. However, this may limit me in some aspects of the model.

# In[120]:
#Removing rows containing NaN values
inpDf = inpDf.dropna(axis = 0,how = 'any')
inpDf


# In[ ]:
#I need to change all the following datatypes to a numerical value
inpDf.dtypes

# In[122]:
print('Removed Rows = '+str((inpDfRaw['city'].count())-(inpDf['city'].count())))

# After this I identified the columns with categorical values and converted them to numerical values. Starting with the 'relevent_experience' column.  

# In[123]:
inpDf['relevent_experience'].unique()


# This column has two unique values as seen above. This is an interval variable. We should use label encoding to convert these to a numerical value.

# In[124]:
encoding_dict = {'No relevent experience': 0, 
                 'Has relevent experience': 1}
inpDf['relevent_experience'] = inpDf['relevent_experience'].map(encoding_dict)
inpDf


# I did the same for the 'enrolled_university' 'education_level' columns 

# In[125]:
inpDf['enrolled_university'].unique()

# In[126]:
encoding_dict = {'no_enrollment': 0, 
                 'Full time course': 1,
                'Part time course':2}
inpDf['enrolled_university'] = inpDf['enrolled_university'].map(encoding_dict)
inpDf


# In[64]:
inpDf['education_level'].unique()


# In[127]:
encoding_dict = {'Graduate': 0, 
                 'Masters': 1,
                'Phd':2}
inpDf['education_level'] = inpDf['education_level'].map(encoding_dict)
inpDf


# The major_discipline column has 6 unique values and as a result I feel it is a categorical but Interval Variable.
# We use one-hot-encoding to convert it to a  dummy variable

# In[103]:
inpDf['major_discipline'].unique()


# In[128]:
inpDf=pd.get_dummies(inpDf, columns = ['major_discipline'])
inpDf

# In[105]:
inpDf.dtypes
# The remaining data to convert are the 'city', 'experience' and 'last_new_job' columns

# In[129]:
#The city column need the string of 'city_' removed, and needs to be converted to numerical values
inpDf['city'] = inpDf['city'].str.replace('city_','')
inpDf['city'] = pd.to_numeric(inpDf['city'], errors='coerce')
inpDf


# In[131]:
#The experience column need the strings of '>' and '<' removed, and needs to be converted to numerical values
inpDf['experience'].unique()


# In[133]:
inpDf['experience'] = inpDf['experience'].str.replace('[^0-9]', '', regex=True).astype(int)
inpDf


# In[107]:
#The lasy new job column need the strings of '>' and 'never' removed, and needs to be converted to numerical values
#For ease of analysis I will take the >4 to be 5, and 0 to be 'never'
inpDf['last_new_job'].unique()


# In[134]:
# I will convert categorical values to numerical using encoding dictionary
mapping = {'never': 0, '>4': 5}
inpDf['last_new_job'] = inpDf['last_new_job'].replace(mapping)
inpDf['last_new_job'] = inpDf['last_new_job'].astype(int)
inpDf


# In[135]:
inpDf.dtypes
#Now all data is a numerical value 


# Now I will attempt to standardize the data.
# This will help us to test the accuracy of the data later. 

# In[137]:


from scipy.stats import zscore
numerical_columns = ['city', 'city_development_index', 'experience', 'last_new_job', 'training_hours']
inpDf[numerical_columns] = inpDf[numerical_columns].apply(zscore)
inpDf


# In[134]:

#These columns must be standardized as they are at a much higher metric than the rest of the data in the dataset which lie between 0 and 2.

 
# In supervised learning, the data and the result is known, we are attempting to train a model on existing data in order for it to be able to be used on unseen data and perform the same actions. 
# We split the data into train and test and check its accuracy using different models 

# In unsupervised learning we do not know the result of the data, we put our data into the model and it makes conclusions for us to evaluate.

# Supervised learning is preferred as there is a clear outcome in this case.
# We want to find out, using the other data, whether the employee will resign. 

# In[140]:
#Here I am splitting the data into yDf the data we use to predict, and xDf the outcome (whether the person does not change job (0) or changes job (1) )
yDf = inpDf['target']
xDf = inpDf.drop(columns = 'target')


# We must split our values into train and test sets in order to train our model, but to not overfit it to the dataset we have a test section.
# The train set receieves the larger set of samples (usually around 70%) as this is how the model will learn. 
# We then test is using the remaining data (usually around 30%)

# In[143]:
from sklearn.model_selection import train_test_split


# In[144]:
X_train, X_test, y_train, y_test = train_test_split(xDf,yDf,test_size=0.3,random_state=0)

# Now, I tested the Dataset using the Support Vector Machine (SVM)

# The support vector machine can be used for classification or regression, or outlier detection. 
# It transforms data using the kernal function. It allows us to see another plain which is not visible on the x and y axis, the z axis 


# In[145]:
from sklearn import svm
from sklearn.metrics import accuracy_score

svc = svm.SVC(C=100, kernel = 'rbf',gamma = 1, random_state=0)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'SVC Accuracy: {accuracy}')


# Lastly, I tested the Dataset using the Random forests Classifier 
# 
# Random forests increase the accuracy of decision trees by combining multiple of them to reach one result. 
# They test multiple features at each node and choose based on how well they fit. 

# In[147]:
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 100 , random_state=0)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy: {accuracy}')

# Finally, I conducted Linear regression on my dataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred = linear_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


