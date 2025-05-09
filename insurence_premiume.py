#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import statsmodels.api as sm


# In[2]:


data = pd.read_csv("C:/Users/govar/Downloads/archive/insurance.csv")


# In[3]:


data.head()


# In[4]:


data.info


# In[5]:


data.describe()


# In[6]:


data.describe(include='object')


# In[7]:


data.dtypes


# In[8]:


data.isnull().sum()


# In[9]:


data['sex'].unique()


# In[10]:


data['region'].unique()


# In[11]:


data.head()


# In[12]:


data = pd.get_dummies(data=data, drop_first=True)


# In[13]:


data['sex_male'] = data['sex_male'].astype(int)
data['smoker_yes'] = data['smoker_yes'].astype(int)
data['region_northwest'] = data['region_northwest'].astype(int)
data['region_southeast'] = data['region_southeast'].astype(int)
data['region_southwest'] = data['region_southwest'].astype(int)

data.head()


# In[14]:


data.shape


# In[15]:


train_data = data.drop(columns='expenses')


# In[16]:


train_data.head()


# In[17]:


train_data.corrwith(data['expenses']).plot.bar(
 figsize=(16,9), title='correlation with charges', rot=45, grid=True
)


# In[18]:


corr = data.corr()


# In[19]:


plt.figure(figsize=(16, 9))
sns.heatmap(corr, annot=True)


# In[20]:


x = data.drop(columns='expenses')
y = data['expenses']


# In[21]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[22]:


x_train.shape


# In[23]:


y_train.shape


# In[24]:


x_test.shape


# In[25]:


y_test.shape


# In[26]:


pip install sklearn.preprocessing


# In[27]:


pip install scikit-learn


# In[ ]:





# In[29]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[30]:


x_train


# In[31]:


x_test


# In[32]:


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)


# In[33]:


y_pred = regression.predict(x_test)


# In[34]:


from sklearn.metrics import r2_score


# In[35]:


r2_score(y_test, y_pred)


# # Random forest regression

# In[36]:


from sklearn.ensemble import RandomForestRegressor
regression_rf = RandomForestRegressor()
regression_rf.fit(x_train, y_train)


# In[37]:


y_pred = regression_rf.predict(x_test)


# In[38]:


r2_score(y_test, y_pred)


# In[ ]:





# In[39]:


data.head()


# In[40]:


frank = [[18,33.8,1,1,0,0,1,0]]


# In[41]:


regression_rf.predict(sc.transform(frank))


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import gradio as gr


data = pd.read_csv("C:/Users/govar/Downloads/insurance.csv")
data = pd.get_dummies(data=data, drop_first=True)

data['sex_male'] = data['sex_male'].astype(int)
data['smoker_yes'] = data['smoker_yes'].astype(int)
data['region_northwest'] = data['region_northwest'].astype(int)
data['region_southeast'] = data['region_southeast'].astype(int)
data['region_southwest'] = data['region_southwest'].astype(int)

columns_to_drop = ['Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10']
for col in columns_to_drop:
    if col in data.columns:
        data = data.drop(columns=col)

x = data.drop(columns='prediction')
y = data['prediction']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

regression_rf = RandomForestRegressor()
regression_rf.fit(x_train, y_train)

def predict_insurance(age, bmi, charges, children, sex_male, smoker_yes, region):
    region_northwest = 0
    region_southeast = 0
    region_southwest = 0
    
    if region == "Northwest":
        region_northwest = 1
    elif region == "Southeast":
        region_southeast = 1
    elif region == "Southwest":
        region_southwest = 1

    input_data = [[age, bmi, children, charges, sex_male, smoker_yes, region_northwest, region_southeast, region_southwest]]
    input_data_scaled = sc.transform(input_data)
    prediction = regression_rf.predict(input_data_scaled)
    return prediction[0]


inputs = [
    gr.Slider(minimum=18, maximum=100, step=1, label="Age"),
    gr.Slider(minimum=10, maximum=50, label="BMI"),
    gr.Slider(minimum=1000, maximum=30000, label="Charges"),  # Renamed to match the function parameter
    gr.Number(label="Children"),
    gr.Radio(choices=[0, 1], label="Sex (Male=1, Female=0)"),
    gr.Radio(choices=[0, 1], label="Smoker (Yes=1, No=0)"),
    gr.Radio(choices=["Northwest", "Southeast", "Southwest"], label="Region"),
]

outputs = gr.Textbox(label="Predicted Insurance Premium")

interface = gr.Interface(
    fn=predict_insurance,
    inputs=inputs,
    outputs=outputs,
    title="Insurance Premium Prediction",
    description="Enter the required details to predict insurance premium.",
    theme = "darkhuggingface",
)

interface.launch(share=True)


# In[ ]:




