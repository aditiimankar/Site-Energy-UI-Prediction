#!/usr/bin/env python
# coding: utf-8

# # import libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import matplotlib.style as style


# In[4]:


get_ipython().system('pip install xgboost')


# In[7]:


get_ipython().system('pip install category_encoders')


# In[15]:


get_ipython().system('pip install catboost')


# In[16]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# In[17]:


from category_encoders import TargetEncoder
from catboost import CatBoostRegressor


# In[18]:


import matplotlib.style as style
pd.set_option('display.max_columns', 70)
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False
style.use('ggplot')
sns.set_palette('Set2')

import warnings
warnings.filterwarnings('ignore')


# In[19]:


import tqdm
import joblib


# In[20]:


get_ipython().system('pip install shap')


# In[21]:


import shap


# In[22]:


import warnings
warnings.filterwarnings('ignore')


# # Import Dataset

# In[26]:


train = pd.read_csv("C:/Users/manka/OneDrive/Desktop/train_dataset.csv")
test = pd.read_csv("C:/Users/manka/OneDrive/Desktop/x_test.csv")


# In[27]:


df_train = train.copy()
df_test = test.copy()


# In[28]:


print(df_train.shape)


# In[29]:


print(df_test.shape)


# In[30]:


df_train.rename(columns={
    'Year_Factor': 'year_factor', 
    'State_Factor': 'state_factor',
    'ELEVATION': 'elevation',
    'id': 'building_id'
}, inplace=True)

df_test.rename(columns={
    'Year_Factor': 'year_factor', 
    'State_Factor': 'state_factor',
    'ELEVATION': 'elevation',
    'id': 'building_id'
}, inplace=True)


# In[31]:


train


# ### test

# # Exploratory Data Analysis

# In[32]:


df_train.info()


# In[33]:


#finding duplicates
print(df_train.duplicated().sum())
print(df_test.duplicated().sum())


# In[34]:


# Print columns in df_train that have a unique value in every row
print([col for col in df_train if df_train[col].nunique()==1])

# Print columns in df_test that have a unique value in every row

print([col for col in df_test if df_test[col].nunique()==1])


# In[35]:


print([col for col in df_train if df_train[col].nunique()==1])
print([col for col in df_test if df_test[col].nunique()==1])


# In[36]:


def missing_values_table(df):
        # Total missing values by column
        mis_val = df.isnull().sum()
        
        # Percentage of missing values by column
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # build a table with the thw columns
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[37]:


print("Train set columns with null values: ")
print(list(df_train.columns[df_train.isnull().any()]))
print('===========================================')
# Missing values for training data
missing_values_train = missing_values_table(df_train)
missing_values_train[:20].style.background_gradient(cmap='Reds')


# In[38]:


print("Test set columns with null values: ")
print(list(df_test.columns[df_test.isnull().any()]))
print('===========================================')
# Missing values for test data
missing_values_test = missing_values_table(df_test)
missing_values_test[:20].style.background_gradient(cmap='Reds')


# In[39]:


# Selecting specific columns from df_test DataFrame

df_test[['year_factor', 'days_above_110F']]


# In[40]:


### combine the datasets for the visualizations


# In[41]:


# Set the 'site_eui' column in the 'test' DataFrame to NaN (missing value)
df_test['site_eui'] = np.nan
# Set the 'dataset' column in the 'test' DataFrame to NaN (missing value)
df_test['dataset'] = 'test'
# Set the 'dataset' column in the 'test' DataFrame to NaN (missing value)
df_train['dataset'] = 'train'


df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)


#     The dataset contains time variable, `Year_Factor`, that has values from 1 to 7. Assuming the values are consecutive years,`train` dataset contains the first 6 years and `test` set contains the 7th year.

# In[42]:


grouped_data = df_all.groupby(['dataset', 'year_factor']).size().reset_index(name='Count')

# Create the bar plot
ax = sns.barplot(x='year_factor', y='Count', hue='dataset', data=grouped_data)
ax.set(title="The number of data points by year", ylabel="Count")
plt.show()


# In[43]:


# Assuming you have a DataFrame called df_all with columns 'dataset' and 'State_Factor'

# Group the data by 'dataset' and 'State_Factor' and calculate the count of records
grouped_data = df_all.groupby(['dataset', 'state_factor']).size().reset_index(name='Count')

# Create the relational plot
ax = sns.relplot(
    x='state_factor',
    y='Count',
    data=grouped_data,
    hue='dataset',
    style='dataset',
    aspect=2,
    height=4,
    s=50,
    alpha=0.9
).set(
    title="The number of data points by States",
    ylabel=None
)

plt.show()


# The datasets, comprising both residential and commercial buildings, exhibit variations in their composition. The 'train' dataset predominantly consists of residential buildings, with a notable contribution from 'State_6'. In contrast, the number of commercial buildings surpasses residential ones overall, with 'State_10' exclusively comprising commercial structures. This distinction in building type, along with the specific state, appears significant in determining Energy Use Intensity (EUI). Further exploration of these factors could shed light on the relationship between building type, state, and EUI.
# 

# In[44]:


fig, ax = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
fig.suptitle("The number of data points by dataset")

# Plot 1: By building class
sns.barplot(
    x='dataset',
    y='Count',
    data=df_all.groupby(['dataset', 'building_class']).size().reset_index(name='Count'),
    hue='building_class',
    ax=ax[0],
    ci=False
)
ax[0].set_title("By building class")
ax[0].set_ylabel("")
ax[0].set_xlabel("")

# Plot 2 and 3: By building class and state
for e, s in enumerate(df_all['building_class'].unique(), 1):
    sns.barplot(
        x='state_factor',
        y='Count',
        data=df_all[df_all['building_class'] == s].groupby(['dataset', 'state_factor']).size().reset_index(name='Count'),
        hue='dataset',
        ax=ax[e],
        ci=False
    )
    ax[e].set_title(s)
    ax[e].set_ylabel("")
    ax[e].set_xlabel("")

plt.show()


# The presence of various facility types in both the training and test datasets is observed. A notable concentration of residential buildings labeled as Multifamily_uncategorized and Office_uncategorized in State_6 within the training set is highlighted. The potential impact of this state on the model learning and prediction process is acknowledged, particularly regarding the generalizability of the model to other states or regions. Careful consideration of this potential bias and evaluation of the model's performance across different states are essential to ensure accurate and reliable predictions

# In[45]:


ax = sns.catplot(x='facility_type',
                 kind='count',
                 hue='dataset',
                 data=df_all,
                 height=5,
                 aspect=3,
                 palette='Set1')

ax.set_xticklabels(rotation=90)
ax.set(title="The number of data points by Facility type",
       ylabel="Count")


# The oldest building was built in 1600 and the latest in 2016. The majority of the buildings were built since 1900. There were some 0 and null values. Not quite sure what 0 signifies.

# In[46]:


df_all['year_built'].value_counts().index.sort_values()


# In[47]:


temp = df_all[['year_built']].fillna(2029).replace({0:2029}).astype('category').value_counts().reset_index().rename({0:'count'},axis=1)\
            .sort_values('year_built')
# temp['year_built'] = temp['year_built'].astype('category')
fig, ax = plt.subplots(figsize=(15,5))

ax=plt.bar(temp['year_built'],
           temp['count']
          )

fig.suptitle(f"The year built min: {min(temp['year_built'])}, max: {max(df_all['year_built'])}");


# `train` set buildings have higher floor areas compared to `test` set buildings and small positive correlation between `floor_area` and `energy_star_rating` can be observable. 

# In[48]:


sns.pairplot(df_all,
             vars=['energy_star_rating', 'floor_area','elevation'],
             hue='dataset',
             height=4,
             plot_kws={'alpha': 0.4, 's': 30, 'edgecolor': 'k'},
             corner=True
            )


# According to the average temperature (`avg_temp`), if we list states from warmest to coldest: State 1, State 10, State 2, and State 8. The range of temperatures of State1, State 6,  State 11 and State 4 are higher compared to the other states. 

# In[49]:


fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(data=df_all[['avg_temp','state_factor']].drop_duplicates(), y='avg_temp', x='state_factor');


# Each month has unique temperature values between 31 and 59 which means the values in temperature volumes are highly repeated for the data points.

# In[50]:


cols = [['january_min_temp', 'january_avg_temp', 'january_max_temp'],
        ['february_min_temp', 'february_avg_temp', 'february_max_temp'],
        ['march_min_temp', 'march_avg_temp', 'march_max_temp'], 
        ['april_min_temp','april_avg_temp', 'april_max_temp'], 
        ['may_min_temp', 'may_avg_temp','may_max_temp'], 
        ['june_min_temp', 'june_avg_temp', 'june_max_temp'],
        ['july_min_temp', 'july_avg_temp', 'july_max_temp'], 
        ['august_min_temp','august_avg_temp', 'august_max_temp'], 
        ['september_min_temp','september_avg_temp', 'september_max_temp'], 
        ['october_min_temp','october_avg_temp', 'october_max_temp'], 
        ['november_min_temp','november_avg_temp', 'november_max_temp'], 
        ['december_min_temp','december_avg_temp', 'december_max_temp']]
        
fig, ax = plt.subplots(2, 6, figsize=(20,6), sharey=True)
fig.suptitle("Monthly temperature and number of unique values")

for e, c in enumerate(cols):
    if e<=5:
        sns.histplot(df_all[c].drop_duplicates(), ax=ax[0,e], legend=False)\
        .set(title=c[0][:c[0].find('_')]+ '_#'+str(len(df_all[c[0]].unique())))
    else:
        sns.histplot(df_all[c].drop_duplicates(), ax=ax[1,e-6], legend=False)\
        .set(title=c[0][:c[0].find('_')]+ '_#'+str(len(df_all[c[0]].unique())))
        
plt.subplots_adjust(hspace=0.4)


# Other weather related numerical columns also have few unique values.

# In[51]:


cols=['cooling_degree_days','heating_degree_days', 'precipitation_inches', 'snowfall_inches',
       'snowdepth_inches', 'avg_temp', 'days_below_30F', 'days_below_20F',
       'days_below_10F', 'days_below_0F', 'days_above_80F', 'days_above_90F',
       'days_above_100F', 'days_above_110F', 'direction_max_wind_speed',
       'direction_peak_wind_speed', 'max_wind_speed', 'days_with_fog']

fig, ax = plt.subplots(6,3, figsize=(15,18), sharey=True)
fig.suptitle("Numerical variables and the number of unique values")

for e, c in enumerate(cols):
    if e<=5:
        sns.histplot(df_all[c].drop_duplicates(), ax=ax[e,0], legend=False)\
        .set(title=c+"_#"+str(len(df_all[c].unique())), ylabel=None, xlabel=None)
    elif (e>=6) & (e<=11):
        sns.histplot(df_all[c].drop_duplicates(), ax=ax[e-6,1], legend=False)\
        .set(title=c+"_#"+str(len(df_all[c].unique())), ylabel=None, xlabel=None)
    else:
        sns.histplot(df_all[c].drop_duplicates(), ax=ax[e-12,2], legend=False)\
        .set(title=c+"_#"+str(len(df_all[c].unique())), ylabel=None, xlabel=None)
        
plt.subplots_adjust(hspace=0.7)


# In[ ]:





# # Target Variable Exploration (EUI)

# In[52]:


plt.figure(figsize=(10,5))
ax1 = plt.subplot(1,2,1)
cp = sns.histplot(x=df_all['site_eui'], kde=True, palette='Set2')
ax1.set_xlabel('Target Histogram', fontsize=14)
ax2 = plt.subplot(1,2,2)
sns.boxplot(y=df_all['site_eui'], palette='Set2')
ax2.set_xlabel('Target boxplot', fontsize=14)
plt.tight_layout();


# State 2 and 4 have slightly higher EUI and State 11 and 8 have lower EUI level.

# In[53]:


fig, ax = plt.subplots(1,2, figsize=(20,5))
fig.suptitle("EUI by States")

sns.histplot(df_all, x='site_eui',hue='state_factor', ax=ax[0])\
.set(title='EUI by State', ylabel=None)

sns.histplot(df_all[df_all['state_factor']!='State_6'], x='site_eui',hue='state_factor', ax=ax[1])\
.set(title='EUI by State (State 6 removed)', ylabel=None);


# In[54]:


fig, ax = plt.subplots(1,2, figsize=(15,5))
fig.suptitle("EUI by State and building class")

sns.violinplot(data=df_all, y='site_eui', x='state_factor', ax=ax[0])
sns.violinplot(data=df_all, y='site_eui', x='building_class', ax=ax[1]);


# Labs and Data Centers have higher EUI compared to the other types of buildings. Grocery stores, Health Care Inpatient, Health Care Uncategorized, Health Care Outpatient, and Food service, restaurants have higher range of EUI. It could be the essential services must operate for longer hours, therefore, have higher EUI.

# In[55]:


fig, ax = plt.subplots(figsize=(20,5))
fig.suptitle("EUI by facility type")
ax=sns.boxplot(data=df_all, y='site_eui', x='facility_type')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90);


# `floor_area` could have positive relationship with `EUI`. The younger buildings tend to have higher `EUI` which could be because building height and size have increased over the years. It looks like the Higher the `energy_star_rating` is, the lower the `EUI` becomes.

# In[56]:


fig, ax = plt.subplots(1,4, figsize=(20,5))

for e, col in enumerate(['floor_area', 'year_built', 'energy_star_rating', 'elevation']):
    if col =='year_built':
        sns.scatterplot(data=df_all[(df_all['year_built']!=0) & (df_all['year_built'].notna())], 
                        x=col, y='site_eui', ax=ax[e]).set(title='EUI by '+ col, ylabel=None)
    else:
        sns.scatterplot(data=df_all, x=col, y='site_eui', ax=ax[e]).set(title='EUI by '+ col, ylabel=None);


# The most data points are in lower number of `cooling_degree_days` and higher number of`heating_degree_days`. The majority of the datapoints are also in the lower levels of `snowfall_inches` and `snowdepth_inches`. `direction_max_wind_speed`, `direction_peak_wind_speed`, `max_wind_speed`, and `days_with_fog` columns have the `NA` values of over 50%. No relationship between `EUI` and the weather related numerica columns can be observed from the plot.

# In[57]:


cols=[['cooling_degree_days','heating_degree_days', 'precipitation_inches', 
      'snowfall_inches','snowdepth_inches'], 
      ['avg_temp', 
      'direction_max_wind_speed','direction_peak_wind_speed', 'max_wind_speed', 
      'days_with_fog']]

fig, ax = plt.subplots(2,5, figsize=(20,8))
fig.suptitle('EUI by numerical columns')

for e1, l in enumerate(cols):
    for e2, col in enumerate(l):
        sns.scatterplot(data=df_all, 
                        x=col, y='site_eui', ax=ax[e1, e2]).set(ylabel=None);


# ### Observations
# 
# - Categorical variables such as `State_Factor`, `building_class` and `facility_type` might have some correlation with `EUI`.
# - `State_6` is not present in `test` set. State 6 removed training data should be tested.
# - `floor_area`, `energe_star_rating` should be included in the modelling to be tested.
# - From the plots, it's difficult to observe direct (linear) relationship between `EUI` and weather related variables. However, this doesn't deny non-linear relationships among the variables.
# - Variables with more than 50% `NA` values should not be imputed (in my opinion) and better to be not included in the training set.
# - Weather variables have few unique values repeated throughout the datapoints. Not sure how this duplicated values might affect the modeling and prediction.

# 
# # Preprocessing

# we will suppose that two  if two buildings have the same values for these features;   
# they are the same building, in other words groupby_cols = (building_id)
# 
# 
# Removing duplicates by clubbing similar building data

# In[58]:


groupby_cols = ['state_factor','building_class','facility_type','floor_area','year_built']
df_all = df_all.sort_values(by=groupby_cols+['year_factor']).reset_index(drop=True)


# In[59]:


df_all.loc[:,df_all.dtypes=='object'].columns


# Null imputation for categorical values: **KNN Imputing**
# 
# 

# In[60]:


cats = ['state_factor', 'facility_type', 'building_class']
for col in cats:
    dummies = pd.get_dummies(df_all[col], dummy_na=False)
    for ohe_col in dummies:
        df_all[f'ohe_{col}_{ohe_col}'] = dummies[ohe_col]


# In[61]:


df_all


# In[62]:


from sklearn.impute import KNNImputer
import pandas as pd
import joblib


# In[63]:


import os
import pandas as pd
from sklearn.impute import KNNImputer
import joblib

knn_imputing = False
target = 'site_eui'
data_dir = 'data'
model_dir = 'models'

if knn_imputing:
    imputer = KNNImputer(n_neighbors=7)
    tmp = df_all[['State_Factor', 'building_class', 'facility_type', 'dataset', target]]
    df = df_all.drop(tmp.columns, axis=1)
    df1 = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    tmp.to_csv(os.path.join(data_dir, 'imputer_tmp.csv'), index=False)
    df1.to_csv(os.path.join(data_dir, 'imputer_df1.csv'), index=False)
    joblib.dump(imputer, os.path.join(model_dir, 'knn_imputer.pkl'))
    
else:
    df1_path = os.path.join(data_dir, 'imputer_df1.csv')
    
    if os.path.exists(df1_path):
        df1 = pd.read_csv(df1_path)
        tmp = df_all[['State_Factor', 'building_class', 'facility_type', 'dataset', target]]
        df_all = df_all.drop(tmp.columns, axis=1)
        
        for col in tmp.columns:
            df_all[col] = tmp[col]
        for col in df1.columns:
            df_all[col] = df1[col]
    else:
        print(f"File '{df1_path}' does not exist.")
        
        


# In[64]:


get_ipython().system('pip install category_encoders')
    
import category_encoders as ce

cats = ['state_factor', 'building_class', 'facility_type']
for col in cats:
    encoder = ce.TargetEncoder()
    df_all[f'te_{col}'] = encoder.fit_transform(df_all[col], df_all[target])


# # Feature Engineering

# In[65]:


# extract new weather statistics from the building location weather features
temp = [col for col in df_all.columns if 'temp' in col]

df_all['min_temp'] = df_all[temp].min(axis=1)
df_all['max_temp'] = df_all[temp].max(axis=1)
df_all['avg_temp'] = df_all[temp].mean(axis=1)
df_all['std_temp'] = df_all[temp].std(axis=1)
df_all['skew_temp'] = df_all[temp].skew(axis=1)

# by seasons
temp = pd.Series([col for col in df_all.columns if 'temp' in col])

winter_temp = temp[temp.apply(lambda x: ('january' in x or 'february' in x or 'december' in x))].values
spring_temp = temp[temp.apply(lambda x: ('march' in x or 'april' in x or 'may' in x))].values
summer_temp = temp[temp.apply(lambda x: ('june' in x or 'july' in x or 'august' in x))].values
autumn_temp = temp[temp.apply(lambda x: ('september' in x or 'october' in x or 'november' in x))].values


### winter
df_all['min_winter_temp'] = df_all[winter_temp].min(axis=1)
df_all['max_winter_temp'] = df_all[winter_temp].max(axis=1)
df_all['avg_winter_temp'] = df_all[winter_temp].mean(axis=1)
df_all['std_winter_temp'] = df_all[winter_temp].std(axis=1)
df_all['skew_winter_temp'] = df_all[winter_temp].skew(axis=1)
### spring
df_all['min_spring_temp'] = df_all[spring_temp].min(axis=1)
df_all['max_spring_temp'] = df_all[spring_temp].max(axis=1)
df_all['avg_spring_temp'] = df_all[spring_temp].mean(axis=1)
df_all['std_spring_temp'] = df_all[spring_temp].std(axis=1)
df_all['skew_spring_temp'] = df_all[spring_temp].skew(axis=1)
### summer
df_all['min_summer_temp'] = df_all[summer_temp].min(axis=1)
df_all['max_summer_temp'] = df_all[summer_temp].max(axis=1)
df_all['avg_summer_temp'] = df_all[summer_temp].mean(axis=1)
df_all['std_summer_temp'] = df_all[summer_temp].max(axis=1)
df_all['skew_summer_temp'] = df_all[summer_temp].max(axis=1)
## autumn
df_all['min_autumn_temp'] = df_all[autumn_temp].min(axis=1)
df_all['max_autumn_temp'] = df_all[autumn_temp].max(axis=1)
df_all['avg_autumn_temp'] = df_all[autumn_temp].mean(axis=1)
df_all['std_autumn_temp'] = df_all[autumn_temp].std(axis=1)
df_all['skew_autumn_temp'] = df_all[autumn_temp].skew(axis=1)


# In[66]:


df_all['month_cooling_degree_days'] = df_all['cooling_degree_days']/12
df_all['month_heating_degree_days'] = df_all['heating_degree_days']/12


# ### Buildig based feature:
# 
# we will extract building statistics

# In[67]:


# total area
df_all['building_area'] = df_all['floor_area'] * df_all['elevation']
# rating energy by floor
df_all['floor_energy_star_rating'] = df_all['energy_star_rating']/df_all['elevation']


# In[68]:


### Checking target variable transformation


# In[69]:


target = 'site_eui'
plt.figure(figsize=(10,7))
# plot the original variable vs sale price    
plt.subplot(2, 1, 1)
train[target].hist(bins=50)
plt.ylabel('Count')
plt.xlabel('Original ' + target)

# plot transformed variable vs sale price
plt.subplot(2, 1, 2)
np.log(train[target]).hist(bins=50)
plt.ylabel('Count')
plt.xlabel('Transformed ' + target);


# In[70]:


nums = df_all.select_dtypes(include='number').columns

for col in nums:
    plt.figure(figsize=(8, 6))
    plt.hist(df_all[col], bins=50)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# In[71]:


df_all[nums].skew().sort_values(key=abs, ascending=False)[:5]


# In[72]:


skewed = ['days_above_110F', 'days_above_100F']

for var in skewed:
    
    # map the variable values into 0 and 1
    df_all[var] = np.where(df_all[var]==0, 0, 1)


# In[ ]:





# ### Saving feature dataset
# 

# In[73]:


import pickle

saved = False
data_directory = 'data'
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

if saved:
    get_ipython().system('pip install pickle5 --quiet')
    import pickle5 as pickle
    data_path = os.path.join(data_directory, 'feature_engineering.pkl')
    with open(data_path, "rb") as fh:
        df = pickle.load(fh)
else:
    df_all.to_pickle(os.path.join(data_directory, 'feature_transformed_set.pkl'))


# In[74]:


df_all.shape


# In[75]:


df_all.head()


# # Baseline Modeling

# In[76]:


cats = ['state_factor', 'facility_type', 'building_class', 'days_above_100F', 'days_above_110F']

# Typecasting numerical features
for col in df_all.columns:
    if col not in cats + ['dataset', 'id', 'site_eui']:
        df_all[col] = df_all[col].astype('float64')   
    


# In[77]:


df_all.drop(columns=cats)


# In[78]:


train = df_all[df_all['dataset'] == 'train']
test = df_all[df_all['dataset'] == 'test']

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

test_ids = test['id'] if 'id' in test.columns else None
train_ids = train['id'] if 'id' in train.columns else None

target = train['site_eui']

train = train.drop(['dataset', 'site_eui'], axis=1)
test = test.drop(['dataset', 'site_eui'], axis=1)


# In[ ]:





# ### Label Encoding discrete features
# 

# In[79]:


# get discrete end categorical features colums indexes 
# needed later for the cat bosst model
cats_discrete_idx = np.where(train.dtypes != 'float64')[0]
# create the label
le = LabelEncoder()
for col_idx in cats_discrete_idx:
    train.iloc[:, col_idx] = le.fit_transform(train.iloc[:, col_idx].astype(str))
    test.iloc[:, col_idx] = le.transform(test.iloc[:, col_idx].astype(str))


# In[80]:


print("Label Encoded Columns:")
for i in cats_discrete_idx:
    print(train.columns[i])


# In[83]:


y_test = pd.read_csv(r'C:\Users\manka\OneDrive\Desktop\Y_test.csv')


X_train = train
X_test = test
y_train = target
y_test = y_test['site_eui']
print('Train: ', X_train.shape)
print('Test:', X_test.shape)
print('Samples: ', y_train.shape)
print('Targets: ', y_test.shape)


# # Catboost

# In[84]:


get_ipython().system('pip install catboost')

from catboost import CatBoostRegressor


# In[85]:


catb = CatBoostRegressor(iterations=500,
                         learning_rate=0.02,
                         depth=12,
                         eval_metric='RMSE',
#                         early_stopping_rounds=42,
                         random_seed = 23,
                         bagging_temperature = 0.2,
                         od_type='Iter',
                         metric_period = 75,
                         od_wait=100)


# In[86]:


catb.fit(X_train, y_train,
                 eval_set=(X_test,y_test),
                 cat_features=cats_discrete_idx,
                 use_best_model=True,
                 verbose=True)

y_pred = catb.predict(X_test)


# In[87]:


print(" Training data scores\n","--"*10)
print(" RMSE:", np.sqrt(mean_squared_error(y_test,y_pred)))
print(" MAE:", mean_absolute_error(y_test,y_pred))
print(" MSE:", mean_squared_error(y_test,y_pred))
print(" R2:", r2_score(y_test,y_pred))


# In[ ]:





# # XGBoost

# In[88]:


xgb = XGBRegressor(n_estimators=500, reg_alpha=0.01, n_jobs=-1)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)


# In[89]:


print(" Training data scores\n","--"*10)
print(" RMSE:", np.sqrt(mean_squared_error(y_test,y_pred)))
print(" MAE:", mean_absolute_error(y_test,y_pred))
print(" MSE:", mean_squared_error(y_test,y_pred))
print(" R2:", r2_score(y_test,y_pred))


# # Random Forest

# In[90]:


from sklearn.impute import SimpleImputer

# Create an instance of the SimpleImputer
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on X_train
X_train_imputed = imputer.fit_transform(X_train)

# Fit the RandomForestRegressor on the imputed data
rf = RandomForestRegressor(random_state=1, max_depth=15, min_samples_split=2)
rf.fit(X_train_imputed, y_train)


# ### 
# rf = RandomForestRegressor(random_state=1, max_depth=15, min_samples_split=2)
# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)

# In[91]:


print(" Training data scores\n","--"*10)
print(" RMSE:", np.sqrt(mean_squared_error(y_test,y_pred)))
print(" MAE:", mean_absolute_error(y_test,y_pred))
print(" MSE:", mean_squared_error(y_test,y_pred))
print(" R2:", r2_score(y_test,y_pred))


# # Model Evaluation

# In[92]:


error_rec = {
    "catboost": {
        "mae": 40.29268484855883,
        "rmse": 61.19378120765249,
    },
    "randomforest": {
        "mae": 52.41012420250038,
        "rmse": 81.8514058171361,
    },
    "xgboost": {
        "mae": 52.41012420250038,
        "rmse": 81.8514058171361,
    },
}
pd.DataFrame(error_rec).plot(kind="bar", 
             color=[
                 sns.color_palette("pastel")[0], 
                 sns.color_palette("pastel")[1], 
                 sns.color_palette("pastel")[2], 
                 sns.color_palette("pastel")[3]]);


# # Hyperparameter Tuning

# ### Using CrossValidation on CatBoost

# In[95]:


for fold, (train_idx, test_idx) in tqdm.tqdm(enumerate(kf.split(train, target))):
    X_train, X_test = train.iloc[train_idx][test.columns], train.iloc[test_idx][test.columns]
    y_train, y_test = target[train_idx], target[test_idx]
    
    catb = CatBoostRegressor(iterations=500,
                         learning_rate=0.02,
                         depth=12,
                         eval_metric='RMSE',
                         random_seed=23,
                         bagging_temperature=0.2,
                         od_type='Iter',
                         metric_period=75,
                         od_wait=100)
    
    # train model
    catb.fit(X_train, y_train,
             eval_set=(X_test, y_test),
             cat_features=cats_discrete_idx,
             use_best_model=True,
             verbose=True)
    
    oof = catb.predict(X_test)  # Use catb to predict, not model
    train_oof[test_idx] = oof
    test_preds += catb.predict(test) / NUM_FOLDS  # Use catb to predict, not model
    print(f"out-of-folds prediction ==== fold_{fold} RMSE", np.sqrt(mean_squared_error(oof, y_test, squared=False)))


# In[96]:


train_oof = np.zeros((train.shape[0],))
test_preds = np.zeros(test.shape[0])

NUM_FOLDS = 5
kf = KFold(n_splits=5, shuffle=True, random_state=0)

for fold, (train_idx, test_idx) in tqdm.tqdm(enumerate(kf.split(train, target))):
    X_train, X_test = train.iloc[train_idx][test.columns], train.iloc[test_idx][test.columns]
    y_train, y_test = target[train_idx], target[test_idx]
    
    catb = CatBoostRegressor(iterations=500,
                         learning_rate=0.02,
                         depth=12,
                         eval_metric='RMSE',
#                         early_stopping_rounds=42,
                         random_seed=23,
                         bagging_temperature=0.2,
                         od_type='Iter',
                         metric_period=75,
                         od_wait=100)
    # train model
    catb.fit(X_train, y_train,
             eval_set=(X_test, y_test),
             cat_features=cats_discrete_idx,
             use_best_model=True,
             verbose=True)

    oof = catb.predict(X_test)
    train_oof[test_idx] = oof
    test_preds += catb.predict(test) / NUM_FOLDS      
    print(f"out-of-folds prediction ==== fold_{fold} RMSE", np.sqrt(mean_squared_error(oof, y_test, squared=False)))


# In[97]:


# prepaere the out of folds predictions
train_oof = np.zeros((train.shape[0],))
test_preds = np.zeros(test.shape[0])

NUM_FOLDS = 5
kf = KFold(n_splits=5, shuffle=True, random_state=0)

for fold, (train_idx, test_idx) in tqdm.tqdm(enumerate(kf.split(train, target))):
    X_train, X_test = train.iloc[train_idx][test.columns], train.iloc[test_idx][test.columns]
    y_train, y_test = target[train_idx], target[test_idx]

    catb = CatBoostRegressor(iterations=500, learning_rate=0.02, depth=12, eval_metric='RMSE',
                             random_seed=23, bagging_temperature=0.2, od_type='Iter', metric_period=75, od_wait=100)
    
    # train model
    catb.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=cats_discrete_idx,
             use_best_model=True, verbose=True)

    oof = catb.predict(X_test)  # Use catb instead of model
    train_oof[test_idx] = oof
    test_preds += catb.predict(test) / NUM_FOLDS
    print(f"out-of-folds prediction ==== fold_{fold} RMSE", np.sqrt(mean_squared_error(oof, y_test, squared=False)))


# In[104]:


### Using Optuna with Random Forest


# In[105]:


# cross validating training data
kfolds = KFold(n_splits=3, shuffle=True, random_state=42)

# Objective function
def random_forest_objective(trial, data=X_train, target=y_train):
    # Dictionary to store best parameters
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "max_features": trial.suggest_float("max_features", 0.01, 0.95)
    }
     
    model = RandomForestRegressor(**param)
    
    # Setting random seed and kfolds for cross-validation
    kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
    
    scores = cross_val_score(model, data, target, cv=kfolds, scoring="neg_root_mean_squared_error")
    return scores.mean()


# In[106]:


def tuner(objective, n=5, direction='minimize'): 
    # Create Study object
    study = optuna.create_study(direction="minimize")

    # Optimize the study
    study.optimize(objective, n_trials=n)

    # Print the result
    best_params = study.best_params
    best_score = study.best_value
    print(f"Best score: {best_score}")
    print(f"Optimized parameters: {best_params}\n")
    print("<<<<<<<<<< Tuning complete... >>>>>>>>>>")
    
    # Return best parameters for the model
    return best_params, best_score


# In[107]:


get_ipython().system('pip install optuna')


# In[110]:


import optuna


# ### %%time
# rf_param, rf_score = tuner(random_forest_objective,1)
# rf_tuned_model = RandomForestRegressor(**rf_param)

# In[113]:


from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Assuming X_train and y_train are your training data and target
# Create a SimpleImputer instance
imputer = SimpleImputer(strategy='mean')

# Impute missing values in X_train
X_train_imputed = imputer.fit_transform(X_train)

# Create a RandomForestRegressor instance
model = RandomForestRegressor(n_estimators=100, max_depth=10)  # Adjust hyperparameters as needed

# Cross-validation with imputed data
scores = cross_val_score(model, X_train_imputed, y_train, cv=5, scoring='neg_mean_squared_error')


# In[115]:


from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Assuming X_train and y_train are your training data and target
# Create a SimpleImputer instance
imputer = SimpleImputer(strategy='mean')

# Impute missing values in X_train
X_train_imputed = imputer.fit_transform(X_train)

# Create a RandomForestRegressor instance
rf_tuned_model = RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_split=7, max_features=0.13882787435900823)

# Fit the tuned model
rf_tuned_model.fit(X_train_imputed, y_train)


# In[ ]:





# # Final Evaluation

# In[123]:


y_hat_tuned = catb.predict(X_test)

plt.figure(figsize = (7,5))
sns.distplot(y_test - y_hat_tuned)
plt.title("Error Rate Distribution");
plt.ylabel("error")
plt.xlabel("iteration")


# In[ ]:




