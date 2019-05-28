# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"toc": true, "cell_type": "markdown"}
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# +
import pandas as pd
import numpy as np



import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
sns.set_style('darkgrid')

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix







import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# -

df = pd.read_csv('agents_with_transactions.csv')

le = LabelEncoder()
df = df.assign(TRANSACTION_ID=le.fit(df['TRANSACTION_ID']).transform(df['TRANSACTION_ID']))


# +
# Transactions contracted at
df.loc[df.TRANSACTION_ID == 7686, 'transaction_contracted_at'] = "2018-07-31 12:00:00 UTC"
df.loc[df.TRANSACTION_ID == 3874, 'transaction_contracted_at'] = "2019-04-17 04:56:02 UTC"
df.loc[df.TRANSACTION_ID == 4084, 'transaction_contracted_at'] = "2019-04-22 04:56:02 UTC"
df.loc[df.TRANSACTION_ID == 11335, 'transaction_contracted_at'] = "2019-01-25 05:50:36 UTC"
df.loc[df.TRANSACTION_ID == 11400, 'transaction_contracted_at'] = "2019-01-29 12:00:00 UTC"
df.loc[df.TRANSACTION_ID == 12536, 'transaction_contracted_at'] = "2019-02-28 05:50:36 UTC"
df.loc[df.TRANSACTION_ID == 5539, 'transaction_contracted_at'] = "2019-04-30 12:00:00 UTC"
df.loc[df.TRANSACTION_ID == 1234, 'transaction_contracted_at'] = "2018-03-31 05:00:00 UTC"
df.loc[df.TRANSACTION_ID == 6642, 'transaction_contracted_at'] = "2018-05-31 12:00:00 UTC"
df.loc[df.TRANSACTION_ID == 7864, 'transaction_contracted_at'] = "2018-07-23 12:00:00 UTC"
df.loc[df.TRANSACTION_ID == 9186, 'transaction_contracted_at'] = "2018-11-15 05:50:36 UTC"
df.loc[df.TRANSACTION_ID == 2106, 'transaction_contracted_at'] = "2019-03-14 12:00:00 UTC"
df.loc[df.TRANSACTION_ID == 4976, 'transaction_contracted_at'] = "2019-05-12 12:00:00 UTC"
df.loc[df.TRANSACTION_ID == 10732, 'transaction_contracted_at'] = "2018-05-20 12:00:00 UTC"
df.loc[df.TRANSACTION_ID == 2153, 'transaction_contracted_at'] = "2018-12-20 12:00:00 UTC"
df.loc[df.TRANSACTION_ID == 3582, 'transaction_contracted_at'] = "2019-04-19 12:00:00 UTC"
df.loc[df.TRANSACTION_ID == 1153, 'transaction_contracted_at'] = "2018-07-28 05:50:36 UTC"
df.loc[df.TRANSACTION_ID == 2474, 'transaction_contracted_at'] = "2019-03-10 12:00:00 UTC"

# Transactions effective at
df.loc[df.TRANSACTION_ID == 8017, 'transaction_effective_at'] = "2018-08-31 12:00:00 UTC"
df.loc[df.TRANSACTION_ID == 5675, 'transaction_effective_at'] = "2019-07-22 12:00:00 UTC"
df.loc[df.TRANSACTION_ID == 3423, 'transaction_effective_at'] = "2019-11-12 12:00:00 UTC"
df.loc[df.TRANSACTION_ID == 5141, 'transaction_effective_at'] = "2019-06-14 12:00:00 UTC"

# -

df['TRANSACTION_CONTRACTED_AT'] = pd.to_datetime(df.TRANSACTION_CONTRACTED_AT, errors = 'coerce')
df['TRANSACTION_EFFECTIVE_AT'] = pd.to_datetime(df.TRANSACTION_EFFECTIVE_AT, errors = 'coerce')

df.isnull().sum()

# +
'10732' '0002-05-20 12:00:00 UTC'
'2153' '0003-12-20 12:00:00 UTC'      
'3582' '0004-04-19 12:00:00 UTC'
'1153' '0201-07-28 05:50:36 UTC'
'2474' '0009-03-10 12:00:00 UTC'

'7686' '2018-07-31 12:00:00 UTC'
'3874' '2019-04-17 04:56:02 UTC'   
'4084' '2019-04-22 04:56:02 UTC'       
'11335' '2019-01-25 05:50:36 UTC'
'11400' '2019-01-29 12:00:00 UTC'
'12536' '2019-02-28 05:50:36 UTC'      
'5539' '2019-04-30 12:00:00 UTC'
'1234' '2018-03-31 05:00:00 UTC'
'6642' '2018-05-31 12:00:00 UTC'
'7864' '2018-07-23 12:00:00 UTC'
'9186' '2018-11-15 05:50:36 UTC'
'2106' '2019-03-14 12:00:00 UTC'
'4976' '2019-05-12 12:00:00 UTC'



8017	2018-08-31 12:00:00 UTC
5675	2019-07-22 12:00:00 UTC
3423	2019-11-12 12:00:00 UTC
5141	2019-06-14 12:00:00 UTC
# -











full_df.head()

full_df.shape

full_df.isnull().sum()

df = full_df[['AGENT_ID',
              'BROKERAGE_ID',
              'COMMISSION_SCHEDULE_STRATEGY',
              'TRANSACTION_ID',
              'TRANSACTION_CONTRACTED_AT',
              'TRANSACTION_EFFECTIVE_AT',
              'TRANSACTION_SALES_AMOUNT',
              'TRANSACTION_SIDE',
              'EARNED_SIDE_COUNT',
              'EARNED_VOLUME',
              'STANDARD_COMMISSION_GCI_AMOUNT',
              'STANDARD_COMMISSION_AGENT_NET_AMOUNT',
              'STANDARD_COMMISSION_BROKERAGE_NET_AMOUNT']]


df

# +
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df = df.assign(AGENT_ID=le.fit(df['AGENT_ID']).transform(df['AGENT_ID']))
df = df.assign(BROKERAGE_ID=le.fit(df['BROKERAGE_ID']).transform(df['BROKERAGE_ID']))
df = df.assign(TRANSACTION_ID=le.fit(df['TRANSACTION_ID']).transform(df['TRANSACTION_ID']))

# -

df

df = df[df['TRANSACTION_SALES_AMOUNT']<35000000]

df.to_csv('cut_df.csv')

df.sort_values('TRANSACTION_CONTRACTED_AT')

















total_sales = df.groupby('AGENT_ID').sum().sort_values('TRANSACTION_SALES_AMOUNT',ascending=False)

five_mil = total_sales[total_sales['TRANSACTION_SALES_AMOUNT']>5000000]

five_mil_agent_id = five_mil.index

five_mil_agent_id

df['is_5mil'] = df['AGENT_ID'].isin(five_mil_agent_id).astype(int)

df



df['TRANSACTION_CONTRACTED_AT'] = pd.to_datetime(df.TRANSACTION_CONTRACTED_AT, errors = 'coerce')
df['TRANSACTION_EFFECTIVE_AT'] = pd.to_datetime(df.TRANSACTION_EFFECTIVE_AT, errors = 'coerce')

df.sort_values('TRANSACTION_CONTRACTED_AT')














df


