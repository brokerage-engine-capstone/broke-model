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

# +
import pandas as pd


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# -

full_df = pd.read_csv('agents_with_transactions.csv')

full_df.head()

full_df.isnull().sum()

df = full_df[['AGENT_ID',
              'BROKERAGE_ID',
              'TRANSACTION_ID',
              'TRANSACTION_CONTRACTED_AT',
              'TRANSACTION_EFFECTIVE_AT',
              'TRANSACTION_SALES_AMOUNT',
              'EARNED_SIDE_COUNT',
              'EARNED_VOLUME',
              'STANDARD_COMMISSION_GCI_AMOUNT',
              'STANDARD_COMMISSION_AGENT_NET_AMOUNT',
              'STANDARD_COMMISSION_BROKERAGE_NET_AMOUNT']]


# +
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['AGENT_ID'] = le.fit(df['AGENT_ID']).transform(df['AGENT_ID'])
df['BROKERAGE_ID'] = le.fit(df['BROKERAGE_ID']).transform(df['BROKERAGE_ID'])

# -

df

df = df[df['TRANSACTION_SALES_AMOUNT']<35000000]



total_sales = df.groupby('AGENT_ID').sum().sort_values('TRANSACTION_SALES_AMOUNT',ascending=False)

five_mil = total_sales[total_sales['TRANSACTION_SALES_AMOUNT']>5000000]

five_mil_agent_id = five_mil.index

five_mil_agent_id

df['is_5mil'] = df['AGENT_ID'].isin(five_mil_agent_id).astype(int)

df[df['is_5mil']==1]



df['TRANSACTION_CONTRACTED_AT'] = pd.to_datetime(df.TRANSACTION_CONTRACTED_AT, errors = 'coerce')
df['TRANSACTION_EFFECTIVE_AT'] = pd.to_datetime(df.TRANSACTION_EFFECTIVE_AT, errors = 'coerce')
















df


