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
# <div class="toc"><ul class="toc-item"><li><span><a href="#Environment" data-toc-modified-id="Environment-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Environment</a></span></li><li><span><a href="#Acquisition" data-toc-modified-id="Acquisition-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Acquisition</a></span></li><li><span><a href="#Preparation" data-toc-modified-id="Preparation-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Preparation</a></span><ul class="toc-item"><li><span><a href="#Encoding-various-ID-fields-for-readability-and-future-usage." data-toc-modified-id="Encoding-various-ID-fields-for-readability-and-future-usage.-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Encoding various ID fields for readability and future usage.</a></span></li><li><span><a href="#Correcting-dates-that-won't-get-transitioned-over" data-toc-modified-id="Correcting-dates-that-won't-get-transitioned-over-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Correcting dates that won't get transitioned over</a></span></li><li><span><a href="#Converting-non-times-to-actual-times" data-toc-modified-id="Converting-non-times-to-actual-times-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Converting non-times to actual times</a></span></li><li><span><a href="#Column-cleanup" data-toc-modified-id="Column-cleanup-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Column cleanup</a></span><ul class="toc-item"><li><span><a href="#Bonus-Columns-Removing-these-due-to-98%+-nulls" data-toc-modified-id="Bonus-Columns-Removing-these-due-to-98%+-nulls-3.4.1"><span class="toc-item-num">3.4.1&nbsp;&nbsp;</span>Bonus Columns Removing these due to 98%+ nulls</a></span></li><li><span><a href="#Dropping-standard_commission_type-as-it-only-has-one-variable-and-nulls" data-toc-modified-id="Dropping-standard_commission_type-as-it-only-has-one-variable-and-nulls-3.4.2"><span class="toc-item-num">3.4.2&nbsp;&nbsp;</span>Dropping standard_commission_type as it only has one variable and nulls</a></span></li><li><span><a href="#Imputing-nulls-with-0" data-toc-modified-id="Imputing-nulls-with-0-3.4.3"><span class="toc-item-num">3.4.3&nbsp;&nbsp;</span>Imputing nulls with 0</a></span></li></ul></li></ul></li><li><span><a href="#Exploration" data-toc-modified-id="Exploration-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Exploration</a></span></li><li><span><a href="#Modeling" data-toc-modified-id="Modeling-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Modeling</a></span></li></ul></div>
# -

# ## Environment 

# +
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# -

# ## Acquisition

df = pd.read_csv('../agents_with_transactions.csv')

# ## Preparation

# **Lowercasing all column names**

[df.rename(columns=lambda x: x.lower(), inplace=True) for col in df]

df.head()

# ### Encoding various ID fields for readability and future usage.

# +
to_encode = ['agent_id', 'brokerage_id', 'transaction_id']

for col in to_encode:
    le = LabelEncoder().fit(df[col])
    df[col] = le.transform(df[col])
# -

# ### Correcting dates that won't get transitioned over

# +
# Transactions contracted at errors fixed
df.loc[df.transaction_id == 7686, 'transaction_contracted_at'] = "2018-07-31 12:00:00 UTC"
df.loc[df.transaction_id == 3874, 'transaction_contracted_at'] = "2019-04-17 04:56:02 UTC"
df.loc[df.transaction_id == 4084, 'transaction_contracted_at'] = "2019-04-22 04:56:02 UTC"
df.loc[df.transaction_id == 11335, 'transaction_contracted_at'] = "2019-01-25 05:50:36 UTC"
df.loc[df.transaction_id == 11400, 'transaction_contracted_at'] = "2019-01-29 12:00:00 UTC"
df.loc[df.transaction_id == 12536, 'transaction_contracted_at'] = "2019-02-28 05:50:36 UTC"
df.loc[df.transaction_id == 5539, 'transaction_contracted_at'] = "2019-04-30 12:00:00 UTC"
df.loc[df.transaction_id == 1234, 'transaction_contracted_at'] = "2018-03-31 05:00:00 UTC"
df.loc[df.transaction_id == 6642, 'transaction_contracted_at'] = "2018-05-31 12:00:00 UTC"
df.loc[df.transaction_id == 7864, 'transaction_contracted_at'] = "2018-07-23 12:00:00 UTC"
df.loc[df.transaction_id == 9186, 'transaction_contracted_at'] = "2018-11-15 05:50:36 UTC"
df.loc[df.transaction_id == 2106, 'transaction_contracted_at'] = "2019-03-14 12:00:00 UTC"
df.loc[df.transaction_id == 4976, 'transaction_contracted_at'] = "2019-05-12 12:00:00 UTC"
df.loc[df.transaction_id == 10732, 'transaction_contracted_at'] = "2018-05-20 12:00:00 UTC"
df.loc[df.transaction_id == 2153, 'transaction_contracted_at'] = "2018-12-20 12:00:00 UTC"
df.loc[df.transaction_id == 3582, 'transaction_contracted_at'] = "2019-04-19 12:00:00 UTC"
df.loc[df.transaction_id == 1153, 'transaction_contracted_at'] = "2018-07-28 05:50:36 UTC"
df.loc[df.transaction_id == 2474, 'transaction_contracted_at'] = "2019-03-10 12:00:00 UTC"

# Transactions effective at errors fixed
df.loc[df.transaction_id == 8017, 'transaction_effective_at'] = "2018-08-31 12:00:00 UTC"
df.loc[df.transaction_id == 5675, 'transaction_effective_at'] = "2019-07-22 12:00:00 UTC"
df.loc[df.transaction_id == 3423, 'transaction_effective_at'] = "2019-11-12 12:00:00 UTC"
df.loc[df.transaction_id == 5141, 'transaction_effective_at'] = "2019-06-14 12:00:00 UTC"
# -

# ### Converting non-times to actual times

df['transaction_contracted_at'] = pd.to_datetime(df.transaction_contracted_at, errors = 'coerce')
df['transaction_effective_at'] = pd.to_datetime(df.transaction_effective_at, errors = 'coerce')

# ### Column cleanup

# #### Bonus Columns Removing these due to 98%+ nulls
#

to_drop = [col for col in df if col.startswith('bonus_')]
df.drop(columns=to_drop, inplace=True)

# #### Dropping standard_commission_type as it only has one variable and nulls

df.drop(columns='standard_commission_type', inplace=True)

# #### Imputing nulls with 0

df.standard_commission_gci_amount.fillna(value=0, inplace=True)
df.standard_commission_brokerage_net_amount.fillna(value=0, inplace=True)

# ## Exploration

# ## Modeling


