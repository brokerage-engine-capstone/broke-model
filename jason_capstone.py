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
# <div class="toc"><ul class="toc-item"><li><span><a href="#Notes" data-toc-modified-id="Notes-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Notes</a></span><ul class="toc-item"><li><span><a href="#Questions" data-toc-modified-id="Questions-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Questions</a></span></li><li><span><a href="#Hypotheses" data-toc-modified-id="Hypotheses-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Hypotheses</a></span></li></ul></li><li><span><a href="#Environment" data-toc-modified-id="Environment-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Environment</a></span></li><li><span><a href="#Acquisition" data-toc-modified-id="Acquisition-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Acquisition</a></span></li><li><span><a href="#Preparation" data-toc-modified-id="Preparation-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Preparation</a></span><ul class="toc-item"><li><span><a href="#Lowercasing-all-column-names" data-toc-modified-id="Lowercasing-all-column-names-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Lowercasing all column names</a></span></li><li><span><a href="#Encoding-various-ID-fields-for-readability-and-future-usage." data-toc-modified-id="Encoding-various-ID-fields-for-readability-and-future-usage.-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Encoding various ID fields for readability and future usage.</a></span></li><li><span><a href="#Correcting-dates" data-toc-modified-id="Correcting-dates-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Correcting dates</a></span></li><li><span><a href="#Converting-date-columns-to-datetime-type" data-toc-modified-id="Converting-date-columns-to-datetime-type-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Converting date columns to datetime type</a></span></li><li><span><a href="#Column-cleanup" data-toc-modified-id="Column-cleanup-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Column cleanup</a></span><ul class="toc-item"><li><span><a href="#Remove-bonus-columns" data-toc-modified-id="Remove-bonus-columns-4.5.1"><span class="toc-item-num">4.5.1&nbsp;&nbsp;</span>Remove bonus columns</a></span></li><li><span><a href="#Drop-standard_commission_type" data-toc-modified-id="Drop-standard_commission_type-4.5.2"><span class="toc-item-num">4.5.2&nbsp;&nbsp;</span>Drop standard_commission_type</a></span></li><li><span><a href="#Imputing-nulls-with-0" data-toc-modified-id="Imputing-nulls-with-0-4.5.3"><span class="toc-item-num">4.5.3&nbsp;&nbsp;</span>Imputing nulls with 0</a></span></li></ul></li><li><span><a href="#Drop-agent_name" data-toc-modified-id="Drop-agent_name-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>Drop agent_name</a></span></li><li><span><a href="#Drop-brokerage-name" data-toc-modified-id="Drop-brokerage-name-4.7"><span class="toc-item-num">4.7&nbsp;&nbsp;</span>Drop brokerage name</a></span></li><li><span><a href="#Drop-commission_schedule_active" data-toc-modified-id="Drop-commission_schedule_active-4.8"><span class="toc-item-num">4.8&nbsp;&nbsp;</span>Drop commission_schedule_active</a></span></li></ul></li><li><span><a href="#Exploration" data-toc-modified-id="Exploration-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Exploration</a></span></li><li><span><a href="#Modeling" data-toc-modified-id="Modeling-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Modeling</a></span></li></ul></div>
# -

# ## Notes

# ### Questions
# * Brokerage Agency may have an impact on agent performance but we want to remain brokerage-agnostic.
# * Commission Schedule ID - is this relevant?
# * Commission split and volume may be correlated - tenure may also have an impact.
# * What is Commission Schedule Active?  Should we drop?
# * What is Transaction Number?
# * If multiple rows have the same transaction ids and multiple agents, are these split?
# * What happens when an agent leaves the agency?
# * Listing price would be very valuable information.
# * Can we get a better understanding of agent tenure?
# * Is 5mil reasonable for a top performer?
# * What does BE want as a deliverable?
# * Can we use mean/median units and volume as a metric?
# * More information on agents?  Age, Gender, City, Vehicle, Full/Part Time Status, Favorite Pizza Toppings

# ### Hypotheses
# * Standard Commission Type will positively correlate with units (higher commissions promote more sales activity)
# * Standard GCI will correlate with overall gross volume
# * Agents with more tenure will have larger overall gross volume
# * Agents with fewer units will not be as representative in overall data (should look to omit)
# * Delta between Listing Price and Sales 

# - [ ] get rid of times in date columns
# - [ ] rename column names to something easier to remember/use

# ## Environment 

# +
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# -

# ## Acquisition

df = pd.read_csv('agents_with_transactions.csv')

# ## Preparation

# ### Lowercasing all column names

df = df.rename(columns=lambda col: col.lower())

# ### Encoding various ID fields for readability and future usage.

# The raw values are long randomized strings whose uniqueness is the only thing of value.

# +
to_encode = ['agent_id', 'brokerage_id', 'transaction_id']

for col in to_encode:
    le = LabelEncoder().fit(df[col])
    df[col] = le.transform(df[col])
# -

# ### Correcting dates

# These dates were corrected because they had bad years. The corrected year for transaction_contracted_at was inferred from the dates for transaction_closed_at and transaction_effective_at columns. A similar approach was used for correcting transaction_effective_at column.

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

# ### Converting date columns to datetime type

df['transaction_contracted_at'] = pd.to_datetime(df.transaction_contracted_at, errors = 'coerce')
df['transaction_effective_at'] = pd.to_datetime(df.transaction_effective_at, errors = 'coerce')

# ### Column cleanup

# #### Remove bonus columns 

# These columns have 98%+ nulls

to_drop = [col for col in df if col.startswith('bonus_')]
df.drop(columns=to_drop, inplace=True)

# #### Drop standard_commission_type

# It only has one variable and nulls

df.drop(columns='standard_commission_type', inplace=True)

# #### Imputing nulls with 0

# Impute 0 for those rows where transaction_status is FELL_THROUGH; but we need to ask Ben why it's NaN for transactions that were COMPLETED

df.standard_commission_gci_amount.fillna(value=0, inplace=True)
df.standard_commission_brokerage_net_amount.fillna(value=0, inplace=True)

# ### Drop agent_name

# Agent ID provides a unique identifier.

df.drop(columns='agent_name', inplace=True)

# ### Drop brokerage name

# Brokerage ID provides a unique identifier.

df.drop(columns='brokerage_name', inplace=True)

# ### Drop commission_schedule_active

df.drop(columns='commission_schedule_active', inplace=True)

# ## Exploration

# ## Modeling

list(df.columns)


