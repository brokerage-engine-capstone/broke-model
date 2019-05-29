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
# <div class="toc"><ul class="toc-item"><li><span><a href="#Notes" data-toc-modified-id="Notes-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Notes</a></span><ul class="toc-item"><li><span><a href="#Questions" data-toc-modified-id="Questions-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Questions</a></span></li><li><span><a href="#Hypotheses" data-toc-modified-id="Hypotheses-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Hypotheses</a></span></li><li><span><a href="#Deliverables" data-toc-modified-id="Deliverables-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Deliverables</a></span></li></ul></li><li><span><a href="#Environment" data-toc-modified-id="Environment-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Environment</a></span></li><li><span><a href="#Acquisition" data-toc-modified-id="Acquisition-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Acquisition</a></span></li><li><span><a href="#Preparation" data-toc-modified-id="Preparation-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Preparation</a></span><ul class="toc-item"><li><span><a href="#Lowercasing-all-column-names" data-toc-modified-id="Lowercasing-all-column-names-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Lowercasing all column names</a></span></li><li><span><a href="#Drop-Columns" data-toc-modified-id="Drop-Columns-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Drop Columns</a></span><ul class="toc-item"><li><span><a href="#agent_name" data-toc-modified-id="agent_name-4.2.1"><span class="toc-item-num">4.2.1&nbsp;&nbsp;</span>agent_name</a></span></li><li><span><a href="#commission_anniversary" data-toc-modified-id="commission_anniversary-4.2.2"><span class="toc-item-num">4.2.2&nbsp;&nbsp;</span>commission_anniversary</a></span></li><li><span><a href="#brokerage_name" data-toc-modified-id="brokerage_name-4.2.3"><span class="toc-item-num">4.2.3&nbsp;&nbsp;</span>brokerage_name</a></span></li><li><span><a href="#commission_schedule_id" data-toc-modified-id="commission_schedule_id-4.2.4"><span class="toc-item-num">4.2.4&nbsp;&nbsp;</span>commission_schedule_id</a></span></li><li><span><a href="#commission_schedule_active" data-toc-modified-id="commission_schedule_active-4.2.5"><span class="toc-item-num">4.2.5&nbsp;&nbsp;</span>commission_schedule_active</a></span></li><li><span><a href="#transaction_number" data-toc-modified-id="transaction_number-4.2.6"><span class="toc-item-num">4.2.6&nbsp;&nbsp;</span>transaction_number</a></span></li><li><span><a href="#transaction_closed_at" data-toc-modified-id="transaction_closed_at-4.2.7"><span class="toc-item-num">4.2.7&nbsp;&nbsp;</span>transaction_closed_at</a></span></li><li><span><a href="#transaction_list_amount" data-toc-modified-id="transaction_list_amount-4.2.8"><span class="toc-item-num">4.2.8&nbsp;&nbsp;</span>transaction_list_amount</a></span></li><li><span><a href="#standard_commission_type" data-toc-modified-id="standard_commission_type-4.2.9"><span class="toc-item-num">4.2.9&nbsp;&nbsp;</span>standard_commission_type</a></span></li></ul></li><li><span><a href="#Encoding-various-ID-fields-for-readability-and-future-usage." data-toc-modified-id="Encoding-various-ID-fields-for-readability-and-future-usage.-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Encoding various ID fields for readability and future usage.</a></span></li><li><span><a href="#Correcting-dates" data-toc-modified-id="Correcting-dates-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Correcting dates</a></span></li><li><span><a href="#Converting-date-columns-to-datetime-type" data-toc-modified-id="Converting-date-columns-to-datetime-type-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Converting date columns to datetime type</a></span></li><li><span><a href="#Column-cleanup" data-toc-modified-id="Column-cleanup-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>Column cleanup</a></span><ul class="toc-item"><li><span><a href="#Remove-bonus-columns" data-toc-modified-id="Remove-bonus-columns-4.6.1"><span class="toc-item-num">4.6.1&nbsp;&nbsp;</span>Remove bonus columns</a></span></li><li><span><a href="#Imputing-nulls-with-0" data-toc-modified-id="Imputing-nulls-with-0-4.6.2"><span class="toc-item-num">4.6.2&nbsp;&nbsp;</span>Imputing nulls with 0</a></span></li></ul></li></ul></li><li><span><a href="#Exploration" data-toc-modified-id="Exploration-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Exploration</a></span></li><li><span><a href="#Modeling" data-toc-modified-id="Modeling-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Modeling</a></span></li></ul></div>
# -

# ## Notes

# - [ ] Work on definition of profitability
# - [ ] Exploration
#     - [ ] Steven
# - [ ] Feature engineering ideas
#     - [ ] Orion
#     - [ ] Jason

# ### Questions
# * Brokerage Agency may have an impact on agent performance but we want to remain brokerage-agnostic.
# * Commission Schedule ID - is this relevant?
#     * We determined it was not, and have removed it.
# * Commission split and volume may be correlated - tenure may also have an impact.
# * What is Commission Schedule Active?  Should we drop?
#     * Yes, we dropped it.
# * What is Transaction Number?
# * If multiple rows have the same transaction ids and multiple agents, are these split?
# * What happens when an agent leaves the agency?
# * Listing price would be very valuable information.
# * Can we get a better understanding of agent tenure?
# * Is 5mil reasonable for a top performer?
# * What does BE want as a deliverable?
#     * They are open to whatever we develop.
# * Can we use mean/median units and volume as a metric?
# * More information on agents?  Age, Gender, City, Vehicle, Full/Part Time Status, Favorite Pizza Toppings
# * Why do we have NaNs in gci and brokerage_net for transactions that were COMPLETED?

# ### Hypotheses
# * Standard Commission Type will positively correlate with units (higher commissions promote more sales activity)
# * Standard GCI will correlate with overall gross volume
# * Agents with more tenure will have larger overall gross volume
# * Agents with fewer units will not be as representative in overall data (should look to omit)
# * Delta between Listing Price and Sales 

# - [ ] get rid of times in date columns
# - [ ] rename column names to something easier to remember/use

# ### Deliverables

# ## Environment 

# + {"init_cell": true}
from pprint import pprint

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

# + {"init_cell": true}
df_raw = pd.read_csv('agents_with_transactions2.csv')
df = df_raw.copy()
# -

# ## Preparation

# ### Lowercasing all column names

# + {"init_cell": true}
df.rename(columns=lambda col: col.lower(), inplace=True)
# -

# ### Drop Columns

# #### agent_name

# Agent ID provides a unique identifier already.

# + {"init_cell": true}
df.head()

# + {"init_cell": true}
df.drop(columns='agent_name', inplace=True)
# -

# #### commission_anniversary

# Most of the data is missing and it's not clear what value this data provides.

# + {"init_cell": true}
df.drop(columns='commission_anniversary', inplace=True)  # not useful information
# -

# #### brokerage_name

# Brokerage ID provides a unique identifier.

# + {"init_cell": true}
df.drop(columns='brokerage_name', inplace=True)
# -

# #### commission_schedule_id

# It's not clear what value this data provides.

# + {"init_cell": true}
df.drop(columns='commission_schedule_id', inplace=True)  # not useful information
# -

# #### commission_schedule_active

# Some rows say TRUE when they should be FALSE. We cannot make sense of it now and may want to revisit and correct these. This column may be useful if we add a feature representing the number of commission schedules the realtor has been through.

# + {"init_cell": true}
df.drop(columns='commission_schedule_active', inplace=True)
# -

# #### transaction_number

# This column provides the same information as transaction_id and transaction_effective_at.

# + {"init_cell": true}
df.drop(columns='transaction_number', inplace=True)
# -

# #### transaction_closed_at

# The column transaction_effective_at provides better information.

# + {"init_cell": true}
df.drop(columns='transaction_closed_at', inplace=True)
# -

# #### transaction_list_amount

# This column is has few non-null entries. We may get better information from another csv.

# + {"init_cell": true}
df.drop(columns='transaction_list_amount', inplace=True)
# -

# #### standard_commission_type

# It is either one value or null and is not useful.

# + {"init_cell": true}
df.drop(columns='standard_commission_type', inplace=True)
# -

# ### Encoding various ID fields for readability and future usage.

# The raw values are long randomized strings whose uniqueness is the only thing of value.

# + {"init_cell": true}
to_encode = ['agent_id', 'account_id', 'brokerage_id', 'transaction_id', 'commission_schedule_strategy', 'transaction_side']

for col in to_encode:
    le = LabelEncoder().fit(df[col])
    df[col] = le.transform(df[col])
# -

# ### Correcting dates

# These dates were corrected because they had bad years. The corrected year for transaction_contracted_at was inferred from the dates for transaction_closed_at and transaction_effective_at columns. A similar approach was used for correcting transaction_effective_at column.

# + {"init_cell": true}
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

# + {"init_cell": true}
df['commission_schedule_effective_start_at'] = pd.to_datetime(df.commission_schedule_effective_start_at, errors = 'coerce')
df['commission_schedule_efffective_end_at'] = pd.to_datetime(df.commission_schedule_effective_end_at, errors = 'coerce')
df['transaction_contracted_at'] = pd.to_datetime(df.transaction_contracted_at, errors = 'coerce')
df['transaction_effective_at'] = pd.to_datetime(df.transaction_effective_at, errors = 'coerce')
# -

df.info()

# + {"init_cell": true}
df.head()
# -

# ### Column cleanup

# #### Remove bonus columns 

# These columns have 98%+ nulls

# + {"init_cell": true}
to_drop = [col for col in df if col.startswith('bonus_')]
df.drop(columns=to_drop, inplace=True)
# -

# #### Imputing nulls with 0

# Impute 0 for those rows where transaction_status is FELL_THROUGH

# + {"init_cell": true}
df.standard_commission_gci_amount.fillna(value=0, inplace=True)
df.standard_commission_brokerage_net_amount.fillna(value=0, inplace=True)
# -



df.transaction_effective_at.head()

df.head()

df = df.assign(trans_year=df.transaction_effective_at.dt.year)
df = df.assign(trans_quarter=df.transaction_effective_at.dt.quarter)
agent_perf = df.groupby(["agent_id", "trans_year"])[["transaction_sales_amount"]].sum().reset_index()

# Looking at the tags column, we're going to replace nulls with 'Other'

df.tags = df.tags.fillna(value='Other')

df.tags.value_counts(dropna=False)

df.trans_quarter.value_counts()

625786.19 / 506489.65


# +
df[df['transaction_status'] == 'COMPLETE']\
    .groupby(['trans_quarter', 'trans_year'])\
    .mean()['transaction_sales_amount']

# ['transaction_sales_amount']\
#     .reset_index().groupby('trans_year')\
#     .mean()['transaction_sales_amount']
# -

df[df['transaction_status'] == 'COMPLETE']\
    .groupby(['agent_id', 'trans_year', 'trans_quarter'])\
    .count()['transaction_id']\
    .reset_index()\
    .groupby('trans_quarter')\
    .mean()['transaction_id']

df[df['transaction_status'] == 'COMPLETE'].groupby('agent_id').count()['transaction_id'].value_counts(dropna=False)














# ## Exploration

# ## Modeling


