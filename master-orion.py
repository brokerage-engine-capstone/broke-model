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
# <div class="toc"><ul class="toc-item"><li><span><a href="#Notes" data-toc-modified-id="Notes-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Notes</a></span><ul class="toc-item"><li><span><a href="#Questions" data-toc-modified-id="Questions-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Questions</a></span></li><li><span><a href="#Hypotheses" data-toc-modified-id="Hypotheses-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Hypotheses</a></span></li><li><span><a href="#To-do-list" data-toc-modified-id="To-do-list-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>To-do list</a></span></li><li><span><a href="#Deliverables" data-toc-modified-id="Deliverables-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Deliverables</a></span></li></ul></li><li><span><a href="#Environment" data-toc-modified-id="Environment-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Environment</a></span></li><li><span><a href="#Acquisition" data-toc-modified-id="Acquisition-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Acquisition</a></span></li><li><span><a href="#Preparation" data-toc-modified-id="Preparation-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Preparation</a></span><ul class="toc-item"><li><span><a href="#Lowercasing-all-column-names" data-toc-modified-id="Lowercasing-all-column-names-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Lowercasing all column names</a></span></li><li><span><a href="#Drop-Columns" data-toc-modified-id="Drop-Columns-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Drop Columns</a></span></li><li><span><a href="#Encoding-various-ID-fields-for-readability-and-future-usage." data-toc-modified-id="Encoding-various-ID-fields-for-readability-and-future-usage.-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Encoding various ID fields for readability and future usage.</a></span></li><li><span><a href="#Correcting-dates" data-toc-modified-id="Correcting-dates-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Correcting dates</a></span></li><li><span><a href="#Convert-dates-to-datetime-type" data-toc-modified-id="Convert-dates-to-datetime-type-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Convert dates to datetime type</a></span></li><li><span><a href="#Column-cleanup" data-toc-modified-id="Column-cleanup-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>Column cleanup</a></span><ul class="toc-item"><li><span><a href="#Remove-bonus-columns" data-toc-modified-id="Remove-bonus-columns-4.6.1"><span class="toc-item-num">4.6.1&nbsp;&nbsp;</span>Remove bonus columns</a></span></li><li><span><a href="#Imputing-nulls-with-0" data-toc-modified-id="Imputing-nulls-with-0-4.6.2"><span class="toc-item-num">4.6.2&nbsp;&nbsp;</span>Imputing nulls with 0</a></span></li></ul></li><li><span><a href="#Rename-Columns" data-toc-modified-id="Rename-Columns-4.7"><span class="toc-item-num">4.7&nbsp;&nbsp;</span>Rename Columns</a></span></li></ul></li><li><span><a href="#Exploration" data-toc-modified-id="Exploration-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Exploration</a></span></li><li><span><a href="#Modeling" data-toc-modified-id="Modeling-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Modeling</a></span></li></ul></div>
# -

# ## Notes

# - [ ] Work on definition of profitability
# - Maggie's thoughts
#     - we have all the data we need right now. look at previous top performers
#     - compare first five months in 2018 and in 2019. find what percentage they are in 2019 of 2018 amount. compare each agent's results to the total to see if there may be a market downturn.
#     - we may want to classify agents. are they a quantity seller or a quality?
#     - we may want to create two models. one for residential and one for commercial
# - [ ] filter out anything that says "lease"

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
#     * Yes
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

# ### To-do list
# - [ ] data dictionary

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
df_raw = pd.read_csv('agents_with_transactions3.csv')
df = df_raw.copy()
df_raw.shape
# -

# ## Preparation

# ### Lowercasing all column names

# + {"init_cell": true}
df.rename(columns=lambda col: col.lower(), inplace=True)
# -

# ### Drop Columns

# + {"init_cell": true}
# Agent ID provides a unique identifier already.
df.drop(columns='agent_name', inplace=True)
# Most of the data is missing and it's not clear what value this data provides.
df.drop(columns='commission_anniversary', inplace=True)
# Brokerage ID provides a unique identifier.
df.drop(columns='brokerage_name', inplace=True)
# It's not clear what value this data provides.
df.drop(columns='commission_schedule_id', inplace=True)
# Some rows say TRUE when they should be FALSE. We cannot make sense of it now
# and may want to revisit and correct these. This column may be useful if we add
# a feature representing the number of commission schedules the realtor has been through.
df.drop(columns='commission_schedule_active', inplace=True)
# This column provides the same information as transaction_id and transaction_effective_at.
df.drop(columns='transaction_number', inplace=True)
# The column transaction_effective_at provides better information.
df.drop(columns='transaction_closed_at', inplace=True)
# This column marks when the realtor/customer signed a contract which isn't
# helpful for this part of the project.
df.drop(columns='transaction_contracted_at', inplace=True)
# This column is has few non-null entries. We may get better information from another csv.
df.drop(columns='transaction_list_amount', inplace=True)
# It is either one value or null and is not useful.
df.drop(columns='standard_commission_type', inplace=True)
# This information is about commission adjustments, which we are not using at this time.
df.drop(columns='transaction_price_override', inplace=True)
# This column is not relevant to predicting our target variable. We may want to revisit
# these fees and liabilities columns to predict profitability.
df.drop(columns='total_fees_charged_against_agent', inplace=True)
# This column is not relevant to predicting our target variable. We may want to revisit these
# fees and liabilities columns to predict profitability.
df.drop(columns='total_fees_charged_against_brokerage', inplace=True)
# This column is not relevant to predicting our target variable. We may want to revisit these
# fees and liabilities columns to predict profitability.
df.drop(columns='total_liabilities_against_brokerage', inplace=True)
# This columns is mostly the product of earned_sales_amount and transaction_sales_amount.
df.drop(columns='earned_volume', inplace=True)
# This columns speak to when agents' commission plan starts and ends and is not necessary for our project.
df.drop(columns='commission_schedule_effective_start_at', inplace=True)
df.drop(columns='commission_schedule_effective_end_at', inplace=True)
# Dropping these as they're not useful for this type of predictive model.
df.drop(columns=(['total_fees_paid_on_transaction',
                  'total_liabilities_against_agent',
                  'total_liabilities_on_transaction',
                  'total_brokerage_income_collected_from_agent_fees',
                  'final_brokerage_income_after_all_liabilities_are_paid_out'

                  ]),
        inplace=True)
# -

# ### Encoding various ID fields for readability and future usage.

# The raw values are long randomized strings whose uniqueness is the only thing of value.

# + {"init_cell": true}
to_encode = ['agent_id',
             'brokerage_id',
             'transaction_id',
             'commission_schedule_strategy',
             'transaction_side',
             'account_id']

label_encs = dict()
for col in to_encode:
    le = LabelEncoder().fit(df[col])
    label_encs[col] = le
    df[col] = le.transform(df[col])
# -

# ### Correcting dates

# These dates were corrected because they had bad years. The corrected year for transaction_contracted_at was inferred from the dates for transaction_closed_at and transaction_effective_at columns. A similar approach was used for correcting transaction_effective_at column.

# + {"init_cell": true}
# Transactions contracted at errors fixed
# df.loc[df.transaction_id == 7686, 'transaction_contracted_at'] = "2018-07-31 12:00:00 UTC"
# df.loc[df.transaction_id == 3874, 'transaction_contracted_at'] = "2019-04-17 04:56:02 UTC"
# df.loc[df.transaction_id == 4084, 'transaction_contracted_at'] = "2019-04-22 04:56:02 UTC"
# df.loc[df.transaction_id == 11335, 'transaction_contracted_at'] = "2019-01-25 05:50:36 UTC"
# df.loc[df.transaction_id == 11400, 'transaction_contracted_at'] = "2019-01-29 12:00:00 UTC"
# df.loc[df.transaction_id == 12536, 'transaction_contracted_at'] = "2019-02-28 05:50:36 UTC"
# df.loc[df.transaction_id == 5539, 'transaction_contracted_at'] = "2019-04-30 12:00:00 UTC"
# df.loc[df.transaction_id == 1234, 'transaction_contracted_at'] = "2018-03-31 05:00:00 UTC"
# df.loc[df.transaction_id == 6642, 'transaction_contracted_at'] = "2018-05-31 12:00:00 UTC"
# df.loc[df.transaction_id == 7864, 'transaction_contracted_at'] = "2018-07-23 12:00:00 UTC"
# df.loc[df.transaction_id == 9186, 'transaction_contracted_at'] = "2018-11-15 05:50:36 UTC"
# df.loc[df.transaction_id == 2106, 'transaction_contracted_at'] = "2019-03-14 12:00:00 UTC"
# df.loc[df.transaction_id == 4976, 'transaction_contracted_at'] = "2019-05-12 12:00:00 UTC"
# df.loc[df.transaction_id == 10732, 'transaction_contracted_at'] = "2018-05-20 12:00:00 UTC"
# df.loc[df.transaction_id == 2153, 'transaction_contracted_at'] = "2018-12-20 12:00:00 UTC"
# df.loc[df.transaction_id == 3582, 'transaction_contracted_at'] = "2019-04-19 12:00:00 UTC"
# df.loc[df.transaction_id == 1153, 'transaction_contracted_at'] = "2018-07-28 05:50:36 UTC"
# df.loc[df.transaction_id == 2474, 'transaction_contracted_at'] = "2019-03-10 12:00:00 UTC"

# Transactions effective at errors fixed
df.loc[df.transaction_id == 8017, 'transaction_effective_at'] = "2018-08-31 12:00:00 UTC"
df.loc[df.transaction_id == 5675, 'transaction_effective_at'] = "2019-07-22 12:00:00 UTC"
df.loc[df.transaction_id == 3423, 'transaction_effective_at'] = "2019-11-12 12:00:00 UTC"
df.loc[df.transaction_id == 5141, 'transaction_effective_at'] = "2019-06-14 12:00:00 UTC"
# -

# ### Convert dates to datetime type
# Drop the time as well

# + {"init_cell": true}
df['transaction_effective_at'] = pd.to_datetime(df.transaction_effective_at, errors = 'coerce')
df.transaction_effective_at = df.transaction_effective_at.dt.date
df['transaction_effective_at'] = pd.to_datetime(df.transaction_effective_at, errors = 'coerce')
# -

# ### Column cleanup

# #### Remove bonus columns 
#
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

# ### Rename Columns
# They need shorter, more descriptive names.

# + {"init_cell": true}
new_col_names = {'commission_schedule_strategy': 'com_plan', 
                 'transaction_effective_at': 'sale_date', 
                 'transaction_status': 'sale_status', 
                 'transaction_sales_amount': 'sale_amount', 
                 'earned_side_count': 'com_split',
                 'tags': 'property_use',
                 'transaction_side': 'trans_side', 
                 'standard_commission_gci_amount': 'com_gross', 
                 'standard_commission_agent_net_amount': 'com_agent_net', 
                 'standard_commission_brokerage_net_amount': 'com_brokerage_net'
                }

df = df.rename(columns=new_col_names)
# -

# ## Exploration

# + {"init_cell": true}
df = df.assign(trans_year=df.sale_date.dt.year)
df = df.assign(trans_quarter=df.sale_date.dt.quarter)
agent_perf = df.groupby(["agent_id", "trans_year"])[["sale_amount"]].sum().reset_index()

# + {"init_cell": true}
df.property_use = df.property_use.fillna(value='Other')

# + {"init_cell": true}
df.property_use.value_counts(dropna=False)

# + {"init_cell": true}
df.loc[df.property_use == 'Residential|Residential', 'property_use'] = 'Residential'

# + {"init_cell": true}
df2 = df[(df.property_use == 'Residential') | (df.property_use == 'Other')]
# -

df2.property_use.value_counts()

df[df['sale_status'] == 'COMPLETE']\
    .groupby(['trans_quarter', 'trans_year'])\
    .mean()['sale_amount']

df[df['sale_status'] == 'COMPLETE']\
    .groupby(['agent_id', 'trans_year', 'trans_quarter'])\
    .count()['transaction_id']\
    .reset_index()\
    .groupby('trans_quarter')\
    .mean()['transaction_id']

df[df['sale_status'] == 'COMPLETE'].groupby('agent_id').count()['transaction_id'].value_counts(dropna=False)







# ## Modeling


