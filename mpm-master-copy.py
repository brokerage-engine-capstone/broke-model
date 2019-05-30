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
# <div class="toc"><ul class="toc-item"><li><span><a href="#Notes" data-toc-modified-id="Notes-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Notes</a></span><ul class="toc-item"><li><span><a href="#Questions" data-toc-modified-id="Questions-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Questions</a></span></li><li><span><a href="#Hypotheses" data-toc-modified-id="Hypotheses-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Hypotheses</a></span></li><li><span><a href="#Deliverables" data-toc-modified-id="Deliverables-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Deliverables</a></span></li></ul></li><li><span><a href="#Environment" data-toc-modified-id="Environment-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Environment</a></span></li><li><span><a href="#Acquisition" data-toc-modified-id="Acquisition-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Acquisition</a></span></li><li><span><a href="#Preparation" data-toc-modified-id="Preparation-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Preparation</a></span><ul class="toc-item"><li><span><a href="#Lowercasing-all-column-names" data-toc-modified-id="Lowercasing-all-column-names-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Lowercasing all column names</a></span></li><li><span><a href="#Drop-Columns" data-toc-modified-id="Drop-Columns-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Drop Columns</a></span><ul class="toc-item"><li><span><a href="#agent_name" data-toc-modified-id="agent_name-4.2.1"><span class="toc-item-num">4.2.1&nbsp;&nbsp;</span>agent_name</a></span></li><li><span><a href="#commission_anniversary" data-toc-modified-id="commission_anniversary-4.2.2"><span class="toc-item-num">4.2.2&nbsp;&nbsp;</span>commission_anniversary</a></span></li><li><span><a href="#brokerage_name" data-toc-modified-id="brokerage_name-4.2.3"><span class="toc-item-num">4.2.3&nbsp;&nbsp;</span>brokerage_name</a></span></li><li><span><a href="#commission_schedule_id" data-toc-modified-id="commission_schedule_id-4.2.4"><span class="toc-item-num">4.2.4&nbsp;&nbsp;</span>commission_schedule_id</a></span></li><li><span><a href="#commission_schedule_active" data-toc-modified-id="commission_schedule_active-4.2.5"><span class="toc-item-num">4.2.5&nbsp;&nbsp;</span>commission_schedule_active</a></span></li><li><span><a href="#transaction_number" data-toc-modified-id="transaction_number-4.2.6"><span class="toc-item-num">4.2.6&nbsp;&nbsp;</span>transaction_number</a></span></li><li><span><a href="#transaction_contracted_at" data-toc-modified-id="transaction_contracted_at-4.2.7"><span class="toc-item-num">4.2.7&nbsp;&nbsp;</span>transaction_contracted_at</a></span></li><li><span><a href="#transaction_closed_at" data-toc-modified-id="transaction_closed_at-4.2.8"><span class="toc-item-num">4.2.8&nbsp;&nbsp;</span>transaction_closed_at</a></span></li><li><span><a href="#transaction_list_amount" data-toc-modified-id="transaction_list_amount-4.2.9"><span class="toc-item-num">4.2.9&nbsp;&nbsp;</span>transaction_list_amount</a></span></li><li><span><a href="#transaction_price_override" data-toc-modified-id="transaction_price_override-4.2.10"><span class="toc-item-num">4.2.10&nbsp;&nbsp;</span>transaction_price_override</a></span></li><li><span><a href="#standard_commission_type" data-toc-modified-id="standard_commission_type-4.2.11"><span class="toc-item-num">4.2.11&nbsp;&nbsp;</span>standard_commission_type</a></span></li><li><span><a href="#total_fees_against_agent" data-toc-modified-id="total_fees_against_agent-4.2.12"><span class="toc-item-num">4.2.12&nbsp;&nbsp;</span>total_fees_against_agent</a></span></li><li><span><a href="#total_fees_against_brokerage" data-toc-modified-id="total_fees_against_brokerage-4.2.13"><span class="toc-item-num">4.2.13&nbsp;&nbsp;</span>total_fees_against_brokerage</a></span></li><li><span><a href="#total_liabilities_against_brokerage" data-toc-modified-id="total_liabilities_against_brokerage-4.2.14"><span class="toc-item-num">4.2.14&nbsp;&nbsp;</span>total_liabilities_against_brokerage</a></span></li></ul></li><li><span><a href="#Encoding-various-ID-fields-for-readability-and-future-usage." data-toc-modified-id="Encoding-various-ID-fields-for-readability-and-future-usage.-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Encoding various ID fields for readability and future usage.</a></span></li><li><span><a href="#Correcting-dates" data-toc-modified-id="Correcting-dates-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Correcting dates</a></span></li><li><span><a href="#Converting-date-columns-to-datetime-type" data-toc-modified-id="Converting-date-columns-to-datetime-type-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Converting date columns to datetime type</a></span></li><li><span><a href="#Column-cleanup" data-toc-modified-id="Column-cleanup-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>Column cleanup</a></span><ul class="toc-item"><li><span><a href="#Remove-bonus-columns" data-toc-modified-id="Remove-bonus-columns-4.6.1"><span class="toc-item-num">4.6.1&nbsp;&nbsp;</span>Remove bonus columns</a></span></li><li><span><a href="#Imputing-nulls-with-0" data-toc-modified-id="Imputing-nulls-with-0-4.6.2"><span class="toc-item-num">4.6.2&nbsp;&nbsp;</span>Imputing nulls with 0</a></span></li></ul></li><li><span><a href="#Sanity-Check" data-toc-modified-id="Sanity-Check-4.7"><span class="toc-item-num">4.7&nbsp;&nbsp;</span>Sanity Check</a></span></li></ul></li><li><span><a href="#Exploration" data-toc-modified-id="Exploration-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Exploration</a></span></li><li><span><a href="#Modeling" data-toc-modified-id="Modeling-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Modeling</a></span></li></ul></div>
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

# - [ ] rename column names to something easier to remember/use

# ### Deliverables

# ## Environment 

# +
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

df_raw = pd.read_csv('agents_with_transactions_05-29-19.csv')
df = df_raw.copy()

df_raw.shape

# ## Preparation

# ### Lowercasing all column names

df.rename(columns=lambda col: col.lower(), inplace=True)

# ### Drop Columns

# #### agent_name

# Agent ID provides a unique identifier already.

df.drop(columns='agent_name', inplace=True)

# #### commission_anniversary

# Most of the data is missing and it's not clear what value this data provides.

df.drop(columns='commission_anniversary', inplace=True)

# #### brokerage_name

# Brokerage ID provides a unique identifier.

df.drop(columns='brokerage_name', inplace=True)

# #### commission_schedule_id

# It's not clear what value this data provides.

df.drop(columns='commission_schedule_id', inplace=True)

# #### commission_schedule_active

# Some rows say TRUE when they should be FALSE. We cannot make sense of it now and may want to revisit and correct these. This column may be useful if we add a feature representing the number of commission schedules the realtor has been through.

df.drop(columns='commission_schedule_active', inplace=True)

# #### transaction_number

# This column provides the same information as transaction_id and transaction_effective_at.

df.drop(columns='transaction_number', inplace=True)

# #### transaction_contracted_at

df.drop(columns='transaction_contracted_at', inplace=True)

# #### transaction_closed_at

# The column transaction_effective_at provides better information.

df.drop(columns='transaction_closed_at', inplace=True)

# #### transaction_list_amount

# This column is has few non-null entries. We may get better information from another csv.

df.drop(columns='transaction_list_amount', inplace=True)

# #### transaction_price_override
#
# This information is about commission adjustments, which we are not using at this time.

df.drop(columns='transaction_price_override', inplace=True)

# #### standard_commission_type
#
# It is either one value or null and is not useful.

df.drop(columns='standard_commission_type', inplace=True)

# #### total_fees_against_agent
#
# This column is not relevant to predicting our target variable. We may want to revisit these fees and liabilities columns to predict profitability.

df.drop(columns='total_fees_against_agent', inplace=True)

# #### total_fees_against_brokerage
#
# This column is not relevant to predicting our target variable. We may want to revisit these fees and liabilities columns to predict profitability.

df.drop(columns='total_fees_against_brokerage', inplace=True)

# #### total_liabilities_against_brokerage
#
# This column is not relevant to predicting our target variable. We may want to revisit these fees and liabilities columns to predict profitability.

df.drop(columns='total_liabilities_against_brokerage', inplace=True)

# ### Encoding various ID fields for readability and future usage.

# The raw values are long randomized strings whose uniqueness is the only thing of value.

to_encode = ['agent_id', 'brokerage_id', 'transaction_id', 'commission_schedule_strategy', 'transaction_side',
             'account_id']
label_encs = dict()
for col in to_encode:
    le = LabelEncoder().fit(df[col])
    label_encs[col] = le
    df[col] = le.transform(df[col])

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

df['commission_schedule_effective_start_at'] = pd.to_datetime(df.commission_schedule_effective_start_at, errors = 'coerce')
df['commission_schedule_efffective_end_at'] = pd.to_datetime(df.commission_schedule_effective_end_at, errors = 'coerce')
df['transaction_contracted_at'] = pd.to_datetime(df.transaction_contracted_at, errors = 'coerce')
df['transaction_effective_at'] = pd.to_datetime(df.transaction_effective_at, errors = 'coerce')

df.commission_schedule_effective_start_at = df.commission_schedule_effective_start_at.dt.date
df.commission_schedule_efffective_end_at = df.commission_schedule_efffective_end_at.dt.date
df.transaction_contracted_at = df.transaction_contracted_at.dt.date
df.transaction_effective_at = df.transaction_effective_at.dt.date

df.head()

df.assign(transdiff=(df.transaction_effective_at - df.transaction_contracted_at)).sort_values(by="transdiff", ascending=False)

# ### Column cleanup

# #### Remove bonus columns 

# These columns have 98%+ nulls

to_drop = [col for col in df if col.startswith('bonus_')]
df.drop(columns=to_drop, inplace=True)

# #### Imputing nulls with 0

# Impute 0 for those rows where transaction_status is FELL_THROUGH

df.standard_commission_gci_amount.fillna(value=0, inplace=True)
df.standard_commission_brokerage_net_amount.fillna(value=0, inplace=True)

# ### Sanity Check

df.shape

# ## Exploration

# ## Modeling


