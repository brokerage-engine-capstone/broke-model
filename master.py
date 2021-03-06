# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"toc": true, "cell_type": "markdown"}
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Notes" data-toc-modified-id="Notes-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Notes</a></span><ul class="toc-item"><li><span><a href="#Questions" data-toc-modified-id="Questions-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Questions</a></span></li><li><span><a href="#Hypotheses" data-toc-modified-id="Hypotheses-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Hypotheses</a></span></li><li><span><a href="#To-do-list" data-toc-modified-id="To-do-list-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>To-do list</a></span></li><li><span><a href="#Deliverables" data-toc-modified-id="Deliverables-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Deliverables</a></span></li><li><span><a href="#Data-Dictionary" data-toc-modified-id="Data-Dictionary-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Data Dictionary</a></span></li></ul></li><li><span><a href="#Environment" data-toc-modified-id="Environment-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Environment</a></span></li><li><span><a href="#Acquisition" data-toc-modified-id="Acquisition-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Acquisition</a></span></li><li><span><a href="#Preparation" data-toc-modified-id="Preparation-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Preparation</a></span><ul class="toc-item"><li><span><a href="#Lowercase-all-column-names" data-toc-modified-id="Lowercase-all-column-names-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Lowercase all column names</a></span></li><li><span><a href="#Drop-Columns" data-toc-modified-id="Drop-Columns-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Drop Columns</a></span></li><li><span><a href="#Rename-Columns" data-toc-modified-id="Rename-Columns-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Rename Columns</a></span></li><li><span><a href="#Select-Finished-Transactions" data-toc-modified-id="Select-Finished-Transactions-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Select Finished Transactions</a></span></li><li><span><a href="#Drop-Outliers" data-toc-modified-id="Drop-Outliers-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Drop Outliers</a></span></li><li><span><a href="#Encoding-columns" data-toc-modified-id="Encoding-columns-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>Encoding columns</a></span><ul class="toc-item"><li><span><a href="#Various-ID-fields-and-categorical-variables" data-toc-modified-id="Various-ID-fields-and-categorical-variables-4.6.1"><span class="toc-item-num">4.6.1&nbsp;&nbsp;</span>Various ID fields and categorical variables</a></span></li></ul></li><li><span><a href="#Date-Columns" data-toc-modified-id="Date-Columns-4.7"><span class="toc-item-num">4.7&nbsp;&nbsp;</span>Date Columns</a></span><ul class="toc-item"><li><span><a href="#Correcting-dates" data-toc-modified-id="Correcting-dates-4.7.1"><span class="toc-item-num">4.7.1&nbsp;&nbsp;</span>Correcting dates</a></span></li><li><span><a href="#Convert-dates-to-datetime-type" data-toc-modified-id="Convert-dates-to-datetime-type-4.7.2"><span class="toc-item-num">4.7.2&nbsp;&nbsp;</span>Convert dates to datetime type</a></span></li></ul></li><li><span><a href="#Miscellaneous-Work-on-Columns" data-toc-modified-id="Miscellaneous-Work-on-Columns-4.8"><span class="toc-item-num">4.8&nbsp;&nbsp;</span>Miscellaneous Work on Columns</a></span></li><li><span><a href="#Create-2018-DataFrame" data-toc-modified-id="Create-2018-DataFrame-4.9"><span class="toc-item-num">4.9&nbsp;&nbsp;</span>Create 2018 DataFrame</a></span></li><li><span><a href="#Data-Sanity/Validation-Checks" data-toc-modified-id="Data-Sanity/Validation-Checks-4.10"><span class="toc-item-num">4.10&nbsp;&nbsp;</span>Data Sanity/Validation Checks</a></span></li></ul></li><li><span><a href="#Exploration" data-toc-modified-id="Exploration-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Exploration</a></span></li><li><span><a href="#Modeling" data-toc-modified-id="Modeling-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Modeling</a></span><ul class="toc-item"><li><span><a href="#Encoding-columns" data-toc-modified-id="Encoding-columns-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Encoding columns</a></span><ul class="toc-item"><li><span><a href="#Various-ID-fields-and-categorical-variables" data-toc-modified-id="Various-ID-fields-and-categorical-variables-6.1.1"><span class="toc-item-num">6.1.1&nbsp;&nbsp;</span>Various ID fields and categorical variables</a></span></li></ul></li><li><span><a href="#Train-test-Split" data-toc-modified-id="Train-test-Split-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Train-test Split</a></span><ul class="toc-item"><li><span><a href="#Logistic-Regression" data-toc-modified-id="Logistic-Regression-6.2.1"><span class="toc-item-num">6.2.1&nbsp;&nbsp;</span>Logistic Regression</a></span></li><li><span><a href="#Decision-Tree" data-toc-modified-id="Decision-Tree-6.2.2"><span class="toc-item-num">6.2.2&nbsp;&nbsp;</span>Decision Tree</a></span></li><li><span><a href="#KNN" data-toc-modified-id="KNN-6.2.3"><span class="toc-item-num">6.2.3&nbsp;&nbsp;</span>KNN</a></span></li><li><span><a href="#Random-Forest" data-toc-modified-id="Random-Forest-6.2.4"><span class="toc-item-num">6.2.4&nbsp;&nbsp;</span>Random Forest</a></span></li></ul></li><li><span><a href="#K-Fold-Cross-Validation" data-toc-modified-id="K-Fold-Cross-Validation-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>K-Fold Cross Validation</a></span><ul class="toc-item"><li><span><a href="#Logistic-Regression" data-toc-modified-id="Logistic-Regression-6.3.1"><span class="toc-item-num">6.3.1&nbsp;&nbsp;</span>Logistic Regression</a></span></li><li><span><a href="#Decision-Tree" data-toc-modified-id="Decision-Tree-6.3.2"><span class="toc-item-num">6.3.2&nbsp;&nbsp;</span>Decision Tree</a></span></li></ul></li></ul></li></ul></div>
# -

# ## Notes

# - Maggie's thoughts
#     - we have all the data we need right now. look at previous top performers
#     - compare first five months in 2018 and in 2019. find what percentage they are in 2019 of 2018 amount. compare each agent's results to the total to see if there may be a market downturn.
#     - we may want to classify agents. are they a quantity seller or a quality?
#     - we may want to create two models. one for residential and one for commercial
#     - keep with $5mil; we can keep it as binary or bin it (quartiles);
#     - zip code data would be nice to see what each realtor's market is like
#     - average days on market might be a good predictor
#     - shot down gci_amount because sales_price is a raw measurement; commission amount is influenced by the type of brokerage not by the skills of the agent.
# - we are selecting only residential properties
# - we have 20/50 rule rather than 20/80. top 20 agents are bringing 50 percent of brokerage income
#
# - We want to optimize for recall. We do not want to miss investing in a high performer because of the return. It's OK if precision suffers a little. At worst we invest in someone who is not going to be a high performer. However, this loss is better than losing out on the return from a high performer. How can we determine how much a brokerage has invested in an agent?
# - Resolution of disagreements
#     - there shouldn't be enough features to derive the dependent variable
#     - When K-folding, do train-test split
#     - drop commercial for now; but come back if we have time
#         - clarify whether $5mil+ is considered high performance for commercial realtor
# - Ben recommends classifying realtors according to what type of properties they usually sell.
# - We will do two approaches to our dependent variable: binary and quintiles

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
#     * Yes for residential, but we do not know for 
# * What does BE want as a deliverable?
#     * They are open to whatever we develop.
# * Can we use mean/median units and volume as a metric?
# * More information on agents?  Age, Gender, City, Vehicle, Full/Part Time Status, Favorite Pizza Toppings
# * Why do we have NaNs in gci and brokerage_net for transactions that were COMPLETED?
# * What does "other" mean for tags/property_use (we will drop it until we get clarification)?
# * Which homes in terms of price are the best to sell?
# * What is the rolling average of home sales price by month
# * what are the key differences between commercial property sales and residential
# * did any agents get \$1.6 million in first trimester of 2018 but not 2019?
# * is the length of the commission schedule indicative of a high performer? (commission_schedule_effective_end_at - start_at)

# ### Hypotheses
# * Standard Commission Type will positively correlate with units (higher commissions promote more sales activity)
# * Standard GCI will correlate with overall gross volume
# * Agents with more tenure will have larger overall gross volume
# * Agents with fewer units will not be as representative in overall data (should look to omit)
# * Delta between Listing Price and Sales 

# ### To-do list
# - ROC curve for models
# - K-fold cross validation
# - use grid search to tune hyperparameters (https://scikit-learn.org/stable/modules/grid_search.html)
# - [ ] Stephen - naive model
# - [ ] Michael - explore leases
# - [ ] Orion - explore observations where there is no sale amount but there is an override amount
# - [ ] Jason - explore whether realtors can be classified by property type they sell

# ### Deliverables

# ### Data Dictionary

# - **agent_id** -> unique identifier for each agent
# - **agent_name** -> bogus name for anonymity reasons
# - **commission_anniversary** -> when the time comes to renegotiate the agent's split rate
# - **account_id** -> unique identifier of parent company
# - **brokerage_id** -> unique identifier for a subunit of a parent company (i.e., each brokerage_id is a subunit of account_id). A brokerage is a subunit of an agency.
# - **brokerage_name** -> bogus name for anonymity reasons
# - **commission_schedule_id** -> unique identifier for a commission schedule
# - **commission_schedule_effective_start_at** -> when the commission schedule starts
# - **commission_schedule_effective_end_at** -> when the commission schedule ends
# - **commission_schedule_active** -> whether the commmission schedule is active or not
# - **commission_schedule_strategy/com_plan** -> how the commission is calculated
#     - Mapping of encoder:
#         - 0: Accumulation Strategy
#         - 1: Flat Rate Strategy
#         - 2: Rolling Accumulation Strategy
# - **transaction_id/trans_id** -> a unique identifier for the transaction
# - **transaction_number** -> BT 'year' 'month' 'day' 'transaction_count'
# 	- unique to a single account, use transaction_id
# - **transaction_contracted_at** -> when the buyers and sellers signed contract to begin transaction
# - **transaction_closed_at** -> when the transaction was closed
# - **transaction_effective_at/sale_date** -> an override for when the transaction actually closed (there might be some last minute changes)
# - **transaction_status/sale_status** -> Open, ~DONT WORRY ABOUT THESE~ ,Cda Sent, Complete, or Fell through
# - **transaction_sales_amount/sale_amount** ->
# - **transaction_list_amount** -> set by users (WILL TRY TO PULL FROM LISTING SIDE SYSTEM)
# - **earned_side_count/com_split** -> a strange representation of how much credit an agent gets for a transaction
#     - can be split between agents on a side
# - **earned_volume** -> typically is the same as the sales amount
# 	- can be split between agents on a side
# - **tags/property_use** -> usage of the property (i.e., residential)
# - **transaction_side/trans_side** ->
# 	- Listing Side -> The agent is representing the seller of the property
# 	- Selling Side -> The agent is representing the buyer of the property
# - **transaction_price_override/price_override** ->
# - **standard_commission_type** -> the regular payout from a transaction
# - **standard_commission_gci_amount/com_gross** ->
# 	- the 3% || base value the brokerage took as commission
# 	- before splitting with agent
# - **standard_commission_agent_net_amount/com_agent_net** -> how much the agent took home
# - **standard_commission_brokerage_net_amount/com_brokerage_net** -> how much the brokerage took
# - **total_fees_charged_against_brokerage** -> the sum total of all the fees charged to the brokerage
# - **total_fees_charged_against_agent** -> the sum total of all fees charged to the agent; these can be paid by the brokerage
# - **total_fees_paid_on_transaction** -> $total\_fees\_charged\_against\_agent + total\_fees\_charged\_against\_brokerage$
# - **total_liabilities_against_brokerage** -> the amount the brokerage is liable to pay out to other parties.
# 	- These parties include the franchise, marketing, vendors, brokerage concessions and many more...
# - **total_liabilities_against_agent** -> the amount the agent is liable to pay to other parties.
# - **total_liabilities_on_transaction** -> $total\_liabilities\_against\_brokerage + total\_liabilities\_against\_the\_agent$
# - **total_brokerage_income_collected_from_agent_fees** -> $total\_fees\_charged\_against\_agent - total\_liabilities\_against\_the\_agent$
# - **final_brokerage_income_after_all_liabilities_are_paid_out** -> $brokerage\_net\_amount + total\_brokerage\_income\_collected\_from\_agent\_fees - total\_liabilities\_on\_transaction$
# - **bonus_commission_type** -> any extra money paid out
# - **bonus_commission_agent_net_amount** ->
# - **bonus_commission_brokerage_net_amount** ->

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

# modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import adalib
# -

# ## Acquisition

df_raw = pd.read_csv('agents_with_transactions.csv')
df = df_raw.copy()
df_raw.shape

# ## Preparation

# ### Lowercase all column names

df.rename(columns=lambda col: col.lower(), inplace=True)

# ### Drop Columns

# +
# Agent ID provides a unique identifier already.
df.drop(columns='agent_name', inplace=True)

# Most of the data is missing and it's not clear what value this data provides.
df.drop(columns='commission_anniversary', inplace=True)

# Brokerage ID provides a unique identifier.
df.drop(columns='brokerage_name', inplace=True)

# It's not clear what value this data provides; there's 3554 unique schedule id's. Many agents each have more than one commission schedule id.
df.drop(columns='commission_schedule_id', inplace=True)

# This columns speak to when agents' commission plan starts and ends and is not necessary for our project.
df.drop(columns='commission_schedule_effective_start_at', inplace=True)
df.drop(columns='commission_schedule_effective_end_at', inplace=True)

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

# This column has few non-null entries. We may get better information from another csv.
df.drop(columns='transaction_list_amount', inplace=True)

# Unreliable data; we don't know if the value reflects reality.
df.drop(columns='earned_side_count', inplace=True)

# This columns is mostly the product of earned_sales_amount and transaction_sales_amount.
df.drop(columns='earned_volume', inplace=True)

# It is either one value or null and is not useful.
df.drop(columns='standard_commission_type', inplace=True)

# This information is about commission adjustments. We have decided to keep this after all
# df.drop(columns='transaction_price_override', inplace=True)

# Post split and too many unknowns going into these values
df.drop(columns=['total_fees_charged_against_brokerage',
                  'total_fees_charged_against_agent',
                  'total_fees_paid_on_transaction',
                  'total_liabilities_against_brokerage',
                  'total_liabilities_against_agent',
                  'total_liabilities_on_transaction',
                  'total_brokerage_income_collected_from_agent_fees',
                  'final_brokerage_income_after_all_liabilities_are_paid_out',], inplace=True)

# bonuses are mostly NaNs
df.drop(columns=['bonus_commission_type',
                 'bonus_commission_agent_net_amount',
                 'bonus_commission_brokerage_net_amount',], inplace=True)
# -

# ### Rename Columns
# They need shorter, more descriptive names.

# +
new_col_names = {'commission_schedule_strategy': 'com_plan', 
                 'transaction_id': 'trans_id',
                 'transaction_effective_at': 'sale_date', 
                 'transaction_status': 'sale_status', 
                 'transaction_sales_amount': 'sale_amount', 
                 'earned_side_count': 'com_split',
                 'tags': 'property_use',
                 'transaction_side': 'trans_side',
                 'transaction_price_override': 'price_override',
                 'standard_commission_gci_amount': 'com_gross', 
                 'standard_commission_agent_net_amount': 'com_agent_net', 
                 'standard_commission_brokerage_net_amount': 'com_brokerage_net'
                }

df = df.rename(columns=new_col_names)
# -

pprint(list(df.columns))

# ### Select Finished Transactions

df = df[df.sale_status == 'COMPLETE']

# ### Drop Outliers

# Drop row where sales_amount is > $1 billion because at least one of the transactions was more than this amount and it was an entry error. We are keeping this code in case it creeps back in.

df = df[df.sale_amount < 1_000_000_000]

# ### Encoding columns

# #### Various ID fields and categorical variables

# For readability and future usage
#
# The raw values for the \_id columns are long randomized strings whose uniqueness is the only thing of value.
#
# commission_schedule_strategy and transaction_side are categorical variables.

# +
to_encode = ['agent_id',
             'brokerage_id',
             'trans_id',
             'account_id']

label_encs = dict()
for col in to_encode:
    le = LabelEncoder().fit(df[col])
    label_encs[col] = le
    df[col] = le.transform(df[col])
# -

# ### Date Columns

# #### Correcting dates
# These dates were corrected because they had bad years. The corrected year for transaction_contracted_at was inferred from the dates for transaction_closed_at and transaction_effective_at columns. A similar approach was used for correcting transaction_effective_at column.

# Transactions effective at errors fixed
df.loc[df.trans_id == 8017, 'sale_date'] = "2018-08-31 12:00:00 UTC"
df.loc[df.trans_id == 5675, 'sale_date'] = "2019-07-22 12:00:00 UTC"
df.loc[df.trans_id == 3423, 'sale_date'] = "2019-11-12 12:00:00 UTC"
df.loc[df.trans_id == 5141, 'sale_date'] = "2019-06-14 12:00:00 UTC"
df.loc[df.trans_id == 6246, 'sale_date'] = "2018-08-31 12:00:00 UTC"

# #### Convert dates to datetime type
# Drop the time as well

# df['commission_schedule_effective_start_at'] = pd.to_datetime(df.commission_schedule_effective_start_at, errors = 'coerce')
# df['commission_schedule_efffective_end_at'] = pd.to_datetime(df.commission_schedule_effective_end_at, errors = 'coerce')
# df['transaction_contracted_at'] = pd.to_datetime(df.transaction_contracted_at, errors = 'coerce')
df['sale_date'] = pd.to_datetime(df.sale_date, errors = 'coerce')
df.sale_date = df.sale_date.dt.date
df['sale_date'] = pd.to_datetime(df.sale_date, errors = 'coerce')

# ### Miscellaneous Work on Columns

# Add columns for year and quarter of sale

df = df.assign(sale_year=df.sale_date.dt.year)
df = df.assign(sale_quarter=df.sale_date.dt.quarter)

# We don't know what usage these properties are being put to.

df.property_use = df.property_use.fillna(value='Unknown')

# Remove transactions where property use is unknown and sale amount is less than 25k. We don't want leases

print("Before drop:", len(df))
df = df[(df.sale_amount > 20_000)]
print("After drop:", len(df))

# Eliminate double residential in property_use column

df.loc[df.property_use == 'Residential|Residential', 'property_use'] = 'Residential'

# ### Create 2018 DataFrame

# Get only 2018 data

df_2018 = df[df['sale_year']==2018]

# Calculate summary statistics

# Counting transactions
agent_df = df_2018.groupby('agent_id')[["agent_id"]].count()
# Sum
sum_df = df_2018.groupby('agent_id')[['sale_amount', 'com_agent_net','com_brokerage_net']].sum()
# Avg
mean_df = df_2018.groupby('agent_id')[['sale_amount', 'com_agent_net','com_brokerage_net']].mean()

# Rename columns for join

# +
# Rename columns for join
agent_df = agent_df.rename(columns={'sale_status': 'trans_count'})

sum_df = sum_df.rename(columns={'sale_amount': 'sum_sales',
                                'com_agent_net': 'sum_agent_com',
                                'com_brokerage_net': 'sum_brokerage_com'})

mean_df = mean_df.rename(columns={'sale_amount': 'avg_sales',
                                  'com_agent_net': 'avg_agent_com',
                                  'com_brokerage_net': 'avg_brokerage_com'})
# -

# Join into one dataframe

agent_df = agent_df.join(sum_df).join(mean_df)

# Add high performer (over 5 mil sum sales) categorical variable

agent_df['high_performer'] = (agent_df.sum_sales > 5_000_000).astype(int)

# ### Data Sanity/Validation Checks

df.isnull().sum()

# ## Exploration

# ## Modeling

# ### Encoding columns

# #### Various ID fields and categorical variables

# Commission_schedule_strategy and transaction_side are categorical variables.

# +
to_encode2 = ['com_plan',
             'trans_side',]

for col in to_encode2:
    le = LabelEncoder().fit(df[col])
    label_encs[col] = le
    df[col] = le.transform(df[col])
# -

# ### Train-test Split

# +
X = agent_df.drop(columns=['high_performer', 'sum_sales'])
y = agent_df[['high_performer']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)
# -

# #### Logistic Regression

# +
logit = LogisticRegression(random_state=42,
                           solver='saga')
logit.fit(X_train, y_train)
y_pred = logit.predict(X_train)
y_pred_proba = pd.DataFrame(logit.predict_proba(X_train), columns=("not_hp", "hp"))
proba_df = pd.DataFrame(y_pred_proba)
print(type(y_pred_proba))


proba_df = proba_df.assign(hp_thresh=np.where(proba_df.hp >= 0.65, 1, 0))
# #pred_df['log_reg_pred'] = y_pred
# #pred_df['log_reg_pred_manual'] = np.where(y_pred[1]>=0.75, 1, 0)
# pred_thresh = pd.Series(np.where(y_pred_proba[1]>=0.75, 1, 0))
# #pred_df['log_reg_pred_proba'] = y_pred_proba[:,1]
# print('Accuracy of Logistic Regression classifier on training set: {:.2f}'
#  .format(logit.score(X_train, y_train)))
# print('---')
# print(classification_report(y_train, y_pred))
# print('---')
# #print(classification_report(y_train, pred_thresh))
print(classification_report(y_train, proba_df.hp_thresh))
# -

# #### Decision Tree

tree = DecisionTreeClassifier(criterion='gini',
                              max_depth=5,
                              random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_train)
y_pred_proba = tree.predict_proba(X_train)
#agent_df['tree_pred'] = y_pred
#agent_df['tree_pred_proba'] = y_pred_proba[:,0]
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
 .format(tree.score(X_train, y_train)))
print('---')
print(classification_report(y_train, y_pred))
print('---')

# #### KNN

knn = KNeighborsClassifier(n_neighbors=3,
                           weights='uniform')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_train)
y_pred_proba = knn.predict_proba(X_train)
#agent_df['knn_pred'] = y_pred
#agent_df['knn_pred_proba'] = y_pred_proba[:,0]
print('Accuracy of KNN classifier on training set: {:.2f}'
 .format(knn.score(X_train, y_train)))
print('---')
print(classification_report(y_train, y_pred))
print('---')

# #### Random Forest

rf = RandomForestClassifier(bootstrap=True, 
                        class_weight="balanced", 
                        criterion='gini',
                        min_samples_leaf=3,
                        n_estimators=100,
                        max_depth=5, 
                        random_state=42)
rf.fit(X_train, y_train)
print(rf.feature_importances_)
y_pred = rf.predict(X_train)
y_pred_proba = rf.predict_proba(X_train)
#agent_df['rf_pred'] = y_pred
#agent_df['rf_pred_proba'] = y_pred_proba[:,0]
print('Accuracy of Random Forest classifier on training set: {:.2f}'
 .format(rf.score(X_train, y_train)))
print('---')
print(classification_report(y_train, y_pred))
print('---')

# ### K-Fold Cross Validation
# We do not have a lot of data, so K-Fold cross validation will allow us to reduce bias and noise by including every observation in the validation set once and in the training set K - 1 times.
#
# Use K = 5 and K = 10
# sklearn.KFold
#
# Maybe try Leave Once Out approach
#
# Use stratification because our dataset is unbalanced and we have few observations
#
# I want to use StratifiedKFold and train a model on each fold and average the coefficients of these models to create an aggregate model.

from sklearn.model_selection import cross_val_score

# #### Logistic Regression

logit = LogisticRegression(random_state=42,
                           solver='saga')
scores = cross_val_score(logit, X_train, y_train.high_performer.ravel(), cv=5, scoring="f1")
print(scores)
print("F1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# #### Decision Tree

tree = DecisionTreeClassifier(criterion='gini',
                              max_depth=3,
                              class_weight="balanced",
                              random_state=42)
scores = cross_val_score(tree, X_train, y_train.high_performer.ravel(), cv=5, scoring="f1")
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

import sklearn
sklearn.metrics.SCORERS.keys()
