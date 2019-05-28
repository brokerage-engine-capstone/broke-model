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

#

# ## Prepare Environment

# +
# %matplotlib inline

# disable warnings
import warnings
warnings.filterwarnings('ignore')

# data wrangling
import pandas as pd
import numpy as np
import itertools
import math
from random import randint
from datetime import datetime
from scipy import stats

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


from sklearn.preprocessing import LabelEncoder
# -

df = pd.read_csv('agents_with_transactions.csv')

df.head()

# ## Data Preparation

# **Peek at data to look for nulls and counts**

df.isnull().sum()

df.shape

# **Lowercasing all column names**

[df.rename(columns=lambda x: x.lower(), inplace=True) for col in df]

list(df.columns)

# ## Column Cleanup

# ### Agent_Id
# **Looks fine so leaving it as-is**

df.agent_id.value_counts()

# ### Agent_Name
# **Dropping agent_name as it's not helpful.**

df.drop(columns='agent_name', inplace=True)

# ### Commission_Anniversary
# **Looks fine so leaving it as-is**

df.commission_anniversary.value_counts(dropna=False)

# ### Brokerage_Id
# **Looks fine so leaving it as-is**

df.brokerage_id.value_counts(dropna=False)

# ### Brokerage_Name
# **Dropping agent_name as it's not helpful.**

df.drop(columns='brokerage_name', inplace=True)

# ### Commission_Schedule_Id
# **Looks fine so leaving it as-is...it maybe helpful later on**

df.commission_schedule_id.value_counts(dropna=False)

# ### Commission_Schedule_Effective_Start_At
# **Looks fine so leaving it as-is...it maybe helpful later on**

df.commission_schedule_effective_start_at.value_counts(dropna=False)

# ### Commission_Schedule_Effective_End_At
# **Looks fine so leaving it as-is...it maybe helpful later on**

df.commission_schedule_effective_end_at.value_counts(dropna=False)

# ### Commission_Schedule_Active
# **Dropping commission_schedule_active as it's not helpful.**

df.commission_schedule_active.value_counts(dropna=False)

df.drop(columns='commission_schedule_active', inplace=True)


