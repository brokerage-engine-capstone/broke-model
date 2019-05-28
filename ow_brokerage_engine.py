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
# <div class="toc"><ul class="toc-item"><li><span><a href="#Data-Preparation" data-toc-modified-id="Data-Preparation-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Data Preparation</a></span></li><li><span><a href="#Column-Cleanup" data-toc-modified-id="Column-Cleanup-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Column Cleanup</a></span><ul class="toc-item"><li><span><a href="#Earned_volume" data-toc-modified-id="Earned_volume-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Earned_volume</a></span></li><li><span><a href="#Bonus-Columns" data-toc-modified-id="Bonus-Columns-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Bonus Columns</a></span></li><li><span><a href="#Transaction_side" data-toc-modified-id="Transaction_side-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Transaction_side</a></span></li><li><span><a href="#Standard_commission_type" data-toc-modified-id="Standard_commission_type-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Standard_commission_type</a></span></li><li><span><a href="#Standard_commission_gci_amount" data-toc-modified-id="Standard_commission_gci_amount-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Standard_commission_gci_amount</a></span></li><li><span><a href="#Standard_commission_brokerage_net_amount" data-toc-modified-id="Standard_commission_brokerage_net_amount-2.6"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>Standard_commission_brokerage_net_amount</a></span></li></ul></li></ul></div>

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
# -

df = pd.read_csv('agents_with_transactions.csv')

df.head()

# ## Data Preparation

# **Peek at data to look for nulls and counts**

df.isnull().sum()

df.shape

# **Lowercasing all column names**

[df.rename(columns=lambda x: x.lower(), inplace=True) for col in df]

# ## Column Cleanup

# ### Earned_volume
# **Looks fine so leaving it as-is**

df.earned_volume.value_counts(dropna=False).sort_values(ascending=False)

# ### Bonus Columns
# **Dropping all three as they're filled with 95%+ nulls.**

to_drop = [col for col in df if col.startswith('bonus_')]
df.drop(columns=to_drop, inplace=True)

# ### Transaction_side
# **Transaction_side looks clean and fairly well balanced - keeping it as is.**

df.transaction_side.value_counts()

# ### Standard_commission_type

df.standard_commission_type.value_counts(dropna=False)

# **Dropping standard_commission_type as it's not helpful**

df.drop(columns='standard_commission_type', inplace=True)

# ### Standard_commission_gci_amount

df.standard_commission_gci_amount.value_counts(dropna=False)

# **Imputing 0 into all nulls**

df.standard_commission_gci_amount.fillna(value=0, inplace=True)

# ### Standard_commission_brokerage_net_amount

df.standard_commission_brokerage_net_amount.value_counts(ascending=False)

df.standard_commission_brokerage_net_amount.isnull().sum()

# **Imputing 0 into all nulls**

df.standard_commission_brokerage_net_amount.fillna(value=0, inplace=True)


