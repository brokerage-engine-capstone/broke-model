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
# <div class="toc"><ul class="toc-item"><li><span><a href="#Environment" data-toc-modified-id="Environment-1">Environment</a></span></li><li><span><a href="#Acquisition" data-toc-modified-id="Acquisition-2">Acquisition</a></span></li><li><span><a href="#Preparation" data-toc-modified-id="Preparation-3">Preparation</a></span></li><li><span><a href="#Exploration" data-toc-modified-id="Exploration-4">Exploration</a></span></li><li><span><a href="#Modeling" data-toc-modified-id="Modeling-5">Modeling</a></span></li></ul></div>
# -

# ## Environment 

# +
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
# -

# ## Acquisition

df = pd.read_csv('agents_with_transactions.csv')

# ## Preparation

# **Lowercasing all column names**

[df.rename(columns=lambda x: x.lower(), inplace=True) for col in df]

df.columns.

le = LabelEncoder()
df = df.assign(TRANSACTION_ID=le.fit(df['TRANSACTION_ID']).transform(df['TRANSACTION_ID']))

# ## Exploration

# ## Modeling


