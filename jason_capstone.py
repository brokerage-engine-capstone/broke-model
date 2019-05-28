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

pd.set_option('display.max_columns', 500)
# -

df = pd.read_csv('agents_with_transactions.csv')

df.head()


