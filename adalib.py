from typing import Dict, List, Tuple
from collections import namedtuple
import io
from copy import deepcopy
import math
import functools as ft

from sqlalchemy import create_engine

# Wrangling
import pandas as pd
import numpy as np

# Exploring
import scipy.stats as stats

# Visualizing
import matplotlib.pyplot as plt
import seaborn as sns

# import seaborn as sns

# Modeling
# import statsmodels.api as sm

# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    median_absolute_error,
)
from sklearn.feature_selection import f_regression

# from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import graphviz
from sklearn.tree import export_graphviz

# TODO: move the dataframe class from model.ipynb and create a copy function
# that does the copying easily so i can create a new model but have the same
# train and test data; have the class have X_train, X_test, y_train,
# y_test attributes that have get methods that will perform the operation
# on the train_df to extract the data

# Write a function that accepts a series (i.e. one column from a data frame)
# and summarizes how many outliers are in the series. This function should
# accept a second parameter that determines how outliers are detected, with
# the ability to detect outliers in 3 ways:

# Using the IQR
# Using standard deviations
# Based on whether the observation is in the top or bottom 1%.
# Use your function defined above to identify columns where you should handle
# the outliers.

# Write a function that accepts the zillow data frame and removes the
# outliers. You should make a decision and document how you will
# remove outliers.

# Is there erroneous data you have found that you need to remove or repair?
# If so, take action.

# Are there outliers you want to "squeeze in" to a max value? (e.g. all
# bathrooms > 6 => bathrooms = 6). If so, make those changes.


# TODO: Create functions that will accept what type of regression the user
# wants and do a linear, lasso, ridge, random forest, etc. (look at
# zillow clustering project code)
# TODO: copy the polynomial regression code from my zillow clustering project

# FUNCTIONAL PROGRAMMING


def compose(*fns):
    """
    right to left
    Written by Zach
    """
    return ft.partial(ft.reduce, lambda x, f: f(x), reversed(fns))


def pipe(v, *fns):
    """
    applies in the order supplied
    Written by Zach
    """
    return ft.reduce(lambda x, f: f(x), fns, v)


def map_exhaust(func, *iters):
    for args in zip(*iters):
        func(*args)


class DataSet(object):
    def __init__(self):
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.xcols = None
        self.ycol = None
        self.model = None
        self.pred_train = None
        self.pred_test = None
        self.score = None
        self.confmatrix = None
        self.classrep = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.support = None

    def get_train(self):
        return pd.concat([self.X_train, self.y_train], axis=1)

    train = property(get_train)

    def get_test(self):
        return pd.concat([self.X_test, self.y_test], axis=1)

    test = property(get_test)

    def data_only_copy(self):
        """
        returns a new object with only the data copied
        """
        copy = DataSet()
        copy.df = self.df.copy()

        return copy


# ACQUISITION FUNCTIONS


def get_db_url(
    hostname: str, username: str, password: str, db_name: str
) -> str:
    """
    return url for accessing a mysql database
    """
    return f"mysql+pymysql://{username}:{password}@{hostname}/{db_name}"


def get_sql_conn(hostname: str, username: str, password: str, db_name: str):
    """
    return a mysql connection object
    """
    return create_engine(get_db_url(hostname, username, password, db_name))


def df_metadata(df: pd.DataFrame) -> Tuple[int, tuple, str]:
    """
    return size, shape, and info of df
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    return df.size, df.shape, buffer.getvalue()


def df_print_metadata(df: pd.DataFrame) -> None:
    """
    print metadata of df
    """
    size, shape, info = df_metadata(df)
    print("DATAFRAME METADATA")
    print(f"Size: {size}")
    print()
    print(f"Shape: {shape[0]} x {shape[1]}")
    print()
    print("Info:")
    print(info, end="")


def df_peek(df: pd.DataFrame, nrows: int = 5) -> pd.DataFrame:
    """
    return DataFrame containing a random sample of nrows of df
    """
    return df.sample(n=nrows)


def df_head_tail(df: pd.DataFrame) -> pd.DataFrame:
    return df.head().append(df.tail())


def cohen_d(ser1: pd.Series, ser2: pd.Series) -> float:
    """
    return Cohen's d of two series
    """
    mean_diff = ser1.mean() - ser2.mean()
    n1, n2 = len(ser1) - 1, len(ser2) - 1
    pooled_var = (n1 * ser1.var() + n2 * ser2.var()) / (n1 + n2 - 2)
    return mean_diff / math.sqrt(pooled_var)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    # head_df = df.head()
    # print(f"HEAD\n{head_df}", end="\n\n")

    # tail_df = df.tail()
    # print(f"TAIL\n{tail_df}", end="\n\n")
    nrows = 10
    sample = df_peek(df, nrows)
    print(f"RANDOM SAMPLE OF {nrows}\n{sample}", end="\n\n")

    shape_tuple = df.shape
    print(f"SHAPE: {shape_tuple}", end="\n\n")

    describe_df = df.describe()
    print(f"DESCRIPTION\n{describe_df}", end="\n\n")

    print(f"INFORMATION")
    df.info()

    print()

    for col in df.columns:
        n = df[col].unique().shape[0]
        col_bins = min(n, 10)
        print(f"{col}:")
        if df[col].dtype in ["int64", "float64"] and n > 10:
            print(df[col].value_counts(bins=col_bins, sort=False))
        else:
            print(df[col].value_counts())
        print("\n")


def df_print_summary(df: pd.DataFrame, columns: List[str] = []) -> None:
    """
    columns is a sequence of columns whose IQR and range will be calculated.
    If columns is not specified, all columns will be summarized

    print describe(), iqr of columns, and range of columns
    """
    if not columns:
        columns = df.columns

    print("SUMMARY")
    description = df.describe()
    print("Description:")
    print(description)
    print()
    print("IQR:")
    # TODO: write function that will identify which columns are numeric
    # so we don't run iqr on non numeric columns and get an error
    # the function should print out which columns iqr was not called on
    # because they were not numeric
    for col in columns:
        print(f"\t{col}: {stats.iqr(df[col])}")
    print()
    print("Range:")
    for col in columns:
        print(f"\t{col}: {df[col].max() - df[col].min()}")


def series_is_whole_nums(series: pd.Series) -> bool:
    """
    return True if series is whole numbers
    """
    try:
        return (series % 1 == 0).all()
    except TypeError:
        return False


def df_float_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """
    return DataFrame with all columns that are whole numbers converted
    to int type
    """
    to_coerce = {}
    for col in df.columns:
        if series_is_whole_nums(df[col]):
            to_coerce[col] = int
    return df.astype(to_coerce)


def df_missing_vals(df: pd.DataFrame) -> pd.Series:
    """
    return Series containing the number of NaNs for each column that
    contains at least one
    """
    null_counts = df.isnull().sum()
    return null_counts[null_counts > 0]


def df_print_missing_vals(df: pd.DataFrame) -> None:
    """
    print the number of NaNs for each column that contains at least one
    """
    print("\nMissing Values:\n")
    series_missing_vals = df_missing_vals(df)
    if len(series_missing_vals):
        print(series_missing_vals)
    else:
        print("No missing values")


def df_percent_missing_vals(df: pd.DataFrame) -> pd.Series:
    """
    return Series containing the percent of NaNs of each column
    """
    return (df.isnull().sum() / df.shape[0]) * 100


def df_print_percent_missing_vals(df: pd.DataFrame) -> None:
    """
    print the percent of NaNs of each column
    """
    print(df_percent_missing_vals)


def df_missing_vals_by_col(df: pd.DataFrame) -> pd.DataFrame:
    null_count = df.isnull().sum()
    null_percentage = (null_count / df.shape[0]) * 100
    empty_count = pd.Series(
        ((df == " ") | (df == "") | (df == "nan") | (df == "NaN")).sum()
    )
    return pd.DataFrame(
        {
            "nmissing": null_count,
            "percentage": null_percentage,
            "nempty": empty_count,
        }
    )


def df_missing_vals_by_row(df: pd.DataFrame) -> pd.DataFrame:
    null_count = df.isnull().sum(axis=1)
    null_percentage = (null_count / df.shape[1]) * 100
    empty_count = pd.Series(((df == " ") | (df == "")).sum(axis=1))
    return pd.DataFrame(
        {
            "nmissing": null_count,
            "percentage": null_percentage,
            "nempty": empty_count,
        }
    )


def get_upper_outliers(s, k):
    """
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    """
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))


def add_upper_outlier_columns(df, k):
    """
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    """
    # outlier_cols = {col + '_outliers': get_upper_outliers(df[col], k)
    #                 for col in df.select_dtypes('number')}
    # return df.assign(**outlier_cols)
    out = df.copy()
    for col in df.select_dtypes("number"):
        out[col + "_outliers"] = get_upper_outliers(df[col], k)

    return out


def df_remove_outliers_normal(
    df: pd.DataFrame, cols: List[str], zscore_limit: int
) -> pd.DataFrame:
    """
    WARNING: use for normal distributions
    return DataFrame without rows that contain a cell under any of cols with a
    z-score that exceeds zscore_limit
    """
    df_clean = pd.DataFrame()
    for col in cols:
        df_clean[col] = df[col][
            (np.abs(stats.zscore(df[col])) < zscore_limit).all(axis=1)
        ]

    return df_clean


def df_remove_outliers(
    df: pd.DataFrame, cols: List[str], bottom_top_percentile: float
) -> pd.DataFrame:
    """
    return DataFrame without rows that contain a cell under any of cols with
    a value that is in the bottom or top percentile as specified by
    bottom_top_percentile

    e.g., if bottom_top_percentile is 3, the function will remove
    rows with cells whose value is in the bottom or top 3 percentile
    """
    df_clean = df.copy()
    for col in cols:
        btmval = stats.scoreatpercentile(df[col], bottom_top_percentile)
        topval = stats.scoreatpercentile(df[col], 100 - bottom_top_percentile)
        df_clean[col] = df[(df[col] >= btmval) & (df[col] <= topval)]
    return df_clean


def series_outliers(series: pd.Series, k: float = 1.5) -> pd.Series:
    """
    returns Series containing outliers
    """
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    return series[(series < lower_bound) | (series > upper_bound)]


def remove_outliers(
    df: pd.DataFrame, cols: list, k: float = 1.5
) -> pd.DataFrame:
    """
    return DataFrame without outliers in cols using IQR method
    with k as the multiplier
    """
    df_outless = df.copy()
    for col in cols:
        outliers = series_outliers(df_outless[col], k)
        df_outless = df_outless[~df_outless[col].isin(outliers)]

    return df_outless


def df_impute_0(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    return DataFrame with only the NaNs in cols replaced with 0
    """
    impute_vals = {col: 0 for col in cols}
    return df.fillna(value=impute_vals)


def df_impute_col(df: pd.DataFrame, col: str, **kwargs) -> pd.DataFrame:
    imp = SimpleImputer(**kwargs)
    imp.fit(df[[col]].dropna())
    df_imputed = df.copy()
    df_imputed.loc[:, col] = imp.transform(df_imputed[[col]])

    return df_imputed


def numeric_to_categorical(df: pd.DataFrame, cols: tuple) -> pd.DataFrame:
    to_coerce = {col: "category" for col in cols}
    return df.astype(to_coerce)


def handle_missing_values(
    df, prop_required_column=0.5, prop_required_row=0.75
):
    threshold = int(round(prop_required_column * len(df.index), 0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df


def series_bin_with_labels(
    series: pd.Series, bins: pd.IntervalIndex, labels: tuple
) -> pd.Series:
    binned = pd.cut(series, bins=bins)
    intervals_to_labels = {b: lab for b, lab in zip(bins, labels)}
    return binned.apply(lambda x: intervals_to_labels[x])


def split_ratio(train, test):
    """
    return train-split ratio
    """
    total_rows = train.shape[0] + test.shape[0]
    return train.shape[0] / total_rows


def df_r_and_p_values(
    X: pd.DataFrame, y: pd.Series
) -> Dict[str, Tuple[float, float]]:
    """
    return a dict containing Pearson's R and p-value for each column
    in X
    """
    return {col: stats.pearsonr(X[col], y) for col in X.columns}


def df_print_r_and_p_values(X: pd.DataFrame, y: pd.Series) -> None:
    """
    print the Pearson's R and p-value for each column in X
    """
    print("Pearson's R")
    for k, v in df_r_and_p_values(X, y).items():
        col = k
        r, p = v
        print(f"{col}:")
        print(
            f"\tPearson's R is {r:.3f} with a significance p-value "
            f"of {p:.3f}\n"
        )


# PLOTTING
def df_plot_numeric(df: pd.DataFrame, cols: list, hue=None, **kwargs) -> None:
    """
    For plotting purely continuous v. purely continuous
    """
    sns.pairplot(data=df, vars=cols, hue=hue, **kwargs)
    plt.show()
    plt.figure(figsize=(15, 15))
    sns.heatmap(df[cols].corr(), cmap="Blues", annot=True)
    plt.show()


def df_jitter_plot(df: pd.DataFrame, X: list, Y: list, hue=None) -> None:
    """
    For plotting semi-continuous semi-categorical variables v. continuous
    """
    sns.pairplot(data=df, x_vars=X, y_vars=Y, hue=hue)
    plt.show()


def relplot_num_and_cat(
    df: pd.DataFrame, x: str, y: str, hue: str, **kwargs
) -> pd.DataFrame:
    """
    Write a function that will use seaborn's relplot to plot 2 numeric
    (ordered) variables and 1 categorical variable. It will take, as input,
    a dataframe, column name indicated for each of the following: x, y, & hue.
    """
    sns.relplot(x=x, y=y, hue=hue, data=df, **kwargs)
    plt.show()


def swarmplot_num_and_cat(
    df: pd.DataFrame, X: str, Y: list, hue: str = None, **kwargs
) -> None:
    """
    Write a function that will take, as input, a dataframe, a categorical
    column name, and a list of numeric column names. It will return a series
    of subplots: a swarmplot for each numeric column. X will be the
    categorical variable.
    """
    cols = 2
    rows = math.ceil(len(Y) / cols) if len(Y) // cols > 0 else 1

    plt.figure(figsize=(15, 20))
    for i, y in enumerate(Y):
        plt.subplot(rows, cols, i + 1)
        sns.swarmplot(x=X, y=y, data=df, hue=hue, **kwargs)
    plt.plot()


def crosstab_cat(df: pd.DataFrame, xcols: list, ycol: str) -> None:
    for col in xcols:
        plt.figure(figsize=(10, 8))
        ct = pd.crosstab(
            df[col], df[ycol], margins=True
        )  # .apply(lambda r: r/r.sum(), axis=1)
        sns.heatmap(ct, cmap="Blues", annot=True, cbar=False, fmt=".2f")
        # print(pd.crosstab(df[outer], df[inner], margins=True)
        # .apply(lambda r: r/r.sum(), axis=1))
        plt.show()


# STATISTICAL TESTS
def series_ttest(s1: pd.Series, s2: pd.Series) -> None:
    t_stat, p_val = stats.ttest_ind(s1, s2)
    print(f"T-stat: {t_stat}\np-val: {p_val}")


def series_chi2_test(s1: pd.Series, s2: pd.Series) -> None:
    ct = pd.crosstab(s1, s2)
    stat, p, dof, expected = stats.chi2_contingency(ct)
    print(f"Chi2: {stat}\np-val: {p}")


# LINEAR REGRESSION


LinRegModel = namedtuple(
    "LinRegModel", ["model", "pred_train", "pred_test", "equation"]
)
# model is a LinearRegression object
# pred_train is a numpy array
# pred_test is a numpy array
# equation is a tuple of str, float, float, str


def linreg_print_model(
    y_label: str, y_intercept: float, coefs: List[float], X_labels: List[str]
) -> None:
    """
    print equation of a linear regression model
    """
    # print(coefs)
    # print(X_labels)
    print(f"{y_label} = {y_intercept:.2f}", end="")
    for coef, label in zip(coefs, X_labels):
        print(f" +\n\t{coef:.3f} * {label}", end="")
    print()


def linreg_fit_and_predict(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    **kwargs,
) -> namedtuple:
    """
    perform linear regression on train data and predict on the training
    and test data

    return predictions of the train and test data
    """
    lm = LinearRegression(**kwargs)
    lm.fit(X_train, y_train)

    pred_train = lm.predict(X_train)
    pred_test = lm.predict(X_test)

    y_label = y_train.columns[0]
    if isinstance(lm.intercept_, float):
        y_intercept = lm.intercept_
    else:
        y_intercept = lm.intercept_[0]
    coefs = lm.coef_[0]
    X_labels = X_train.columns

    return LinRegModel(
        lm, pred_train, pred_test, (y_label, y_intercept, coefs, X_labels)
    )


def linreg_evaluate_model(
    X: pd.DataFrame, y: pd.Series, y_pred: np.ndarray
) -> None:
    """
    print evaluation metrics for linear regression model
    """
    y_label = y.columns[0]
    X_labels = X.columns

    print("Univariate Linear Regression Model Evaluation")
    meanse = mean_squared_error(y, y_pred)
    print(f"\tMean SE: {meanse:.3f}")

    meanae = mean_absolute_error(y, y_pred)
    print(f"\tMean AE: {meanae:.3f}")

    print()

    medianae = median_absolute_error(y, y_pred)
    print(f"\tMedian AE: {medianae:.3f}")

    print()

    r2 = r2_score(y, y_pred)
    print(
        f"\t{r2:.2%} of the variance in {y_label} can be explained "
        f"by {X_labels.tolist()}."
    )
    print()

    print("P-VALUE")
    f_vals, p_vals = f_regression(X, y)
    print(f"\tTrain: {p_vals[0]:.3}")


def linreg_plot_residuals(y_test: pd.Series, pred_test: np.ndarray):
    """
    plot residuals for a linear regression model
    """
    y_label = y_test.columns[0]
    plt.scatter(pred_test, pred_test - y_test, c="g", s=20)
    plt.hlines(y=0, xmin=pred_test.min(), xmax=pred_test.max())
    plt.title("Residual plot")
    plt.ylabel("Residuals")
    plt.xlabel(y_label)
    plt.show()


def linreg_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    **kwargs,
):
    """
    models, evaluates (train and test), and plots residuals (test only)
    for linear regression model
    """
    lrm = linreg_fit_and_predict(X_train, X_test, y_train, y_test, **kwargs)

    # TODO: print equation using linreg_print_model()
    print("EQUATION")
    linreg_print_model(
        lrm.equation[0], lrm.equation[1], lrm.equation[2], lrm.equation[3]
    )

    print("-" * 50)

    print("TRAIN")
    linreg_evaluate_model(X_train, y_train, lrm.pred_train)

    print("-" * 50)

    print("TEST")
    linreg_evaluate_model(X_test, y_test, lrm.pred_test)

    linreg_plot_residuals(y_test, lrm.pred_test)


# def multi_linreg_evaluate_model(
#     X: pd.DataFrame, y: pd.Series, y_pred: np.ndarray
# ) -> None:
#     """
#     print evaluation metrics for linear regression model
#     """
#     y_label = y.columns[0]
#     X_labels = X.columns[0]

#     print("Multivariate Linear Regression Model Evaluation")
#     meanse = mean_squared_error(y, y_pred)
#     print(f"\tMean SE: {meanse:.3f}")

#     meanae = mean_absolute_error(y, y_pred)
#     print(f"\tMean AE: {meanae:.3f}")

#     print()

#     medianae = median_absolute_error(y, y_pred)
#     print(f"\tMedian AE: {medianae:.3f}")

#     print()

#     r2 = r2_score(y, y_pred)
#     print(
#         f"\t{r2:.2%} of the variance in {y_label} can be explained "
#         f"by {X_labels.tolist()}."
#     )
#     print()

#     print("P-VALUE")
#     f_vals, p_vals = f_regression(X, y)
#     print(f"\tTrain: {p_vals[0]:.3}")
#     print()


# def multi_linreg_fit_and_predict(
#     X_train: pd.DataFrame,
#     X_test: pd.DataFrame,
#     y_train: pd.Series,
#     y_test: pd.Series,
#     **kwargs,
# ) -> namedtuple:
#     """
#     perform multivariate linear regression on train data and predict on the
#     training and test data

#     return predictions of the train and test data
#     """
#     lm = LinearRegression(**kwargs)
#     lm.fit(X_train, y_train)

#     pred_train = lm.predict(X_train)
#     pred_test = lm.predict(X_test)

#     y_label = y_train.columns[0]
#     y_intercept = lm.intercept_[0]
#     coefs = lm.coef_[0]
#     X_labels = X_train.columns

#     return LinRegModel(
#         lm, pred_train, pred_test, (y_label, y_intercept, coefs, X_labels)
#     )


# def multi_linreg_model(
#     X_train: pd.DataFrame,
#     X_test: pd.DataFrame,
#     y_train: pd.Series,
#     y_test: pd.Series,
#     **kwargs,
# ):
#     """
#     models, evaluates (train and test), and plots residuals (test only)
#     for a multi linear regression model
#     """
#     lrm = multi_linreg_fit_and_predict(X_train, X_test, y_train, y_test,
# **kwargs)

#     linreg_print_model(lrm.equation[0], lrm.equation[1], lrm.equation[])

#     print("TRAIN")
#     multi_linreg_evaluate_model(X_train, y_train, lrm.pred_train)

#     print("-" * 50)

#     print("TEST")
#     multi_linreg_evaluate_model(X_test, y_test, lrm.pred_test)

#     linreg_plot_residuals(y_test, lrm.pred_test)


def normalize_cols(df_train, df_test, cols):
    df_train_norm = pd.DataFrame()
    for col in cols:
        minimum = df_train[col].min()
        maximum = df_train[col].max()
        df_train_norm[f"{col}_norm"] = (df_train[col] - minimum) / (
            maximum - minimum
        )

    df_test_norm = pd.DataFrame()
    for col in cols:
        minimum = df_train[col].min()  # use the min and max from the train set
        maximum = df_train[col].max()
        df_test_norm[f"{col}_norm"] = (df_test[col] - minimum) / (
            maximum - minimum
        )
    return df_train_norm, df_test_norm


# TODO: Not sure whether to have this accept DataFrames or Series
# What does sklearn want?
# def series_minmax_scale(train: pd.DataFrame, test: pd.DataFrame,
# cols: list) -> tuple:
#     scaler = MinMaxScaler()
#     scaler.fit(train[cols])

#     out_train = train.copy()
#     out_test = test.copy()

#     out_train[cols] = scaler.transform(out_train[cols])
#     out_test[cols] = scaler.transform(out_test[cols])

#     return out_train, out_test


def series_encode(series: pd.Series) -> tuple:
    encoder = LabelEncoder()
    encoder.fit(series)
    return encoder.transform(series), encoder


def logreg_fit_and_predict(X_train, y_train, X_test, y_test, **kwargs):
    logreg = LogisticRegression(**kwargs)
    logreg.fit(X_train, y_train)

    preds_train = logreg.predict(X_train)
    preds_test = logreg.predict(X_test)

    return preds_train, preds_test, logreg


def logreg_evaluate_model(y_true, y_preds):
    confmatrix = confusion_matrix(y_true, y_preds)
    df_confmatrix = pd.DataFrame(
        {"Pred -": confmatrix[:, 0], "Pred +": confmatrix[:, 1]}
    )
    df_confmatrix.rename({0: "Actual -", 1: "Actual +"}, inplace=True)
    print("Confusion matrix:")
    print(df_confmatrix)
    print()

    print("Classification report:")
    classrep = classification_report(y_true, y_preds)
    print(classrep)
    print()

    tp = confmatrix[1][1]
    fp = confmatrix[0][1]
    tn = confmatrix[0][0]
    fn = confmatrix[1][0]

    print(f"Accuracy: {accuracy_score(y_true, y_preds):.3f}")
    print()

    print("Rates:")
    print(f"True positive rate: {tp / (tp + fn):.3f}")
    print(f"False positive rate: {fp / (fp + tp):.3f}")
    print()
    print(f"True negative rate: {tn / (tn + fp):.3f}")
    print(f"False negative rate: {fn / (fn + tn):.3f}")
    print()


def logreg_model(X_train, y_train, X_test, y_test, **kwargs):
    preds_train, preds_test, _ = logreg_fit_and_predict(
        X_train, y_train, X_test, y_test, **kwargs
    )

    print("TRAIN EVALUATION")
    logreg_evaluate_model(y_train, preds_train)

    print("TEST EVALUATION")
    logreg_evaluate_model(y_test, preds_test)


def dectree_fit_and_predict(X_train, y_train, X_test, y_test, **kwargs):
    dectree = DecisionTreeClassifier(**kwargs)
    dectree.fit(X_train, y_train)

    preds_train = dectree.predict(X_train)
    preds_test = dectree.predict(X_test)

    return preds_train, preds_test, dectree.classes_, dectree


def dectree_evaluate_model(y_true, y_preds, classes):

    print("Accuracy:", accuracy_score(y_true, y_preds))
    print()

    confmatrix = confusion_matrix(y_true, y_preds)
    frame = {
        f"Pred {clas}": confmatrix[:, i] for i, clas in enumerate(classes)
    }
    df_confmatrix = pd.DataFrame(frame)
    for i, clas in enumerate(classes):
        df_confmatrix.rename({i: f"Actual {clas}"}, inplace=True)
    print("Confusion matrix:")
    print(df_confmatrix)
    print()

    print("Classification report:")
    classrep = classification_report(y_true, y_preds)
    print(classrep)
    print()

    for i, clas in enumerate(classes):
        total_row = confmatrix[i].sum()
        tru = confmatrix[i][i]

        total_col = confmatrix[:, i].sum()
        fls = total_col - tru

        print(f"True {clas} rate: {tru / total_row: .3f}")
        print(f"False {clas} rate: {fls / total_col: .3f}")
        print()


def dectree_model(X_train, y_train, X_test, y_test, **kwargs):
    preds_train, preds_test, classes, _ = dectree_fit_and_predict(
        X_train, y_train, X_test, y_test, **kwargs
    )

    print("TRAIN EVALUATION")
    dectree_evaluate_model(y_train, preds_train, classes)

    print("TEST EVALUATION")
    dectree_evaluate_model(y_test, preds_test, classes)


def dectree_visualize(
    model, features: List[str], classes: List[str], fname: str
) -> None:
    dot = export_graphviz(
        model,
        out_file=None,
        feature_names=features,
        class_names=classes,  # target value names
        special_characters=True,
        filled=True,  # fill nodes w/ informative colors
        impurity=False,  # show impurity at each node
        leaves_parallel=True,  # all leaves at the bottom
        proportion=True,  # show percentages instead of numbers at each leaf
        rotate=True,  # left to right instead of top-bottom
        rounded=True,  # rounded boxes and sans-serif font
    )

    graphviz.Source(dot, filename=fname, format="png")


def random_forest_fit_and_predict(X_train, y_train, X_test, y_test, **kwargs):
    rf = RandomForestClassifier(**kwargs)
    rf.fit(X_train, y_train)

    preds_train = rf.predict(X_train)
    preds_test = rf.predict(X_test)

    return preds_train, preds_test, rf


def random_forest_model(X_train, y_train, X_test, y_test, **kwargs):
    pred_train, pred_test, model = random_forest_fit_and_predict(
        X_train, y_train, X_test, y_test, **kwargs
    )

    print("TRAIN EVALUATION")
    knn_evaluate_model(y_train, pred_train, X_train.columns)

    print("TEST EVALUATION")
    knn_evaluate_model(y_test, pred_test, X_train.columns)


def knn_fit_and_predict(X_train, y_train, X_test, y_test, **kwargs):
    knn = KNeighborsClassifier(**kwargs)
    knn.fit(X_train, y_train)

    preds_train = knn.predict(X_train)
    preds_test = knn.predict(X_test)

    return preds_train, preds_test, knn.classes_, knn


# TODO: I may be able to combine this with dectree_evaluate_model
def knn_evaluate_model(y_true, y_preds, classes):
    print("Accuracy:", accuracy_score(y_true, y_preds))
    print()

    confmatrix = confusion_matrix(y_true, y_preds)
    frame = {
        f"Pred {clas}": confmatrix[:, i] for i, clas in enumerate(classes)
    }
    df_confmatrix = pd.DataFrame(frame)
    for i, clas in enumerate(classes):
        df_confmatrix.rename({i: f"Actual {clas}"}, inplace=True)
    print("Confusion matrix:")
    print(df_confmatrix)
    print()

    print("Classification report:")
    classrep = classification_report(y_true, y_preds)
    print(classrep)
    print()

    for i, clas in enumerate(classes):
        total_row = confmatrix[i].sum()
        tru = confmatrix[i][i]

        total_col = confmatrix[:, i].sum()
        fls = total_col - tru

        print(f"True {clas} rate: {tru / total_row: .3f}")
        print(f"False {clas} rate: {fls / total_col: .3f}")
        print()


def knn_model(X_train, y_train, X_test, y_test, **kwargs):
    pred_train, pred_test, classes, model = knn_fit_and_predict(
        X_train, y_train, X_test, y_test, **kwargs
    )

    print("TRAIN EVALUATION")
    knn_evaluate_model(y_train, pred_train, classes)

    print("TEST EVALUATION")
    knn_evaluate_model(y_test, pred_test, classes)

    return pred_train, pred_test, model


def kmeans_elbow(X: pd.DataFrame, nclusters_width, **kwargs):
    intertias = []
    nclusters_range = range(1, nclusters_width + 1)
    for n in nclusters_range:
        kmeans = KMeans(n_clusters=n, **kwargs)
        kmeans.fit(X)
        intertias.append(kmeans.inertia_)

    kmeans_perf = pd.DataFrame(
        list(zip(nclusters_range, intertias)), columns=["n_clusters", "ssd"]
    )

    plt.scatter(kmeans_perf.n_clusters, kmeans_perf.ssd)
    plt.plot(kmeans_perf.n_clusters, kmeans_perf.ssd)

    plt.xticks(nclusters_range)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of Squared Distances")
    plt.title("The elbow method")
    plt.show()


def kmeans_fit_and_predict(X: pd.DataFrame, **kwargs) -> np.ndarray:
    kmeans = KMeans(**kwargs)
    kmeans.fit(X)
    return kmeans.predict(X), kmeans.labels_, kmeans.inertia_


def dbscan_data_trans(df):
    """
    Transforms data for use with DBSCAN

    Returns a standard scaled numpy array of df
    """
    array = df.values.astype("float64", copy=False)
    stscaler = StandardScaler().fit(array)
    return stscaler.transform(array)


def dbscan_model(array, **params):
    """
    Performs a DBSCAN clustering in array using params

    Returns DBSCAN objects, array of boolean values for indexing
    core samples, identified by True value
    """
    dbsc = DBSCAN(**params).fit(array)
    labels = dbsc.labels_
    core_samples = np.zeros_like(labels, dtype=bool)
    core_samples[dbsc.core_sample_indices_] = True
    return dbsc, core_samples


def dbscan_visualize(x, y, dbsc):
    """
    Creates and shows a scatter plot of the features of a DBSCAN model
    """
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x, y, hue=dbsc.labels_)
    plt.title(f"eps={dbsc.eps}, min_samples={dbsc.min_samples}")
    plt.show()
