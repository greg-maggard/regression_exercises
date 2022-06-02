import pandas as pd
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import seaborn as sns

def plot_variable_pairs(df):
    features = df.select_dtypes(include = 'number').columns.tolist()
    pairs = it.combinations(features, 2)
    for pair in pairs:
        print(pair)
        sns.lmplot(x = pair[0], y = pair[1], line_kws={'color':'red'}, data = df.sample(100000))
        plt.show();

def plot_categorical_and_continuous_vars(df):
    cat_columns = ['county']
    cont_columns = df.select_dtypes(include = 'number').columns.tolist()
    pairs = it.product(cat_columns, cont_columns)
    for pair in pairs:
        print(pair)
        sns.set(rc={'figure.figsize':(15,6)})
        fig, axes = plt.subplots(1,3)
        
        sns.stripplot(x= pair[0], y=pair[1], data=df, ax = axes[0])
        sns.boxplot(x= pair[0], y=pair[1], data=df, ax = axes[1])
        sns.barplot(x= pair[0], y=pair[1], data=df, ax = axes[2])
        plt.show();