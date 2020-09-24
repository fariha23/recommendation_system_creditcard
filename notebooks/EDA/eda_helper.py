import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import random
import matplotlib.ticker as mtick
from datetime import datetime
from pandas import ExcelWriter

#functions
def fill_NaN_between_two_columns(df, col1, col2):
    df[col1]= df[col1].fillna(df[col1].groupby(df[col2]).transform('first'))
    df[col2] = df[col2].fillna(df[col2].groupby(df[col1]).transform('first'))

def rows_to_del(df, column, kw_list):
    for kw in kw_list:
        rowstodel=df[df[column]==kw].index
        df.drop(rowstodel, inplace=True)

def createExcel(filename, dataframe):
    writer = ExcelWriter(filename)
    dataframe.to_excel(writer, encoding='utf8', index=False)
    
    writer.save()
	
def rows_to_del_with_index(df,rowstodel):
    df.drop(rowstodel, inplace=True)


def catPerMonthGraph2(month,df,ax):
    datetime_object = datetime.strptime(str(month), "%m")
    title=datetime_object.strftime("%B")
    df_mean=df[df.index.month==month].groupby(['category']).mean('amountnum')['amountnum'].to_frame()
    #df_mean.plot.bar(title=title, legend=False, figsize=(12,5))
    ax.bar(df_mean.index,df_mean['amountnum'])
    ax.set_title(title)
    ax.set_ylabel('Mean Amount($)')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)