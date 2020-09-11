#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis

# In[1]:


from __future__ import print_function
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
from eda_helper import fill_NaN_between_two_columns
from eda_helper import rows_to_del
from eda_helper import createExcel
from eda_helper import rows_to_del_with_index
from eda_helper import catPerMonthGraph2

PROJ_ROOT = os.path.join(os.pardir)
PATH=os.path.abspath(PROJ_ROOT)


# In[2]:


#creating df from json file and changing df['date'] to datetime type

df=pd.read_json('anon_transactions.json', orient='records')
df['date']=pd.to_datetime(df['date'], format='%Y-%m-%d')
df.info()


# In[3]:


#There are 29 columns and 90086 rows/entries
df.head()


# # Data Wrangling

# ### Deletion of Columns

# In[4]:


df.drop(['detail_category','high_level_category','transactionid','client_id','currency_code','country','pending','original_description','category0','finsight_api_uid','finsight_image','insight_ctaurl','insight_text','original_uid','tenant_id','accountId'], axis=1,inplace=True)

#Reasons
#1 deleting [category0','finsight_api_uid','finsight_image','insight_ctaurl','insight_text','original_uid','tenant_id','accountId']
#2 deleting ['client_id',currency_code','country','pending','original_description']as these cols don't add much to the data. country==USA, currency==USD, Pending=1,0 and original_description=simple_description
#3 Looks like if I delete transactionid col I can delete multiple duplicates. its the only col that makes every row differnet even though they are not. 


# In[5]:


df.info() #13 cols left


# ### Dealing with NaN values

# #### From df.info() output I can see that there are several columns that could be used to fill NAN values of their "like" columns. For example: zip_code and city columns can be used to fill each others NaN. 
# Process: Find a row(#1) with NaN in zip_code and city with some city1 in it. Find another row(#2) where city is set to city1 with corresponding zip_code(#1). Replace NaN of row(#1) with zip_code(#1) as they both have same city

# In[6]:


df.info()# Number of non-null values before exchanging 
exchange_list=[['city','zip_code'],['category','categoryid']]
for pair in exchange_list:
    fill_NaN_between_two_columns(df, pair[0], pair[1])
print( '##############')
df.info() # Number of non-null values after exchanging


# ###### From the latter df.info() above, there is significant gain of values from the exchanges

# ### Dealing with NaN in 'accountid' using 'uid'

# In[7]:


#df['accountid'].max() #10449313.0
#(df[df.accountid.isnull()]) #28700
#There are 28700 NaN in accountid and the max accountid for now goes to 10449313.0. 
#The range from which NaN in accountid will be replaced is 10449313.0 and 10449313+28701=10478014


# In[8]:


temp=df.groupby('uid')['accountid'].first().to_dict()

for k, v in temp.items():
    if np.isnan(v):
        temp[k]=random.randint(10449313, 10478014)
df['accountid']=df['uid'].map(temp)

#check
assert df[df.accountid.isnull()].empty


# In[9]:


df.info()


# ### Category Column

# Category column needs to understood so we can keep the relevant data and delete non significant data

# In[10]:


#8/31 deleting 'Refunds/Adjustments' per Ana's email
#9/2 Adding 'Paychecks/Salary', 'Tax', 'Other Bills' and 'Other Expenses' in kw_list
#9/3 Other expenses and other bills have information not specific to anything. I think it will not help with recomm system.
#9/3: deleting rewards and printing too

kw_list=['Other Bills','Other Expenses','Paychecks/Salary','Tax','Refunds/Adjustments','Transfer','Transfers', 'Interest','Credit Card Payments','Uncategorized','Business Miscellaneous','Paychecks/Salary', 'Insurance', 'Savings',            'Other Income','Securities Trades','Deposits', 'Services','Tax','Taxes','Rent',            'Mortgages', 'Loans','Investment Income', 'ATM/Cash Withdrawals','Investment Income', 'Loans',           'Retirement Contributions','Expense Reimbursement', 'Retirement Income', 'Charitable Giving', 'Bank Fees',          'Wages Paid', 'Checks' ,'Rewards','Printing','Payment']
    
rows_to_del(df, 'category', kw_list)

df['category'].nunique() #Unique values after deleting rows with specific categories. 


# In[11]:


df['category'].unique()


# In[12]:


df.reset_index(inplace=True, drop=True)


# In[13]:


df.info() #66609 non null entries


# In[14]:


#Delete nan values in category col. 

rowstodel=df[df['category'].isnull()].index
df.drop(rowstodel, inplace=True)

df.reset_index(inplace=True, drop=True)


# In[15]:


df.info() #66446 rows left.


# In[16]:


len(df[(df.state.isnull()) & (df.city.isnull())]) #there are 31K rows with no city and state info. 
#Question for Ana: This could be an issue for next part of the goal


# #### There are still 16K rows where categoryid is still NaN
# 
# Which categories don't have a categoryid? 

# In[17]:


df[df.categoryid.isnull()].groupby('category').first()

#There are 5 categories that dont have categoryids


# In[18]:


#Creating masks for selecting rows with specific category that has NaN in column categoryid. 
#Adding a new number for id in categoryid field of those rows. 

mask = (df['category'] == 'Shops')
df['categoryid'] = df['categoryid'].mask(mask, 201)

mask = (df['category'] == 'Service')
df['categoryid'] = df['categoryid'].mask(mask, 202)

mask = (df['category'] == 'Food and Drink')
df['categoryid'] = df['categoryid'].mask(mask, 203)

mask = (df['category'] == 'Healthcare')
df['categoryid'] = df['categoryid'].mask(mask, 204)

mask = (df['category'] == 'Community')
df['categoryid'] = df['categoryid'].mask(mask, 205)

mask = (df['category'] == 'Recreation')
df['categoryid'] = df['categoryid'].mask(mask, 206)


#Check:
assert df[df.categoryid.isnull()].empty


# ### Dealing with rows that have <=0 amount (amountnum column) (outliers) 

# In[19]:


df['amountnum'].describe() # min value is -ve (could be reimburement of some sort??). 75% of amount is under $52. 
#Max value of 24800 needs to be examined


# In[20]:


#Deleting rows that have -ve amount. Seems like those are refunds given to the customers
#Deleting rows that have 0 amount. I dont think these will help us build recomm system 

df[df.amountnum <=0].sort_values(by='category') #100 rows. Shows that these rows are from alot of categories

df[df.amountnum <=0]['category'].unique() #['Service Charges/Fees', 'Travel', 'Clothing/Shoes',
       #'Restaurants/Dining', 'Service', 'Food and Drink', 'Payment',
       #'Shops', 'Gasoline/Fuel', 'Healthcare/Medical', 'Groceries',
       #'General Merchandise', 'Other Expenses', 'Telephone Services',
       #'Gifts', 'Home Improvement', 'Dues and Subscriptions']

rowstodel=df[df.amountnum <=0].index
df.drop(rowstodel, inplace=True)


# In[21]:


df['amountnum'].describe() # describe after deleting -ve or 0 amount


# #### Dealing with Outliers in amountnum

# In[22]:


len(df[df.amountnum > 6000]) # since majority of data has amount <=$52 (75% of data), I am considering all amounts > 6000 as outliers


# In[23]:


#There are only 44 lines in amoutnum>6000 including royal swimming pool stuff. 
#A few airline purchanse of > 8K
# "Other expenses" category has one 61K row and doesn't specify what it is for. 
#Deleting those will not have severe impact on the df

rowstodel=df[df.amountnum > 6000].index
rows_to_del_with_index(df,rowstodel)

df.reset_index(inplace=True, drop=True)


# In[24]:


df.info() #non null rows = 69309


# In[25]:


df.drop_duplicates(inplace=True)
df.reset_index(inplace=True, drop=True)
df.info() # 52757


# In[ ]:


### There are some categorys with wrong merchants (simple_description column). Example: rockymoutnain kung fu is a sprots or something like that


# In[26]:


#### NOT SURE WHY THIS CODE IS NOT WORKING 

#df[(df.city=='Westminster') & (df.state.isnull())]
#mask=((df.city=='Westminster') & (df.state.isnull())
#df['state']==df['state'].mask(mask,'CO')

#assert df[(df.city=='Westminster') & (df.state.isnull())].empty


# In[27]:


#Question for Ana: How can I change state of following rows to "CO" [for city=='Westminster']. I have tried several ways but none seems to work

#df[(df.city=='Westminster')]


# #### Still alot of NaNs in several columns.Columns amountnum, category, categoryid, date, simple_description, uid and accountid have no NaN in any of the rows. For amount/category analysis and date/time analysis of the data this should be a good start

# # PLOTS

# ### Number of Transactions per State

# In[28]:


df_state=df.groupby(['state'])['state'].count()
df_state.plot.bar(figsize=(13,7))

plt.title('Number of Transactions per State', fontsize=20)
plt.xlabel("State", fontsize=18)
plt.ylabel("Number of Transactions", fontsize=18)
plt.legend('', frameon=False)


# #### Short commentary on above graph "Number of Transactions per State"

# From above graph: CO has the most records (approx 7K)in this data, Next is CA (approx 4K) , followed by IL, NC, WA, and NY. 
# All other states have considerably less number of records

# ### Number of Transactions per Category

# In[29]:


df_cat=df.groupby(['category'])['category'].count()
df_cat.plot.bar(figsize=(13,7))

plt.title('Number of Transactions per Category', fontsize=20)
plt.xlabel("Category", fontsize=18)
plt.ylabel("Number of Transactions", fontsize=18)
plt.legend('', frameon=False)


# #### Short commentary on above graph "Number of Transactions per Category"

# From above graph: the majority of records are coming from "Resturants/Dinning" Category.
# Followed by "General Merchandise", "Shops", "Groceries", "Service", "Travel". There are other categories even though with less records could be significant in recommdation system

# ### Total Amount Spent per Category

# In[30]:


fig, ax = plt.subplots(1, 1, figsize=(13, 7))

fmt = '${x:,.0f}'
df_amt_cat=df[['amountnum','category']].groupby('category').sum('amountnum')
df_amt_cat.plot.bar(ax=ax)


plt.title('Total Amount($) Spent per Category', fontsize=20)
plt.xlabel("Category", fontsize=18)
plt.ylabel("Amount($)", fontsize=18)

tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick) 

plt.legend('', frameon=False)

#df[['amountnum','category']].groupby('category').sum('amountnum').plot.bar()


# #### Short commentary on above graph " Amount (\$\) Spent per Category"

# The category where most $ were spent is "General Merchandise". "Service", "Travel", 
# "Restuarants/Dinning", "Groceries" come next.
# There are a few categories that have almost zero amount spent. 
# question for Ana: Should I consider deleting rows per category where amount is significantly less?

# ## Date/Time Analysis

# Creating a copy of 'df' with date as index. Also reordering columns to better view the data frame

# In[31]:


df1=df[['date','uid','category','categoryid','simple_description','amountnum','city','state','zip_code','address','latitude','longitude','accountid']]
df1=df1.set_index('date')
df1.index=df1.index.normalize()
df1.info() #non-null rows=52757


# ### Highest Mean Amount($) Per Category

# Which category has highest mean amount in each month?

# In[32]:


df_Office_Supply = pd.DataFrame(columns = ['Month', 'Highest_Cat', 'Mean_Amount'])

for month in list(df1.index.month.unique()):
    df1_mean=df1[df1.index.month==month].groupby(['category']).mean('amountnum')['amountnum']
    cat=df1_mean.sort_values(ascending=False).index[0]
    amt=df1_mean.sort_values(ascending=False)[0]
    df_Office_Supply=df_Office_Supply.append({'Month' : month, 'Highest_Cat' : cat, 'Mean_Amount' : amt}, ignore_index=True)

#Plot
sns.catplot(x="Month", y="Mean_Amount", hue='Highest_Cat', data=df_Office_Supply);

df_Office_Supply.set_index('Month').sort_values('Mean_Amount', ascending=False)


# Looks like the months 3,4,5 and 7 have highest mean amount in the category "Office Supplies".
# 
# Dissecting the "Office Supplies" category to see why is this happening?

# In[33]:


#March
df1[(df1.index.month==3) & (df1.category=='Office Supplies')] #two rows from two different uid. $5637.50 exists in this month


# In[34]:


#April
df1[(df1.index.month==4) & (df1.category=='Office Supplies')] 
#two rows from two different uid. $5637.50 ALSO exists in this month. Same exact row as above, except the date is now 2020-04-22 instead of 2020-03-22


# In[35]:


#May
df1[(df1.index.month==5) & (df1.category=='Office Supplies')]
#two rows from two different uid. $5637.50 ALSO exists in this month.Same exact row as above, except the date is now 2020-05-20


# In[36]:


#June
df1[(df1.index.month==6) & (df1.category=='Office Supplies')]
#two rows from two different uid. $5637.5 ALSO exists in this month.Same exact row as above, except the date is now 2020-06-25


# In[37]:


#July
df1[(df1.index.month==7) & (df1.category=='Office Supplies')]
#two rows from two different uid. $5637.75 ALSO exists in this month.Same exact row as above, except the date is now 2020-07-26


# #### Keeping one entry per accountid in one of the months that it exists in.

# In[38]:


#For accountid=10082151.0, deleting row from May, June, July. Keeping only March's row as is


# In[39]:


#March
#Checking:

assert len(df1[(df1.index.month==3) & (df1.category=='Office Supplies') & (df1.accountid==10082151.0)]) !=0


# In[40]:


df1.drop(df1[(df1.index.month==5) & (df1.category=='Office Supplies') & (df1.accountid==10082151.0)].index, inplace=True)

#Checking:
assert df1[(df1.index.month==5) & (df1.category=='Office Supplies') & (df1.accountid==10082151.0)].empty


# In[41]:


df1.drop(df1[(df1.index.month==6) & (df1.category=='Office Supplies') & (df1.accountid==10082151.0)].index, inplace=True)

#Checking:
assert df1[(df1.index.month==6) & (df1.category=='Office Supplies') & (df1.accountid==10082151.0)].empty


# In[42]:


df1.drop(df1[(df1.index.month==7) & (df1.category=='Office Supplies') & (df1.accountid==10082151.0)].index, inplace=True)

#Checking:
assert df1[(df1.index.month==7) & (df1.category=='Office Supplies') & (df1.accountid==10082151.0)].empty


# ### Why the mean amount in the month of August sky rocketed?

# In[43]:


#From graph titled "Highest Mean Amount($) Per Category", the max mean amount in the month of Aug is in cat='General Merchandise"
#Dissecting August


# In[44]:


len(df1[(df1.category=='General Merchandise') & (df1.index.month==8)].sort_values(by='amountnum', ascending=False))

#clearly just like multiple similar entries of "Office Supplies" category in April,in Aug there are several GM rows. 


# ### NOTE: Looks like there are several "similar" transations in General Merchandise category. I am going to stop going through each line and delete kinda duplicates for every category, as it will bring total transactions way down. 

# ### Mean Amount (\$\) Spent per Month

# In[45]:


fig, ax = plt.subplots(1, 1, figsize=(15, 8))
df1['amountnum'].resample('M').mean().plot(ax=ax)

plt.title('Mean Amount($) spent per Month', fontsize=20)
plt.xlabel("Month", fontsize=18)
plt.ylabel("Mean Amount($)", fontsize=18)


# #### Short commentary on above graph "Mean Amount Spent per Month"

# There are clearly peaks (high and low) in different periods. I can't spot a pattern.

# In[46]:


#Mean Amount spent per category (all months)
df1.groupby(['category']).mean('amountnum')['amountnum'].sort_values(ascending=False) 


# ### Mean Amount Per Category Per Month Starting From Jan 2020

# In[47]:


plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 12
fig, ax =plt.subplots(4,2)
fig.set_size_inches(25, 35, forward=True)
catPerMonthGraph2(1,df1,ax[0,0])
catPerMonthGraph2(2,df1,ax[0,1])
catPerMonthGraph2(3,df1,ax[1,0])
catPerMonthGraph2(4,df1,ax[1,1])
catPerMonthGraph2(5,df1,ax[2,0])
catPerMonthGraph2(6,df1,ax[2,1])
catPerMonthGraph2(7,df1,ax[3,0])
catPerMonthGraph2(8,df1,ax[3,1])
plt.tight_layout()

#Jan 2020: most mean amount spent was on pets and least amount was spent on "office supplies"
#Feb 2020: most mean amount spent was on Cable/Satellite Services and least was spent on "Advertising"
#March 2020: Most mean amount was spent on "Office Supplies" and least was on everything else
#April 2020: Looks pretty much same as March 2020
#May 2020: Most mean amount was spent on "Home Maintenance" and least amount was spent on "office supplies"
#June 2020: Most mean amount was spent on "Health Care" and least amount was on "Electronics"
#July 2020: Most mean amount was spent on Education and least mean amount was spent on "Food and Drink"
#Aug 2020: Most mean amount was spent on General Merchandise and least mean amount was spent on "Postage and Shipping"


# In[48]:


df1.info()


# In[50]:


df.info()


# In[ ]:




