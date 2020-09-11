import pandas as pd
from EDA import df1
import numpy as np
from eda_helper import rows_to_del_with_index
from sklearn.metrics.pairwise import cosine_similarity

#Thresholds
Threshold_P = 0.3
Threshold_C = 0.7
Threshold_Cosine=0.6
Threshold_Num_Visit = 6
Threshold_Num_MeanAmt = 10

#functions for recommendation_system_v1.ipynb
def merchCleanup(df, deleteList):
   
    for merch in deleteList:
        rowstodel=df[df.merchant==merch].index
        rows_to_del_with_index(df,rowstodel)
    df.reset_index(inplace=True, drop=True)
    return df


def top_merch(df,currentCatid,top_num):
    
    return df[df.categoryid==currentCatid].groupby('merchant')['merchant'].count().sort_values(ascending=False).\
           head(top_num)

def createCatDF(df, categoryId, merch_list):
    df_cat=df[df['merchant'].isin(merch_list) & (df.categoryid==categoryId)]
    df_cat.reset_index(inplace=True,drop=True) #8419 records
    return df_cat

def takeInputMerch(catid, df_cat,top_list):
    default={22:"mcdonald's", 201:'target',44.0:'walmart',202:'geico',10:'publix',23:'uber',203: 'sonic',5.0: 'gap',7.0:'hulu',\
            8.0:'shell',13.0:'true value',11.0:'walgreens'} 
    merchant_user_visited=input(top_list)
    if merchant_user_visited=='':
        merchant_user_visited=default[catid]
    else: merchant_user_visited = merchant_user_visited
    return merchant_user_visited
    
def takeInputUser(df_cat): 
    user_list=df_cat.uid
    userid = input(user_list)
    if userid=='':
        userid='A4Kyy9PFU7QLr08iECIlJOMI0zx2'
    else: userid=userid
    return userid

def filter1_count_per_merch(df_cat):
    return df_cat.groupby(['merchant'])['merchant'].count()

def filter2_meanamt_per_merch(df_cat):
    return df_cat.groupby(['merchant'])['amountnum'].mean()


def rec_df_merch_list(df,currentCatId, top_num=15):
    
    #creating DF of all the remaining records of current category
    top_merch_df=top_merch(df, currentCatId, top_num)
    top_merch_list=top_merch_df.index.to_list()
    
    df1_rec_category = createCatDF(df, currentCatId, top_merch_list) 
    
    return df1_rec_category, top_merch_list

def filters(currentCatId,df, top_merch_list):

    merchant_user_visited=takeInputMerch(currentCatId, df, top_merch_list)
    
    userid=takeInputUser(df1_rec_category)
     
    #1 (filter #1) Num of Records per merchant in top 15.
    count_per_merch=filter1_count_per_merch(df1_rec_category)
    #count_per_merch

    #2 (filter #2) Mean amount of $ spent per merchant in top 15.
    meanamt_per_merch=filter2_meanamt_per_merch(df1_rec_category)
    #meanamt_per_merch
    return userid, merchant_user_visited,count_per_merch, meanamt_per_merch
    
def sparseMtx(df):
    
    
    #sparse matrix of merchants and users. Values are the number of times a user went to a resturant.

    df1_rec_merch_count_mtx=df.pivot_table(index='uid',columns='merchant', values='amountnum',\
                                                         aggfunc='count', fill_value=0)
    #df1_rec_merch_count_mtx 
    
    #3 Create matrix showing how popular the merchant "merchat_user_visited" was with users in the data
    merchant_popularity_count=df1_rec_merch_count_mtx[merchant_user_visited] #How many times each user visited the "merchant_user_visited"
    return df1_rec_merch_count_mtx, merchant_popularity_count


    #df1_rec_merch_count_mtx
    return df1_rec_merch_count_mtx, merchant_popularity_count


def recommendationSystem(userid,merchant_user_visited,SimMethod,df_mtx,merchant_popularity_count,filter1,filter2):
    if df_mtx.get(userid)==None:
        user_merchant_visits=2
    else: user_merchant_visits=df_mtx.loc[userid][merchant_user_visited]
    
    if SimMethod=='Pearson':
        Threshold_Sim=Threshold_P
        similar_to_merchant = df_mtx.corrwith(merchant_popularity_count)
        #create dataframe using series "similar_to_merchant"
        similar_to_merchant = pd.DataFrame(similar_to_merchant, columns=[SimMethod])
        
    if SimMethod=='Cosine':
        Threshold_Sim=Threshold_P
        temp=cosine_similarity(df1_rec_merch_count_mtx.T)
        temp_df=pd.DataFrame(temp, columns=top_merch_list, index=top_merch_list)
        similar_to_merchant_cosine = temp_df[merchant_user_visited]
        similar_to_merchant=pd.DataFrame(similar_to_merchant_cosine)
        similar_to_merchant.columns=[SimMethod]

  
    
    # Joing similar_to_merchant with count_per_merch(filter1) and meanamt_per_mearch(filter2)
    merchant_corr_summary = similar_to_merchant.join(filter1).join(filter2)
    merchant_corr_summary.rename(columns={"merchant": "Num_Times_Merch_Was_Visited",'amountnum':'Mean_Amt_Per_Merch'}, inplace=True)
    merchant_corr_summary.sort_values('Num_Times_Merch_Was_Visited',ascending=False)
        
    #print(merchant_corr_summary)
        
    if 0 <= user_merchant_visits < 10:
        final_recommendation=\
        merchant_corr_summary[(merchant_corr_summary[SimMethod] >= Threshold_Sim) & \
                                                       (merchant_corr_summary.Num_Times_Merch_Was_Visited > Threshold_Num_Visit) \
                                                       & (merchant_corr_summary.index!= merchant_user_visited) & \
                                                       (merchant_corr_summary.Mean_Amt_Per_Merch > Threshold_Num_MeanAmt)].sort_values(SimMethod, ascending=False).head(3)
        print("Folks like you, are enjoying {}".format(final_recommendation.index.tolist()))
    else: 
        final_recommendation=np.nan
        #print(final_recommendation)