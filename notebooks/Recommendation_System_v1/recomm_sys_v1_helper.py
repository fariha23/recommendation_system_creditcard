import pandas as pd
import numpy as np
#from eda_helper import rows_to_del_with_index
from sklearn.metrics.pairwise import cosine_similarity
#Thresholds are in main ipynb for easy of use

#functions for recommendation_system_v1.ipynb

def rows_to_del_with_index(df,rowstodel):
    df.drop(rowstodel, inplace=True)
############################################################################################################################

def merchCleanup(df, deleteList):
   
    for merch in deleteList:
        rowstodel=df[df.merchant==merch].index
        rows_to_del_with_index(df,rowstodel)
    df.reset_index(inplace=True, drop=True)
    return df

##############################################################################################################################

def top_merch(df,currentCatid,top_num):
    return df[df.categoryid==currentCatid].groupby('merchant')['merchant'].count().sort_values(ascending=False).head(top_num).index.to_list()

##############################################################################################################################

def createCatDF(df, categoryId, merch_list):
    df_cat=df[df['merchant'].isin(merch_list) & (df.categoryid==categoryId)]
    df_cat.reset_index(inplace=True,drop=True) 
    return df_cat

##############################################################################################################################

def takeInputMerch(catid, df_cat,top_list):
    default={22:"mcdonald's", 201:'target',44.0:'walmart',202:'geico',10:'publix',23:'uber',203: 'sonic',5.0: 'gap',7.0:'hulu',\
            8.0:'shell',13.0:'true value',11.0:'walgreens'}
    print("Please select one of the merchants from this list. ")
    merchant_user_visited=input(top_list)
    
    if merchant_user_visited=='':
        merchant_user_visited=default[catid]
    else: merchant_user_visited = merchant_user_visited
    return merchant_user_visited
    
##############################################################################################################################

def filters(currentCatId,df,top_merch_list):

    #1 (filter #1) Num of Records per merchant in top 15.
    count_per_merch=df.groupby(['merchant'])['merchant'].count()
    #count_per_merch

    #2 (filter #2) Mean amount of $ spent per merchant in top 15.
    meanamt_per_merch=df.groupby(['merchant'])['amountnum'].mean()
    #meanamt_per_merch

    return count_per_merch, meanamt_per_merch
    
##############################################################################################################################

def recommendationSystem(merchant_user_visited,SimMethod,df_mtx,merchant_popularity_count,count_per_merch, meanamt_per_merch,Threshold_Sim_Score,\
                         Threshold_Num_MeanAmt, Threshold_Num_Visit,top_merch_list):
    
    if SimMethod=='Pearson':
        similar_to_merchant = df_mtx.corrwith(merchant_popularity_count)
        #create data frame using series "similar_to_merchant"
        similar_to_merchant = pd.DataFrame(similar_to_merchant, columns=[SimMethod])
        #print(similar_to_merchant)
        
    if SimMethod=='Cosine':
        
        temp=cosine_similarity(df_mtx.T)
        temp_df=pd.DataFrame(temp, columns=top_merch_list, index=top_merch_list)
        similar_to_merchant= temp_df[merchant_user_visited]
        similar_to_merchant=pd.DataFrame(similar_to_merchant)
        similar_to_merchant.columns=[SimMethod]
        #print(similar_to_merchant)
        

    
    # Joining similar_to_merchant with count_per_merch(filter1) and meanamt_per_mearch(filter2)
    merchant_corr_summary = similar_to_merchant.join(count_per_merch).join(meanamt_per_merch)
    merchant_corr_summary.rename(columns={"merchant": "Num_Times_Merch_Was_Visited",'amountnum':'Mean_Amt_Per_Merch'}, inplace=True)
    #merchant_corr_summary.sort_values('Num_Times_Merch_Was_Visited',ascending=False)

   
    final_recommendation = merchant_corr_summary[(abs(merchant_corr_summary[SimMethod]) >= Threshold_Sim_Score) &\
                                                     (merchant_corr_summary.Num_Times_Merch_Was_Visited > Threshold_Num_Visit)\
                                                     & (merchant_corr_summary.index!= merchant_user_visited) &\
                                                     (merchant_corr_summary.Mean_Amt_Per_Merch > Threshold_Num_MeanAmt)].sort_values(SimMethod, ascending=False).head(5)
    print("Other merchants you may like {}".format(final_recommendation.index.tolist()))
    
##############################################################################################################################

def recommendationSystem_CosineAndPearson(merchant_user_visited,df_mtx,merchant_popularity_count,count_per_merch,meanamt_per_merch,\
                                          Threshold_Sim_Score,Threshold_Num_MeanAmt,Threshold_Num_Visit,top_merch_list):
    #Calculate Pearson Score

    similar_to_merchant_pearson =df_mtx.corrwith(merchant_popularity_count)
    #create dataframe using series "similar_to_merchant"
    similar_to_merchant_pearson = pd.DataFrame(similar_to_merchant_pearson, columns=['Score'])

        
    #Calculate Cosine Score

    temp=cosine_similarity(df_mtx.T)
    temp_df=pd.DataFrame(temp, columns=top_merch_list, index=top_merch_list)
    similar_to_merchant_cosine = temp_df[merchant_user_visited]
    #create dataframe using series "similar_to_merchant"
    similar_to_merchant_cosine=pd.DataFrame(similar_to_merchant_cosine)
    similar_to_merchant_cosine.columns=['Score']


    #create similarity summary data frame for pearson
    merchant_corr_summary_pearson = similar_to_merchant_pearson.join(count_per_merch).join(meanamt_per_merch)
    merchant_corr_summary_pearson.rename(columns={"merchant": "Num_Times_Merch_Was_Visited",'amountnum':'Mean_Amt_Per_Merch'}, inplace=True)



    #create similarity summary data frame for cosine
    merchant_corr_summary_cosine = similar_to_merchant_cosine.join(count_per_merch).join(meanamt_per_merch)
    merchant_corr_summary_cosine.rename(columns={"merchant": "Num_Times_Merch_Was_Visited",'amountnum':'Mean_Amt_Per_Merch'}, inplace=True)



    #create final_recommendation df for pearson
    final_recommendation_pearson=merchant_corr_summary_pearson[(abs(merchant_corr_summary_pearson['Score']) >= Threshold_Sim_Score) &\
                                                     (merchant_corr_summary_pearson.Num_Times_Merch_Was_Visited > Threshold_Num_Visit)\
                                                     & (merchant_corr_summary_pearson.index!= merchant_user_visited) &\
                                                     (merchant_corr_summary_pearson.Mean_Amt_Per_Merch > Threshold_Num_MeanAmt)].sort_values('Score', ascending=False).head(5)

    #create final_recommendation df for cosine
    final_recommendation_cosine=merchant_corr_summary_cosine[(abs(merchant_corr_summary_cosine['Score']) >=Threshold_Sim_Score ) &\
                                                     (merchant_corr_summary_cosine.Num_Times_Merch_Was_Visited > Threshold_Num_Visit)\
                                                     & (merchant_corr_summary_cosine.index!= merchant_user_visited) &\
                                                     (merchant_corr_summary_cosine.Mean_Amt_Per_Merch > Threshold_Num_MeanAmt)].sort_values('Score', ascending=False).head(5)



    
    #Return the best scored merchant from both similarity matrix
    if (len(final_recommendation_cosine) >= 3):
        final_recommendation=final_recommendation_cosine
    
    elif ((len(final_recommendation_cosine) < 3) & (len(final_recommendation_pearson) < 3)):
        final_recommendation=final_recommendation_cosine.append(final_recommendation_pearson)
    
    elif (len(final_recommendation_cosine)==0):
        final_recommendation=final_recommendation_pearson
    
    else:
        final_recommendation=final_recommendation_cosine

    print("Other merchants you may like {}".format(final_recommendation.index.tolist()))
                                          
##############################################################################################################################
