This folder contains code for Recommendation_System_v1 (recommendation based on merchant similarities)
Here are the function descriptions in eda_helper.py files. 

#recomm_sys_v1 helper functions

1. Function Name: merchCleanup(df, deleteList)
   Inputs: data frame (df), list of the merchants to be deleted 
		   from the merchant(simple_description) column (deleteList)
   Output: df with the dropped rows that matched values in deleteList
   
   
2. Function Name: top_merch(df,currentCatid,top_num):
   Inputs: data frame (df), category id (currentCatid) for which top merchants are needed, 
		   number of top merchants (top_num) 
   Output: top 'top_num' sorted list of top merchants in category 'currentCatId'
   
3. Function Name: createCatDF(df, categoryId, merch_list)
   Inputs:  data frame (df), category id (categoryId), 
		    list of merchants to be used to create a new data frame(merch_list)
   Output:  data frame with rows belonging to the category 'categoryId' and 
            merchant column matching values in 'merch_list'
			
4. Function Name: takeInputMerch(catid, df_cat,top_list):
   

   
 
           