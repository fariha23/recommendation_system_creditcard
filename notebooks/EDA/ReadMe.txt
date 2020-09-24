This folder contains EDA on the data provided.
Here are the function descriptions in eda_helper.py files. 

#EDA helper functions

1. Function Name: fill_NaN_between_two_columns(df, col1, col2)
   Inputs: data frame, first column, second column
   Output: The data frame with new values where ever NaN exists in their cells using each other's fields
   
   
2. Function Name: rows_to_del(df, column, kw_list):
   Inputs: data frame, column name, keyword list of values in the column
   Output: data frame with dropped rows that have values matched to kw_list in the column
 
3. Function Name: catPerMonthGraph2(month,df,ax)
   Inputs: Month for which categories' distributions need to be observed,
		  data frame, axis from the plt.subplots() method
   Output: Graph showing categories distributions in the month specified
   
 
           