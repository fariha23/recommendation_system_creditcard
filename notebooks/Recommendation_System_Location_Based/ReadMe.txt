This folder contains code for recommendation_system_location_based (pre-work, HDBSCAN sol and KMeans (experiment)

Here are the functions descriptions in recomm_loc_based_helper.py files. 

#recomm_loc_based_helper.py

1. Function Name:elbowMethod(ax,cluster_df,maxNum)
   Inputs: axis(ax) from plt's subplot method, data frame with features to be used for clustering(cluster_df),
           maximum numbers(maxNum) of k (clusters) to find optimal k
   Output: A elbow_knee plot of the given cluster
   
2. Function Name:col_transformed(df,col,colname)
   Inputs: data frame (df), categorical column (col) to be transformed into integers for clustering, name of the new column(colname) 
   Output: Updated df with the new column name(colname)
 
 
3. Function Name: findKmeans(cluster_df, numCluster):
   Inputs: data frame (cluster_df) for which KMeans clustering method is to be used, number of clusters to be formed using KMeans fit method
   Output: predicted clusters data(pred_clusters), kmeans object used to fit and predict data, prints 'Silhouette score for each value of k'

4. Function Name:addPredictedClusters(df,clusterCol,columnName)
   Inputs: data frame (df), pred_clusters from findKmeans func and name of the column to be added
   Output: df with predicted cluster values in new column (columnName)

5. Function Name: mapVisualize(df,lat_col,long_col,color_col, title, zoomNum=3, hover_data_cols= ['category','city','merchant', 'latitude', 'longitude']):
   Inputs:  dataframe (df), latitude column (lat_col), longitude column(long_col), title of the map to be given, 
            Zoom In or Out Value (zoomNum), data shown when hovering mouse on a data point (hover_data_cols).

6. Function Name: numOfClusters(df,clusterCol):
   Inputs:  data frame (df) which has predicted cluster values added. Name of the column containing predicted cluster values (clusterCol)
   Output:  Table with predicted number of clusters as columns. Rows are the total number of records tagged for a specific cluster

Descriptions for 'rows_to_del_with_index', and 'fill_NaN_between_two_columns' can be found in ReadMe of EDA folder. 
 
