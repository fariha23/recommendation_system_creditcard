This folder contains code for recommendation_system_location_based (pre-work, HDBSCAN and KMeans)

Here are the function descriptions of recomm_loc_based_helper.py file. 

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

7. Function Name: dataframe_with_top_merchants(df):
   Inputs:  The data frame (df) with longitude, latitude, cluster, category and city features
   Output:  The data frame grouped by cluster number and merchant within it, aggregated by first value of latitude,longitude, city,category and count of each of those rows.
            The final results are sorted by cluster number (ascending) and count of rows (cluster_count) by descending order 

8. Function Name: test_data_point_extractor(df,rand_state=42)
   Inputs:  The data frame from which the data point is to be extracted, a number for random_state variable of .sample() function.(default=42)
   Outputs: Values of the test points' features, latitude, longitude, city, and merchant. 

9. Function Name: recommend_co_merchants_hdb(df,lat,long,city,merchant,cluster_object) 
   Inputs: The data frame (recomm_df) that was returned from dataframe_with_top_merchants, test data point's lat,long,city,merchant and hdbscan's invoked object name    
   Output: Recommendations using the best possible matches (top 5 merchants) to test data point's feature values. 
   
Descriptions for 'rows_to_del_with_index', and 'fill_NaN_between_two_columns' can be found in ReadMe of EDA folder. 
 
