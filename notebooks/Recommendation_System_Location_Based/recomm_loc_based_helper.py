import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import hdbscan
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.metrics import silhouette_score


#Functions for Recommendation_System_Location_Based_Pre_Work.ipynb

def elbowMethod(ax,cluster_df,maxNum):
    inertia = []
    K = range(1,maxNum)
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(cluster_df)
        inertia.append(kmeans.inertia_)
    
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method showing the optimal k')
#######################################################################################################################################
   
    
def col_transformed(df,col,colname):
    enc=preprocessing.LabelEncoder()
    enc=enc.fit(df[col])
    df[colname]=enc.transform(df[col])
#######################################################################################################################################
   
    
def findKmeans(cluster_df, numCluster):
    kmeans=KMeans(n_clusters=numCluster,random_state=45).fit(cluster_df)
    pred_clusters = kmeans.predict(cluster_df)
    pred_centers= kmeans.cluster_centers_
    kmeans.fit(cluster_df)
    y = kmeans.labels_
    print(" ")
    print("k =  " + str(numCluster) + " silhouette_score ", silhouette_score(cluster_df, y, metric='euclidean'))
    print(" ")
    return pred_clusters,kmeans

#######################################################################################################################################

def addPredictedClusters(df,clusterCol,columnName):
    dfcopy=df.copy()
    dfcopy[columnName]= clusterCol
    df=dfcopy
    return df
#######################################################################################################################################

def mapVisualize(df,lat_col,long_col,color_col, title, zoomNum=3, hover_data_cols= ['category','city','merchant', 'latitude', 'longitude']):
   
    fig=px.scatter_mapbox(df, lat=lat_col, lon=long_col, color=color_col,title=title, zoom=zoomNum,
                  hover_data= hover_data_cols, mapbox_style="open-street-map")
    
    fig.show()
#######################################################################################################################################
    
def numOfClusters(df,clusterCol):
    df_dist=df.groupby(clusterCol)[clusterCol].count().rename('count').to_frame()
    return df_dist.T
#######################################################################################################################################

def rows_to_del_with_index(df,rowstodel):
    df.drop(rowstodel, inplace=True)
#######################################################################################################################################

def fill_NaN_between_two_columns(df, col1, col2):
    df[col1]= df[col1].fillna(df[col1].groupby(df[col2]).transform('first'))
    df[col2] = df[col2].fillna(df[col2].groupby(df[col1]).transform('first'))
    
#######################################################################################################################################

def dataframe_with_top_merchants(df):
    recomm_df = df.copy()
    recomm_df = (recomm_df.groupby(['cluster', 'merchant']).agg({'latitude' : 'first',
                                                  'longitude' : 'first',
                                                  'city' : 'first',
                                                  'category' : 'first',
                                                  'cluster' : 'count'})
          .rename({'cluster' : 'cluster_count'},axis=1).reset_index()
          .sort_values(['cluster', 'cluster_count'], ascending = [True, False])
          .drop('cluster_count', axis=1))
    return recomm_df
#######################################################################################################################################

def test_data_point_extractor(df,rand_state=42):
    test_index=df.sample(random_state=rand_state).index
    
    test_lat=df.iloc[test_index[0]]['latitude']
    test_long=df.iloc[test_index[0]]['longitude']
    test_city=df.iloc[test_index[0]]['city']
    test_merchant=df.iloc[test_index[0]]['merchant']
    
    return test_lat, test_long, test_city, test_merchant

##################################################################################################################################
# Recommendation Engine Code:

def recommend_co_merchants_hdb(df,lat,long,city,merchant,cluster_object):
    # Predict the cluster for longitude and latitude provided
    test_labels, strengths = hdbscan.approximate_predict(cluster_object, [[lat,long]])
    predicted_cluster=test_labels[0]
    print('Predicted cluster for this lat/long combination is: '  + str(predicted_cluster))
    print("_______________________________________________________________________________")
     
    if predicted_cluster==-1:
        return ('No merchants close by')
    # Get the best merchant in this cluster
    else:
        pop_merch_recomm_df=(df[df['cluster']==predicted_cluster].iloc[0:5][['merchant','city','latitude','longitude']])
        pop_merch_recomm_df=pop_merch_recomm_df.reset_index(drop=True)
        mask = (pop_merch_recomm_df.merchant==merchant) & (pop_merch_recomm_df.latitude==lat) & \
               (pop_merch_recomm_df.longitude==long)
        print ('Since you are currently in '+ city.capitalize() + ' ' + 'at ' + \
               merchant.capitalize() + ', how about you visit these merchants around this area? ')
    return pop_merch_recomm_df[~mask]
