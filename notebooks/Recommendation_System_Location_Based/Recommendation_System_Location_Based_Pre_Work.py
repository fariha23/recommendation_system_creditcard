#!/usr/bin/env python
# coding: utf-8

# ### IMPORT PACKAGES & DATA MINING 

# In[7]:


import pandas as pd
from EDA import df1
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from eda_helper import rows_to_del_with_index
from sklearn import preprocessing
from eda_helper import fill_NaN_between_two_columns
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.metrics import silhouette_score


# In[8]:


#creating copy of dataframe from EDA notebook for this recommendation notebook
df2=df1 
df2=df2.reset_index() #changing index to int
df2.rename(columns={"simple_description": "merchant"}, inplace=True) #renaming column 'simple_description' to 'merchant'
df2.info()


# #### FUNCTIONS

# In[44]:


#Functions
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
    
    
def col_transformed(df,col,colname):
    enc=preprocessing.LabelEncoder()
    enc=enc.fit(df[col])
    df[colname]=enc.transform(df[col])
   
   
    
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


def addPredictedClusters(df,clusterCol,columnName):
    dfcopy=df.copy()
    dfcopy[columnName]= clusterCol
    df=dfcopy
    return df

def mapVisualize(df,lat_col,long_col,color_col, title, zoomNum=3, hover_data_cols= ['category','city','merchant', 'latitude', 'longitude']):
   
    fig=px.scatter_mapbox(df, lat=lat_col, lon=long_col, color=color_col,title=title, zoom=zoomNum,
                  hover_data= hover_data_cols, mapbox_style="open-street-map")
    
    fig.show()
    
def numOfClusters(df,clusterCol):
    df_dist=df.groupby(clusterCol)[clusterCol].count().rename('count').to_frame()
    return df_dist.T


# ### DATA WRANGLING 

# #### 1. Cleaning a few merchant's names that have some obvious words that are not in their actual names

# In[10]:


rest_list=[]
for merch in df2.merchant:
    if ((merch[0:8]=='*pending')):
        rest_list.append(merch[30:])
    elif (merch[0:7]=='pending'):
        rest_list.append(merch[23:])
    elif (merch[0:2]=='p '):
        rest_list.append(merch[11:25])
    elif (merch[0:2]=='pp'):
        rest_list.append(merch[3:])
    elif (merch[0:2]=='a '):
        rest_list.append(merch[7:])
    elif (merch[0:8]=='visa pp*'):
        rest_list.append(merch[8:])
    elif (merch[0:7]=='visa sq'):
        rest_list.append(merch[9:])
    elif (merch[0:4]=='visa'):
        rest_list.append(merch[5:26])
    elif (merch[0:2]=='sq'):
        rest_list.append(merch[4:])
    elif (merch[0:2]=='sp'):
        rest_list.append(merch[5:])
    elif(merch[0:7]=='*******'):
        rest_list.append(merch[7:])
    else:
        rest_list.append(merch)
        
sorted(rest_list)
df2.reset_index(inplace=True, drop=True)
df2['merchant']=rest_list


# #### 2. Creating subset for CO only from df2 as it has the most number of rows in the data

# In[11]:


##Focusing on CO only 
df2_co=df2[df2.state=='CO']
df2_co.reset_index(drop=True, inplace=True)#6693 rows
df2_co.info() #6693


# #### 3. Cleaning data as best as possible for CO only

# 3a: Swapping correct names for some merchants that contain certain strings

# In[12]:


dict_contains={
'unique nails':'unique nails',
'chateaux family':'chateaux family & cosmatics',
'hmart-aurora':'hmart-aurora',
'huakee bbq':'huakee bbq',
'hmart-westminster': 'hmart-westminster',
'five & hoek': 'five & hoek',
'basecamp': 'basecamp',
'batteries': 'batteries plus bulbs',
'benny': "benny's tacos",
'vapor distillery':'vapor distillery',
'twisted pine':'twisted pine brewery',
'tundra restauran': 'thundra restaurant',
'walnut cafe': 'walnut cafe',
'attic bar': 'attic bar',
'broadmoor': 'broadmoor',
'sundown saloon': 'sundown saloon',
'tune up': 'tune up',
'rayback': 'rayback collective',
'pho haus':'pho haus',
'pearl st': 'pearl st pub',
'fedex': 'fedex',
'blackjack': 'blackjack pizza',
'colorado plains': 'colorado plains',
"tommy's subs":"tommy's subs",
"thorobred liquors":"thorobred liquors",
'the sing':'the singing cook',
'one way disposal':'one way disposal',}

## Function to excecute above dictionary
for k,v in dict_contains.items():
    df2copy=df2_co.copy()
    mask=(df2copy.merchant.str.contains(k))
    df2copy['merchant']=np.where(mask,v,df2_co.merchant)
    df2_co=df2copy


# 3b: Swapping correct names for some merchants whose names starts with certain strings

# In[13]:


dict_startswith={'10th mountain': '10th mountain','12degree': '12degree brewing',
'aloha trading': 'aloha trading','ziggi':'ziggis coffee','ziggi':"ziggi's coffee",
'apple': 'apple',
'art mart': 'art mart gifts',
'7 hermits': '7 hermits',
'the bardo':'The Bardo Coffee','target':'target',
'smack daddy':'smack daddy',
'monument liquor':'monument liquor',
'king soop':'king soopers',
'alfalfa': "alfalfa's market",
'10th mountai':'10th mountain',
'the perch coffee':'the perch coffee',
"tommy's subs":"tommy's subs",
"thorobred liquors":"thorobred liquors",
'the market':'the market','the boulder book':'the boulder bookstore','the daily bread':'the daily bread',"broken tee":"broken tee",
' pizza co.':'pizza co.',  ' the tea':'the tea',
' urban winery ':'urban winery',
' usa':'usa',
' w hinman md l':'Hinman MD',
'sondermind':'sondermind',
'sweets':'sweets ice cream',
'taj mahal':'taj mahal',
'sweet cow denver':'sweet cow denver',
'sweet cow north boulder':'sweet cow boulder',
'smart cow 5 castle rock co usa':'smart cow',
'smack daddy piz':'smack daddy', 
'shells and sauce denver co 07/09':'shells and sauce',
'sei x6927 fort':'sei fort collins','scrooge maki':'scrooge maki','salon halcyon':'salon halcyon',
'roo jumps':'roo jumps','rio grande':'rio grande','playstation':'playstation',
'pinto chiropractic center':'pinto chiropractic center', 'pikes peak brewing':'pikes peak brewing',
'pica': "pica's mexican", 'ozo coffee':'ozo coffee','otis':"otis craft",
'olde glory':'olde glory','niwot liquor':'niwot liquor','murphy':'murphy usa',
'monument':"monument liquor",'mid vail':'mid vail','mcguckin hardwar':'mcguckin hardware',
'left hand brewing':'left hand brewing', 'jefe':"jefe tacos & tequila",'jax':'jax outdoor',
'jai thai':'jai thai','hacienda denver':'hacienda denver','gunbarrel':"gunbarrel liquor",
'grappa':"grappa fine wines", 'frontier':'frontier airlines', 
'front range brewin': 'front range brewing','european':'european market',
'einstein':"einstein bagels",'down under':'down under wine','dsw':'dsw','winot':'winot coffee'}

## Function to excecute above dictionary
#merch_name_startswith_del(df2_co,dict_startswith)

for k,v in dict_startswith.items():
    df2copy=df2_co.copy()
    mask=(df2copy.merchant.str.startswith(k))
    df2copy['merchant']=np.where(mask,v,df2_co.merchant)
    df2_co=df2copy


# 3c: Deleting some rows that match these strings as these are are ambiguous and probably were distroyed while making data anonymous

# In[14]:


list_to_del=[' ',"european *******et","/ beltran's meat market, l northglenn co date xx 5422",' towing sterling co #xx0595jqs66kvgsj','/ #xx0938 broomfield-util-pmnt xx19 co','/ #xx0752 broomfield-util-pmnt xx19 co','/ #xx0423 broomfield-util-pmnt xx19 co','/ busaba thai restaurant louisville co date xx 5812','/ mc2 ice cream company broomfield co date xx 5499','/ sq *areyto broomfield co date xx 5814','/ sq *kseni mademoise denver co date xx 5499','2020 eyevenue broomfield co 07/23','652 flight terminal denver co xx0249','affirm.com payme affirm.com st-o6d9s4z9x6x5 ******* *******','albert savings d edi pymnts 24685834 ******* *******','alpine wine & spiritvail co xx8116','animal c&c online xx3','axs.com*denver co','b.o.b.s. louisville co card9815']
list_to_del2=['backdoor gri steamboat spr co card0591','bc barber co denver co 07/08','boulder colonic cen niwot co on xx 1194','boulder general |boulder general sto boulder co us|card nbr: 0853','boyd lake state prk hpb loveland co 07/25','bp#8797383lindbergh & co','brewing *******et lon','brewing *******et coffee','brewing *******et','burgerfi denver denver co 07/21','candys dance academy xx05 co','cavegirl coffee longmont co on xx 7618','cdle ui benefits ui payment ppd id: 1840802678','centers for gastroentero fort collins co #xx2300dmks4q3a5x','checkcard 1205 georgia school of xx5220 co xx8394','checkcard 0105 georgia school of xx5220 co xx3387','cicino enterpri louisville co on xx 7618','cicino enterpri louisville co on xx 7618','citi card online des:payment id:xxxxx3990151571 indn:******* ******* co id:citictp web','city of loui louisville co card9815','colorado early fort collins co on xx 7618','columbia 553 thorton co 07/05','crossroads tradi |crossroads trading boulder co us|card nbr: 0853','cumberland far 07/20 purchase','dbt ares thrift stor |ares thrift store boulder co us|card nbr: 0853','dbt kathmandu restau |kathmandu restauran boulder co us|card nbr: 0853']
list_to_del3=['dbt dhha hospital an |dhha hospital and c denver co us|card nbr: 0853 00:00','dbt busey brews |busey brews nederland co us|card nbr: 0853','dbt maplecreek spa s |maplecreek spa scor boulder co us|card nbr: 0853','dbt mcguckin hardwar |mcguckin hardware boulder co us|card nbr: 0853','dbt pp*root llc |pp*root llc niwot co us|card nbr: 0853','dbt sq *glacier ice |sq *glacier ice cre boulder co us|card nbr: 0853','dbt sq *mcdevitt tac |sq *mcdevitt taco s boulder co us|card nbr: 0853','dbt thrivepass |thrivepass xx85 co us|card nbr: 0853','dbt tst* salto |tst* salto nederland co us|card nbr: 0853','dbt tst* verde - bou |tst* verde - boulde boulder co us|card nbr: 0853','dbt vapor distillery |vapor distillery xx6134 co us|card nbr: 0853 10:54','debit -visa card 0641 buff one card ofboulder co','debit card credit 7509345004 vis 0713 rep fitness ecommerce','digit.co savings v9vysua93dedwpz ******* *******','dog daze photo boulder co nt_gzybzv5h xx3556','dollar tr 600 s holly denver co 07/05','doordash*north italia','dpac garage-xx9093 de','e','electronic deposit sm usd 512','exa 0176 forever 2 littleton co','fitr.tv denver co nt_hevhhxla +xx7574','flying b bar ranch llc xx06 co','foxtail pines veterina xx0000 co', 'gosq.com','hall*******','hays *******et of b b','heroku *oct-xx2954 san franciscous','honolulu poke b johnstown co on xx 7618','icp*airborne gymnasti','ikon pass','ikon pass xx8200 co','in *word travel llc xx8940 co','in *wild view veterinary xx6212 co','in *synergy dog training xx2874 co','inc. moneyl visa direct wi079480 07/27','interest charge on purchases','interest payment',"jenny's *******et",'joy of sox breckenridge co on x 0x70($164.95)','keepthechange credit from acct3747 effective 07/20','little bite of pueblo co on xx 0498','louisville family hosp superior co pxx8389 card 9815', "lucky's *******et", 'marczyk fine fo denver co 07/26','marianjoy rehabilitati wheaton il 07/20', 'mimis cafe 133 westminster co 07/05','modern *******et','morgancnty qlty wtr dist xx3054 co #xx9487hns66k33t6','morgancnty qlty wtr dist xx3054 co #xx9487g5s66ett1w','morgancnty qlty wtr dist xx3054 co #xx9487e3s66j4k0f',"morton's organi lafayette co on x 1x28($57.00)",'mr ds ace home center fort morgan co #xx0197h12mb1rbp1','murder by death llc estes park jtmrrpyoiy7 squareup.com/receipts','national car rxx2167denver co r/a# xx7373','nbfs denver 13801 gran thornton co 07/05','ncal kaiser online pay 07/18 purchase','niwot wh niwot co card9815','njh-events xx6551 co ','nst longmont co xx030','omo st vrain *******e','online scheduled transfer to sav 0460 confirmation# 1464778215','online stores','online transfer from ******* b way2save savings xxxxxx9557 ref #ib08jhb6vb on 07/22/20','online transfer from sav ...8976 transaction#: 9982504545','online transfer to chk ...7034 transaction#: 9972015271 07/20','oreganos bistro 1024 fort collins co','payment to waste connections','payment to visa','payment to verizon','payment to speedpay.com','payment to sendgrid','payment to private internet access','payment to one way disposal','payment to morgan county rural water','payment to left hand water district','payment to dish network','payment to comcast','payment to city of longmont co','payment to city of boulder city','payment to chase card ending in 0503 07/20','payment thank you-mobile','payment thank you - web','petco 1437 63514376','pets and pals veterinary lafayette co','pho brothers ii 0000monument co xx02','pies & gri* pmonument co nt_hnclg6k2 +xx2469']
list_to_del4=['stride bank des:p2p id:******* ******* indn:bank of america co id:xxxxx22151 web','pos 018001 usa petro #68218 south el montca ##8443','rack attack denver internxx6900 co','re','recurring authorized name.com, inc xx2374 co sxx4299 card 0093','renee madison loveland co 07/17','red hawk raffle rhes.svvsd.orco','ritas house of pizza dbt crd 1152 07/21/20 07415779 winslow me c#0104','rocket fizz den denver co on xx 0498','rocky mountain whitew','s&s #302 greenwood vi','s&s #534 longmont co ',"s&s #536 boulder co o",'sai eyebrow des boulder co','savings od protection transfer fee','scofield fruits llc loveland co','sfts consulting sftsaurora co xx9285 misc apparel store','shop 2','simply bulk *******et longmont co on x 7x18($7.59)','sinclair i-70 golden co 07/05','southpark *******et f', 'sos registra denver co card9815','ssp*rockymoutnain kung fuxx2839 co', 'st louis restoration co.','st vrain super*******et','state of nj - la uemploymen ppd id: 4216000928','ste*knightscope','stickergiant.com xx00 co xx00','t2 parking w boulder co card9815','t2 parking 1 boulder co card9815','summit tacos longmont co on x 0x70($44.74)','stickergiant.com xx00 co xx00','stride bank des:p2p id:******* ******* indn:bank of america co id:xxxxx22151 web']
list_to_del5=["yeti's grind vail co sxx7553 card 9815",'withdrawal credit/debit card du xx15 stout 2015 stout st denver co date xx 7523 % card 06 #7407','withdrawal / sq *two arrows vail co date xx 5814',"withdrawal / sq *audrey jane's pizza g boulder co date xx 5812",'withdrawal / scrooge maki bento boulder co date xx 5814','withdrawal / pp*canoenterpr englewood co date xx 5999','withdrawal / cke*simple eatery 402 buena vista co date xx 5812','withdrawal / buffs wash2, llc boulder co date xx 7542','withdrawal / buff wash llc boulder co date xx 7542','withdrawal #xx5311 / mountain exchange 1608 miner street idaho springs co','vp*kci communications xx07 co','usa*vend at air servmonument co tdxx misc/specialty retail','usa*vend at air servmonument co tdxx fast food restaurant','usa*snack soda vending allendale', 'upstart network upst loans 558560 web id: ach4945590','university of northern c xx2201 co','univ n. co bkst #1249 greeley co']
list_to_del6=['tst* centro mexica boulder co sxx2391 card 4578',
 'tst* cheffin s cheese',
 'tst* foolish boulder co card4578',
 'tst* illegal boulder co card4578',
 'tst* la vita bella ca',
 'tst* laura s fine candie estes park co',
 'tst* motomak boulder co card4578',
 'tst* southside wal boulder co sxx2183 card 4578',
 'tst* the post - longm',
 'tst* west en boulder co card4578','tp* bodbalan evergreen co card0093',
 'tp* bodbalance kymbreannellc co sxx5747 card 0093',
 'tp* bodbalance kymbreannellc co sxx9432 card 0093',
 'tp* bodbalance kymbreannellc co sxx9898 card 0093','tip top savory boulder co on x 8x17($14.40)','the ice cave cider hmonument co f5p3feczlia squareup.com/receipts',\
 'pine cleaners 1x3170huntersvill xx9298 andystroud83@gmail.co', 'nnt cherry creek acxx',
 'nnt longmont arc thxx',
 'nnt the mountain foxx','dbc irrigation supply',
 'dbt boulder rdhs |boulder rdhs boulder co us|card nbr: 0853',
 'dbt crossroads tradi |crossroads trading boulder co us|card nbr: 0853',
 'dbt l2g*co springs t |l2g*co springs tran colorado spri co us|card nbr: 0853','college transcript il 07/17','color street llc 9736893088',
 'cicino enterpri louisville co on xx 0498','/ los #3136 broomfield co date xx 5631',
 '/ prosper foods company denver co date xx 5199','12 point disti lafayette co sxx9567 card 9815',
 '12th avenue ace hardwar denver co 07/09',
 '3 hundred days distimonument co rlqi8tvp1xm squareup.com/receipts','yps*re/maxaspenleafrealty leadville co',
 'zen babe, llc longmont co on xx 1194','town of vail parkingvail co xx24','northside bottl 7878','monument tasting roomonument co vzofjnbxywr squareup.com/receipts',
'monument tasting roomonument co vzofjnbxywr squareup.com/receipts']

#####
lists=[list_to_del,list_to_del2,list_to_del3,list_to_del4,list_to_del5,list_to_del6]
dfcopy=df2_co.copy()
for list in lists:
    for item in list:
        rowstodel=dfcopy[dfcopy.merchant==item].index
        rows_to_del_with_index(dfcopy,rowstodel)
df2_co=dfcopy


# 3d. There are 9 rows with wrong lat/long w.r.t CO. These coordinates are showing up in Ocean somewhere. 
# (Latitude < 30 and Longitude > -85 are not in CO)

# In[15]:


df2_co.sort_values(by=['longitude'], ascending=False).head(9) #9 rows have latitude of < 30 and longitude of > -85


# In[16]:


# DROPPING 9 rows that ave latitude of < 30 and longitude of > -85
rowstodel=df2_co[df2_co.longitude > -85].index #These are showing up in Caribean Ocean. 
rows_to_del_with_index(df2_co, rowstodel)
df2_co.reset_index(inplace=True, drop=True) 


# In[17]:


df2_co.reset_index(inplace=True, drop=True)
df2_co.info() #5986 #lat/long = 3143


# 3e. 3143 rows have information about lat/long. Filling these using fill_NaN_between_two_columns.

# In[18]:


fill_NaN_between_two_columns(df2_co,'city','latitude')


# In[19]:


fill_NaN_between_two_columns(df2_co,'city','longitude')


# In[20]:


df2_co.info() #5986 total lat/long = 5455


# 3f: Dropping not needed columns. 
#     1. As state will remain same for df2_co, dropping 'state'. 
#     2. 'accountid' and 'uid' are 1:1 match, therefore keeping 'uid' and dropping 'accountid'.
#     3. Dropping 'address' as we dont have all addresses available and also this field wont help with clustering as well 

# In[21]:


df2_co=df2_co[['uid','category','categoryid','merchant','city','latitude','longitude']]
df2_co.info() 


# 3g. There are 82 rows that are missing lat/long, however they have city information. Using dictionary with general coordinates of the city, replacing those missing values in lat and long columns.

# In[22]:


#df2_co.groupby('city')['latitude'].first()


# ###### These are the values used for missing lat/long for the known cities
# city: Basalt, latitude:39.3689, longitude:-107.0328
# city: breckenridge  39.4803, -106.0667
# city: Commerce City, 39.8083, -104.9339
# city: Creede, 37.8492, -106.9264
# city: Dillion, 39.6303, -106.0434
# city: Eaton, 40.5194, -104.7025
# city: Fairplay: 39.2247, -106.0020
# city: Federal Heights: 39.8599, -105.0161
# city: FoxFields: 39.5917, -104.7925
# city: Ft Collins: 40.5853, -105.0844
# city: Johnstown: 40.3369, -104.9122
# city: Yuma: 40.1222, -102.7252
# city: Usafa: 38.9984, -104.8618
# city: Trinidad: 37.1695, -104.5005
# city: South Fork: 37.6700, -106.6398
# city: Sheridan: 39.6469, -105.0253
# city: Saguache: 38.0875, -106.1420
# city: Palmer Lake: 39.1222, -104.9172
# city: Maysville: 38.5386, -106.1903
# city: Lonetree: 39.5365, -104.8971

# In[23]:


city_lat={'Basalt':39.3689, 'Breckenridge':39.4803,'Commerce City':39.8083,
         'Creede':37.8492,'Dillon':39.6303,'Eaton':40.5194,
         'Fairplay': 39.2247,
         'Federal Heights': 39.8599,
         'FoxFields': 39.5917,
          'Foxfield' : 39.5917,
         'Ft Collins': 40.5853,
         'Johnstown': 40.3369,
         'Yuma': 40.1222,
         'Usafa': 38.9984,
         'Trinidad': 37.1695, 
         'South Fork': 37.6700,
         'Sheridan': 39.6469,
         'Saguache': 38.0875,
         'Palmer Lake': 39.1222,
         'Maysville': 38.5386,
         'Lonetree': 39.5365}
dfcopy=df2_co
dfcopy['latitude']=dfcopy['city'].map(city_lat).fillna(dfcopy['latitude'])
df2_co=dfcopy


# In[24]:


city_long={'Basalt':-107.0328, 'Breckenridge':-106.0667,'Commerce City':-104.9339,
         'Creede':-106.9264,'Dillon':-106.0434,'Eaton':-104.7025,
         'Fairplay': -106.0020,
         'Federal Heights':-105.0161,
         'FoxFields': -104.7925,
         'Foxfield': -104.7925,
         'Ft Collins':-105.0844,
         'Johnstown': -104.9122,
         'Yuma':-102.7252,
         'Usafa':-104.8618,
         'Trinidad':-104.5005, 
         'South Fork':-106.6398,
         'Sheridan':-105.0253,
         'Saguache':-106.1420,
         'Palmer Lake': -104.9172,
         'Maysville': -106.1903,
         'Lonetree': -104.8971}
dfcopy=df2_co
dfcopy['longitude']=dfcopy['city'].map(city_long).fillna(dfcopy['longitude'])
df2_co=dfcopy


# 3h. There are two cities(Crossroads, San Antonio) that are not in CO and therefore those rows don't belong in df2_co.
#     1.#33 rows of San Antonio in state=CO?? Seems like a mistake.
#     2.#3 rows of 'Crossroads' city. No such town as Crossroads in CO

# In[25]:


#len(df2_co[df2_co.city=='San Antonio']) #33 rows
len(df2_co[df2_co.city=='Crossroads']) #3 rows

rowstodel=df2_co[df2_co.city=='San Antonio'].index
rows_to_del_with_index(df2_co, rowstodel)

rowstodel=df2_co[df2_co.city=='Crossroads'].index
rows_to_del_with_index(df2_co, rowstodel)

assert (len(df2_co[df2_co.city=='San Antonio'])==0) & (len(df2_co[df2_co.city=='Crossroads'])==0)


# 3i: 436 rows of no city and no lat/long info exist. Since I am dealing with very small dataset, I will fillna city with Denver and lat/long will be Denver's generic lat/long 

# In[26]:


df2_co[df2_co.city.isnull()]
#436 rows of no city and no lat/long info. Since I am dealing with very small dataset, I will fillna city with Denver and lat/long will be Denver's lat/long 


# In[27]:


#replacing NaN in latitude with 39.7392 of specific rows
#replacing NaN in longitude with-104.9903 of specific rows
#replacing NaN city with 'Denver' of specific rows

mask = (df2_co.latitude.isnull()) & (df2_co.city.isnull())
df2_co.loc[mask, 'city'] = df2_co.loc[mask, 'city'].fillna('Denver')
df2_co.loc[mask, 'latitude'] = df2_co.loc[mask, 'latitude'].fillna(39.7392)
df2_co.loc[mask, 'longitude'] = df2_co.loc[mask, 'longitude'].fillna(-104.9903)

mask=df2_co.city.isnull()
df2_co.loc[mask, 'city'] = df2_co.loc[mask, 'city'].fillna('Denver')

assert df2_co[df2_co.city.isnull()].empty #assert when cond is false


# 3j: There are 4 rows with Fort Collins named as 'Ft.Collins'. I am going to replace that with 'Fort Collins'
#     There are few rows with 'Colorado Springs' named as  'Colorado Spgs.'. Replacing that with 'Colorado Springs'

# In[28]:


mask=df2_co.city=='Ft Collins'
df2_co.loc[mask,'city'] = df2_co.loc[mask, 'city'].replace('Ft Collins','Fort Collins')

mask=df2_co.city=='Colorado Spgs.'
df2_co.loc[mask,'city'] = df2_co.loc[mask, 'city'].replace('Colorado Spgs.','Colorado Springs')


# In[29]:


df2_co.reset_index(inplace=True, drop=True)


# In[30]:


df2_co.info() #5937 rows (non null) - No NaNs in any col 


# ### DATA VISUALIZATION

# ##### MAP OF TRANSACTIONS IN CO

# In[31]:


fig=px.scatter_mapbox(df2_co, lat="latitude", lon="longitude", color="category", zoom=3,
                  hover_data= ['merchant', 'latitude', 'longitude'], mapbox_style="open-street-map")
fig.show()


# ### There are three transactions showing up outside of CO. Fixing coordinates for those merchant transactions below

# In[32]:


#correct coordinates of 'sally beauty supply' in Fort Collins
mask=df2_co.longitude==-114.04298
df2_co.loc[mask, 'longitude'] = df2_co.loc[mask, 'longitude'].replace(-114.04298,-105.1165541)


# In[33]:


#correct coordinates of 'sally beauty supply' in Longmont
mask=df2_co.longitude==-120.12291
df2_co.loc[mask, 'longitude'] = df2_co.loc[mask, 'longitude'].replace(-120.12291,-105.10193)


# In[34]:


#correct coordinates of big 5 sporting goods' in Denver
mask=df2_co.longitude==-117.293944
df2_co.loc[mask, 'longitude'] = df2_co.loc[mask, 'longitude'].replace(-117.293944,-104.940683)
df2_co.loc[mask, 'latitude'] = df2_co.loc[mask, 'latitude'].replace(34.10362,39.653013)


# In[35]:


fig=px.scatter_mapbox(df2_co, lat="latitude", lon="longitude", color="category", zoom=3,
                  hover_data= ['merchant', 'latitude', 'longitude'], mapbox_style="open-street-map")
fig.show()


# ### Focusing further on categories that have large number of transactions (>100)

# In[36]:


df2_co.groupby('category')['category'].count().sort_values(ascending=False)


# In[37]:


##trying to see if having unbalance data in category is the reason I am not getting good defined cluster. 
## Delete rows that have < 75 entries.Or Keep only

#Restaurants/Dining          1802
#Groceries                   1339
#General Merchandise          908
#Home Improvement             384
#Gasoline/Fuel                290
#Clothing/Shoes               244
#Healthcare/Medical           170
#Entertainment                166


# In[38]:


df2_co_topcat=df2_co[(df2_co.category=='Restaurants/Dining')|(df2_co.category=='Groceries')|(df2_co.category=='General Merchandise')|                    (df2_co.category=='Home Improvement')|(df2_co.category=='Gasoline/Fuel')|(df2_co.category=='Clothing/Shoes')|                     (df2_co.category=='Healthcare/Medical')]


# In[39]:


df2_co_topcat.reset_index(inplace=True, drop=True)
df2_co_topcat.info() #2371 non null rows


# In[40]:


df2_co_topcat.groupby('city')['city'].count().sort_values(ascending=False)


# #######################################################################

# In[41]:


df2_co_topcat.info() #5141 non-null rows


# In[ ]:





# In[ ]:




