#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 16:22:13 2024

@author: ls
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import sys
# def store_csv_files(filepath):
#     for file_name in os.listdir(filepath):
#         if file_name.endswith('.csv'):
#             file_path = os.path.join(filepath, file_name)

def find_variance(data):
    mean = np.mean(data)
    deviations = mean-data
    sqr_dev = deviations**2
    variance = np.sum(sqr_dev)/(len(data)-1)
    return variance

def calculate_path_complexity(x, y):
    x = np.array(x)
    y = np.array(y)

    total_path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    straight_line_distance = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
    if straight_line_distance == 0:
        return 0

    path_complexity = total_path_length / straight_line_distance

    return path_complexity


    


crux = pd.DataFrame(columns=['user','mean_acceleration','mean_jerk','vel_variance'])
# crux['user']=0
# crux['mean_acceleration']=0
# crux['mean_jerk']=0
# crux['vel_variance']=0

comp = pd.DataFrame()
filepath = "/Users/ls/Desktop/Gajra/web_bot_detection_dataset/phase1/data/mouse_movements/humans_and_moderate_bots/csv_files"
store_csv = [os.path.join(filepath,file_name) for file_name in os.listdir(filepath) if file_name.endswith('.csv')]
for i in range(0,len(store_csv)-1):
    #plt.subplot(1,1,i+1)
    data1 = pd.read_csv(store_csv[i], sep=',')
    df = pd.DataFrame(data1.mousemove_total_behaviour[0].split(']['))
    df.loc[0,0] = df.loc[0,0].replace('[','')
    df.loc[df.shape[0]-1,0] = df.loc[df.shape[0]-1,0].replace(']','')
    df[['x','y']] = df[0].str.split(',',expand=True).astype(int)
    df = df.drop(0,axis=1)
    df = pd.concat([df,pd.DataFrame(data1.mousemove_times[0].split(','))],axis=1)
    df = df.rename(columns={0:'mousemove_times'})
    df['user']= data1.session_id
    df['distance']=0
    df['velocity'] = 1
    
    for l in range(0,(df.shape[0]-1)):
        df['user'][l] = data1.session_id
        df.mousemove_times[l] = datetime.fromtimestamp(int(df.mousemove_times[l])/1000)
        # df.loc[l,'mousemove_times'] = (df.loc[0,'mousemove_times'])
    # df = df.drop([df.shape[0]-1],axis=0)
    df['mousemove_times'] = pd.to_datetime(df['mousemove_times'], errors='coerce')
    df['interval']=df['mousemove_times'].diff()
    # df['interval']=df['interval'].dt.total_seconds()
    for l in range(0,(df.shape[0]-1)):
        df['interval'][l]= df['interval'][l].total_seconds()*1000
    
    for l in range(0,(df.shape[0]-1)):
        yd=pow(df.y[l+1]-df.y[l],2)
        xd=pow(df.x[l+1]-df.x[l],2)
        df['distance'][l+1] = pow(abs(yd-xd),0.5)
    df.interval = df.interval.dropna()
    for l in range(0,(df.shape[0])):
        if df['interval'][l]!= 0:
            df['velocity'][l]=df.distance[l]/(df['interval'][l])
        else:
            df['velocity'][l] = np.inf
    for l in range(0,df.shape[0],10):
        df.loc[l:l+10,'velocity_variance']=df.loc[l:l+10,'velocity'].replace([np.inf, -np.inf], np.nan).var()
    #for l in range(0,df.shape[0],100):
    #    df.loc[l:l+100,'path_complex']=calculate_path_complexity(df.loc[l:l+100,'xpos'], df.loc[l:l+100,'ypos'])
    df['acceleration'] = df['velocity'].diff() / df['mousemove_times'].diff().dt.total_seconds()
    
    df['jerk'] = df['acceleration'].diff() / df['mousemove_times'].diff().dt.total_seconds()
    df['mousemove_times'].fillna(method='ffill', inplace=True)
    crux.loc[i,'user']=data1.session_id[0]
    crux.loc[i,'mean_acceleration']=df['acceleration'].mean()
    crux.loc[i,'mean_jerk']=df.jerk.mean()
    crux.loc[i,'vel_variance']=df.velocity.var()
    comp = pd.concat([df,comp],axis=0)
    
lala = comp
modata = comp
modata = modata.drop(['x','y','mousemove_times'],axis=1)

testcrux = pd.DataFrame()
testcomp = pd.DataFrame()
filepath = "/Users/ls/Desktop/Gajra/test_csv_data"
test_store_csv = [os.path.join(filepath,file_name) for file_name in os.listdir(filepath) if file_name.endswith('.csv')]
for i in range(0,len(test_store_csv)-1):
    #plt.subplot(1,1,i+1)
    df = pd.read_csv(test_store_csv[i], sep=',')
    df['user'] = i
    df['distance']=0
    df['velocity'] = 1
    df['time']=df['timestamp']
    for l in range(0,(df.shape[0])):
        df.timestamp[l] = datetime.fromtimestamp(df.timestamp[l]/1000)
    df['interval']=df['timestamp'].diff()
    #df['interval']=df['interval'].dt.total_seconds()
    for l in range(0,(df.shape[0])):
        df['interval'][l]= df['interval'][l].total_seconds()*1000

    for l in range(0,(df.shape[0]-1)):
        yd=pow(df.y[l+1]-df.y[l],2)
        xd=pow(df.x[l+1]-df.x[l],2)
        df['distance'][l+1] = pow(abs(yd-xd),0.5)
    df.interval = df.interval.dropna()
    for l in range(0,(df.shape[0])):
        if df['interval'][l]!= 0:
            df['velocity'][l]=df.distance[l]/(df['interval'][l])
        else:
            df['velocity'][l] = np.inf
    for l in range(0,df.shape[0],10):
        df.loc[l:l+10,'velocity_variance']=df.loc[l:l+10,'velocity'].replace([np.inf, -np.inf], np.nan).var()
    #for l in range(0,df.shape[0],100):
    #    df.loc[l:l+100,'path_complex']=calculate_path_complexity(df.loc[l:l+100,'xpos'], df.loc[l:l+100,'ypos'])

    df['acceleration'] = df['velocity'].diff() / df['timestamp'].diff().dt.total_seconds()

    df['jerk'] = df['acceleration'].diff() / df['timestamp'].diff().dt.total_seconds()
    testcrux.loc[i,'mean_acceleration']=df['acceleration'].mean()
    testcrux.loc[i,'mean_jerk']=df.jerk.mean()
    testcrux.loc[i,'vel_variance']=df.velocity.var()

    testcomp = pd.concat([df,testcomp],axis=0)
    
testcrux = testcrux.fillna(0)

# #modata = pd.get_dummies(modata, columns = ['event'], drop_first = True)

# modata = modata.replace([np.inf], sys.maxsize)
# modata = modata.replace([-np.inf], -sys.maxsize)
# modata = modata.fillna(0)
# copy1 = modata
# s = StandardScaler()
# #from sklearn.model_selection import StratifiedShuffleSplit
# #strat_shuf_split = StratifiedShuffleSplit()(n_splits=1, test_size=0.3, random_state=42)
# #train_idx, test_idx = next(start_shuf_split.split())


# #modata = modata.drop('timestamp',axis=1)
# #modata.columns
# #modata = pd.get_dummies(modata, columns = ['event'], drop_first = True)
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid
   
from sklearn.metrics import make_scorer, silhouette_score
import numpy as np
# param = {'n_clusters':[2,3,4,5]}
# results=[]
# for p in ParameterGrid(param):
#     kmeans = KMeans(n_clusters=p['n_clusters'],random_state=0)
#     kmeans.fit(modata)
    
#     labels = kmeans.labels_
#     silhouette_avg = silhouette_score(modata,labels)
#     results.append({
#         'params': params,
#         'silhouette_score': silhouette_avg,
#         'inertia': kmeans.inertia_  # Sum of squared distances of samples to their closest cluster center
#     })
 

# Define a custom scoring function for clustering
def silhouette_scorer(estimator, X):
    cluster_labels = estimator.fit_predict(X)
    return silhouette_score(X, cluster_labels)
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps = 0.5, min_samples = 5)
crux = crux.fillna(0)
labels = dbscan.fit_predict(crux.drop('user',axis=1))
# param = {'eps':[0.3,5,10], 'min_samples':[5]}
# grid = GridSearchCV(dbscan, param, scoring=silhouette_score)
# dbscan.fit((modata))
