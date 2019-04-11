
"""
Original code by Rob van Putten
Changes by Nynke ter Heide:
    - only final ML-model, stripped from intermediate steps
    - addition of GEF name to df
    - implementation of train test split (randomly 1 gef picked as test set)
    - feature engineering: adding average of previous and proceeding rows
    - save noise that was filtered out in separate df
    - add noise again at the end with nan as result
    - export of results to a csv-file
    - SGD classifier replaced by decision tree algorithm
"""

import sys, os, math
import pandas as pd
sys.path.append("C:\\Anaconda3\\Lib\\site-packages")
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from bbgeolib.objects.gef import get_from_file
from bbgeolib.tools.gef import get_unit_weight_from_cpt, get_soilstress_from_cpt
import numpy as np

files = [
    'gefs\\E06-1441.GEF',
    'gefs\\E06-1443.GEF',
    'gefs\\E06-1444.GEF',
    'gefs\\E06-1445.GEF',
    'gefs\\E07-1107.GEF',
    'gefs\\E07-1109.GEF',
    'gefs\\E07-1110.GEF',
    'gefs\\E07-1112.GEF',
    'gefs\\E07-1113.GEF'
] 

soils = {
        'E06-1441':'99,OB,-0.6,KS,-4.0,HV,-6.0,KS,-6.7,WZ, -8.8,KM,-10.5,BV,-10.6,PZ,-13.3,KM,-14.3,PZ',
        'E06-1443':'99,OB,-0.9,KS,-4.0,HV,-6.2,KS,-6.9,WZ, -9.3,KM,-11.1,BV,-11.4,PZ,-13.7,KM,-15.1,PZ',
        'E06-1444':'99,OB,-1.0,KS,-3.9,HV,-5.9,KS,-6.2,WZ,-10.4,KM,-11.2,BV,-11.9,PZ,-14.2,KM',
        'E06-1445':'99,OB,-1.0,KS,-4.1,HV,-6.0,KS,-6.2,WZ,-10.5,KM,-11.2,BV,-12.5,PZ,-14.6,KM,-15.7,PZ',
        'E07-1107':'99,OB,-0.6,KS,-3.2,HV,-6.2,KS,-6.9,WZ, -9.8,KM,-11.6,BV,-12.3,PZ,-14.7,KM',
        'E07-1109':'99,OB,-0.4,KS,-2.7,HV,-5.8,KS,-6.1,WZ,-10.2,KM,-12.0,BV,-12.4,PZ,-14.7,KM,-15.7,PZ',
        'E07-1110':'99,OB,-0.5,KS,-3.5,HV,-6.4,KS,-6.8,WZ, -9.9,KM,-11.2,BV,-11.6,KM,-17.7,PZ',
        'E07-1112':'99,OB,-0.9,KS,-3.6,HV,-6.4,KS,-6.8,WZ, -9.9,KM,-10.9,BV,-11.2,KM,-12,BV,-12.6,PZ,-14.6,KM,-17.2,PZ',
        'E07-1113':'99,OB,-0.9,KS,-4.2,HV,-6.2,KS,-6.4,WZ, -9.8,KM,-10.5,BV,-10.8,KM,-11.9,BV,-12.3,PZ,-14.6,KM,-17,PZ'
}

# functon to extract data from gef files
def create_input_with_soilstress():
    dfs = []    
    for file in files:
        gef = get_from_file(file)
        name = os.path.basename(gef.filename).split('.')[0]
        df = gef.as_dataframe()
        df = get_soilstress_from_cpt(gef)
        df['soilname'] = 'unknown'
        args = soils[name].split(',')
        soilnames = args[1::2]
        tops = [float(a) for a in args[0::2]]
        bottoms = tops[1:]
        bottoms.append(-99)
        for soilname, top, bottom in zip(soilnames,tops,bottoms):  
            df.loc[(df['depth']<=top) & (df['depth']>=bottom), 'soilname']=soilname
        df = df[['depth', 'qc','fs', 'sv', 'soilname']]
        n = 100 # number of rows for averaging
        df['qc_average_before'] = df['qc'].rolling(n, min_periods=1).mean()
        df['qc_average_after'] = df.sort_values(['depth'], ascending = True)['qc'].rolling(n, min_periods=1).mean()
        df['fs_average_before'] = df['fs'].rolling(n, min_periods=1).mean()
        df['fs_average_after'] = df.sort_values(['depth'], ascending = True)['fs'].rolling(n, min_periods=1).mean()
        df['GEFname'] = name
        dfs.append(df)
        
    return pd.concat(dfs) 

# extract data from gef files
df3 = create_input_with_soilstress()

# pick 1 GEF random as validation set
import random
GEFnames = list(df3['GEFname'].unique())
validationGEF = random.choice(GEFnames)
print("validation GEF: ", validationGEF)

# transform features qc and fs to log scale
df3['qc'] = df3['qc'].apply(lambda x: math.log(x))
df3['fs'] = df3['fs'].apply(lambda x: math.log(x))
df3['qc_average_before'] = df3['qc_average_before'].apply(lambda x: math.log(x))
df3['qc_average_after'] = df3['qc_average_after'].apply(lambda x: math.log(x))
df3['fs_average_before'] = df3['fs_average_before'].apply(lambda x: math.log(x))
df3['fs_average_after'] = df3['fs_average_after'].apply(lambda x: math.log(x))

# make soil classes numeric
le = LabelEncoder()
df3['class'] = le.fit_transform(df3['soilname'])
soillabels = le.inverse_transform([0,1,2,3,4,5,6])

# split in validation and train/test set
df_test = df3[df3['GEFname'] == validationGEF]
df_train = df3[df3['GEFname'] != validationGEF]

# split in features and target
X_test = df_test[['qc', 'fs', 'sv', 'qc_average_before', 'qc_average_after', 'fs_average_before', 'fs_average_after']].values
y_test = df_test['class'].values
X_train = df_train[['qc', 'fs', 'sv', 'qc_average_before', 'qc_average_after', 'fs_average_before', 'fs_average_after']].values
y_train = df_train['class'].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

# remove noise from y
y_train = y_train[X_train[:,1]>0.1]
y_test = y_test[X_test[:,1]>0.1]
# save noise in seperate
df_noise_train = df_train[X_train[:,1]<=0.1]
df_noise_test = df_test[X_test[:,1]<=0.1]
# remove noise from df_train and df_test
df_train = df_train[X_train[:,1]>0.1]
df_test = df_test[X_test[:,1]>0.1]
# remove noise from X
X_train = X_train[X_train[:,1]>0.1,:]
X_test = X_test[X_test[:,1]>0.1,:]
# set result for noise to nan
df_noise_test['dt']= math.nan
df_noise_train['dt']= math.nan 

# train decision tree classifier with train set
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth = 15)
df_train['dt'] = clf.fit(X_train,y_train).predict(X_train)
print("Accuracy DecisionTree train set:", accuracy_score(y_train, df_train['dt']))

# apply trained model to test set
df_test['dt'] = clf.predict(X_test)
print("Accuracy DecisionTree test set:", accuracy_score(y_test, df_test['dt']))

# to df and add labels for clarity and plotting
fi = pd.DataFrame(clf.feature_importances_, columns =['feature_importance'])
fi['features'] = ['qc', 'fs', 'sv', 'qc_average_before', 'qc_average_after', 'fs_average_before', 'fs_average_after']
fi = fi.sort_values(by = ['feature_importance'], axis=0, ascending = False)

# plot the coefficients
import seaborn as sns
sns.set(style='whitegrid')
_ = sns.barplot(y='features', x = 'feature_importance', data = fi, color = 'blue')
_ = plt.legend()
_ = plt.tight_layout()
_ = plt.xlabel('feature importance')
_ = plt.ylabel('features')
_ = plt.xlim([0,0.5])
plt.savefig("feature_importance_tree.jpg", dpi = 300)

# translate class back to soilname
df_train['pred_soilname'] = le.inverse_transform(df_train['dt'])
df_test['pred_soilname'] = le.inverse_transform(df_test['dt'])

# add label to df's
df_train['traintestnoise']='train'
df_test['traintestnoise']='test'
df_noise_train['traintestnoise']='train_noise'
df_noise_test['traintestnoise']='test_noise'

# Bring train data, test data and noise back together
df_traintestnoise = pd.concat([df_train, df_noise_train, df_test, df_noise_test], ignore_index=True)

# expert results to csv file
df_traintestnoise.to_csv("output_python_geoAI_tree.csv", index = False)

# export viz of tree
data_feature_names = ['qc', 'fs', 'sv', 'qc_average_before', 'qc_average_after', 'fs_average_before', 'fs_average_after']
tree.export_graphviz(clf,out_file='geo_AI_tree_viz.dot', feature_names=data_feature_names)
