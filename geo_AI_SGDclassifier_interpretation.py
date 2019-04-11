
"""
Original code by Rob van Putten
Changes by Nynke ter Heide:
    - only final ML-model, stripped from intermediate steps
    - addition of GEF name to df
    - print and plot feature coefficients
"""

import sys, os, math
import pandas as pd
sys.path.append("C:\\Anaconda3\\Lib\\site-packages")
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
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
        df['GEFname'] = name
        dfs.append(df)
        
    return pd.concat(dfs) 

# extract data from gef files
df3 = create_input_with_soilstress()

# transform features qc and fs to log scale
df3['qc'] = df3['qc'].apply(lambda x: math.log(x))
df3['fs'] = df3['fs'].apply(lambda x: math.log(x))

# normalise the features to a value between 0 and 1 (min = 0, max = 1)
df3['qcn']=(df3['qc']-df3['qc'].min())/(df3['qc'].max()-df3['qc'].min())
df3['fsn']=(df3['fs']-df3['fs'].min())/(df3['fs'].max()-df3['fs'].min())
df3['svn']=(df3['sv']-df3['sv'].min())/(df3['sv'].max()-df3['sv'].min())

# remove noise
df3 = df3[df3['fsn']>0.1]

# make soil classes numeric
le = LabelEncoder()
df3['class'] = le.fit_transform(df3['soilname'])
soillabels = le.inverse_transform([0,1,2,3,4,5,6])

# split in features (X) and target (y)
X = df3[['qcn', 'fsn', 'svn']].values
y = df3['class'].values

clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000, tol=1e-4, class_weight= 'balanced', shuffle = True)
df3['sgdclassifier'] = clf.fit(X,y).predict(X)
print("Accuracy SGDClassifier:", accuracy_score(y, df3['sgdclassifier']))

# print the coefficients of the features
print(clf.coef_)

# to df and add labels for clarity and plotting
coefs = pd.DataFrame(clf.coef_, columns = ['feature_coef_1', 'feature_coef_2', 'feature_coef_3'])
coefs['soilnames'] = soillabels
coefs = pd.wide_to_long(coefs, stubnames = 'feature_coef_', i='soilnames', j = 'feature')
coefs = coefs.reset_index()
coefs['feature_name'] = np.select(condlist = [(coefs['feature'] == 1), (coefs['feature'] == 2), (coefs['feature'] == 3)], 
     choicelist = ['qcn', 'fsn', 'svn'], default='empty')

# plot the coefficients
import seaborn as sns
sns.set(style='whitegrid')
_ = sns.barplot(x='soilnames', y = 'feature_coef_', hue = 'feature_name', data = coefs)
_ = plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
_ = plt.tight_layout()
_ = plt.ylabel('feature coefficient')
_ = plt.xlabel('soil class')
plt.savefig("feature_coefficient_SGDclassifier.jpg", dpi = 300)


