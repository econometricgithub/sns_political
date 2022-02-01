#matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('seaborn-white')
import numpy as np
df=pd.read_csv('Dataset/data_cleaning.csv')
def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

x = df['political_efficacy']
y =df['OEP']

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

#plt.contour(X, Y, Z, 20, cmap='RdGy');
#plt.show()

import seaborn as sns
import matplotlib.pyplot as plt



###########################################################
'''import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("Dataset/data_cleaning.csv")
df=df[['OEP','OFEP','political_efficacy','three_usage','informational_use','SIU','entertainment_use','gender','age','education','income','location']]
correlation_matrix = df.corr().round(2)
plt.figure(figsize=(14, 12))
sns.heatmap(data=correlation_matrix, annot=True)
plt.savefig('static/correlation')'''

#Coefficients plots
import statsmodels.api as sm
df = pd.read_csv("Dataset/data_cleaning.csv")
idv=df[['informational_use','SIU','entertainment_use','gender','age','education','income','location']]
dv=df['OEP']
X = sm.add_constant(idv)
y = dv
X = sm.add_constant(X)
mlr_results = sm.OLS(y, X).fit()
# Extract the coefficients from the model
#df_coef = mlr_results.params.to_frame().rename(columns={0: 'coef'})
# Visualize the coefficients
#ax = df_coef.plot.barh(figsize=(14, 7))
#ax.axvline(0, color='black', lw=1)
#plt.show()

from scipy import stats
sns.distplot(mlr_results.resid, fit=stats.norm)
plt.show()

#import seaborn as sns
#sns.histplot(mlr_results.resid)
#plt.show()

#sm.graphics.plot_fit(mlr_results,1, vlines=False)
#plt.show()


#sns.boxplot(x=mlr_results.resid, showmeans=True)
#plt.show()

