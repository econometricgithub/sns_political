#importing necessary library
import numpy as np
import pandas as pd
import csv
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template,flash, request
from wtforms import Form, StringField, validators, SubmitField, SelectField
from bioinfokit.analys import get_data, stat
#Importing SQLAlcheny
from flask_sqlalchemy import SQLAlchemy
#from custom_validators import height_validator, weight_validator
from myModules import results_summary_to_dataframe, result_one,results_two, main_fun, reg_metric, linerity,normality,homoscedasticity,multicollinearity
#input are just variables - not as in the short form [ Y, X ]

from sqlalchemy import create_engine
import pymysql
desired_width=800
pd.set_option('display.width',desired_width)
pd.set_option('display.max_columns', 40)

#db_connection_str = 'mysql+pymysql://b5f3d54f1ae019:2577be25@us-cdbr-east-05.cleardb.net/heroku_8b3d35cf0aa7c8f'
#db_connection = create_engine(db_connection_str)
'''
#df = pd.read_sql('SELECT * FROM answers', con=db_connection)
df=pd.read_csv("Dataset/raw_data.csv")

df["d_informational_use"]=0
df["d_communication"]=0
df["d_new_friends"]=0
df["d_share"]=0
df["d_study"]=0
df["d_entertainment"]=0
a=0
b=0
c=0
d=0
e=0
f=0

for i in df['reasoning_using']:
    str =list(i.split(","))
    arr=np.array(str)
    #convert string array to int array
    new_arr=list(map(int, arr))
    #print(new_arr)



   

    new_list=[1 if i in new_arr else 0 for i in range(1, 7)]
    print(new_list[0])
    df['a']=new_list[0]
    df['b'] = new_list[1]
    df['c'] = new_list[2]
    df['d'] = new_list[3]
    df['e'] = new_list[4]
    df['f'] = new_list[5]
    print("####################")
print(df)



path_csv = "Dataset/MyCsv.csv"
with open(path_csv, mode='a', newline='') as f:
    fields = ['Name', 'Branch', 'Year', 'CGPA', 'dd', 'cc']
    csv_writer = csv.writer(f)
    csv_writer.writerow(fields)
fields = ['Name', 'Branch', 'Year', 'CGPA','dd','cc']
for j in list_2:
    with open(path_csv, mode='a', newline='') as f:
        csv_writer = csv.writer(f)

        csv_writer.writerow(j)

idv=['aa','bb','cc']
dv='dd'
idv.append(dv)
#print(idv)

#Data Summary
df=pd.read_csv("Dataset/data_cleaning.csv")
dv=['OEP','OFEP','political_efficacy']
df_dv=df[dv]
result=df_dv.describe()
output=result.to_html()
print(output)

threeUsage=['informational_use','entertainment_use','SIU']
dv_2=df[threeUsage]
#print(dv_2.describe())

dv_3=df['three_usage']
#print(dv_3.describe())

age=df['age']
#print(age.describe())


def label_function(val):
    return f'{val / 100 * len(df):.0f}\n{val:.0f}%'

fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5))

df.groupby('gender').size().plot(kind='pie',autopct=label_function,textprops={'fontsize': 20},
                                  colors=['r', 'b'], ax=ax1)'''
#plt.show()

"""#matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
df3=pd.read_csv("Dataset/data_cleaning.csv")
carrier_count = df3['location'].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
positions = (0,1,2,3,4,5,6,7,8,9,10,11,12,13)
labels = ('Kachin','Kayah','Kayin','Chin','Mon','Shan','Sagaing','Magway','Mandalay','Bagu','Tanintharyi','Yangon','Ayeyarwady','Overseas')
plt.xticks(positions, labels, rotation='vertical')
plt.title('Frequency Distribution of Location')
plt.ylabel('Number of Occurrences', fontsize=6)
plt.xlabel('Location',fontsize=6)
# Pad margins so that markers don't get clipped by the axes
plt.margins(0.01)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.25)
plt.savefig("static/location_1.png",dpi=72)

# for visualize for coefficients.
def visualize_coefficients(coefs, names, ax):
    # Make a bar plot for the coefficients, including their names on the x-axis
    ax.bar(____, ____)
    ax.set(xlabel='Coefficient name', ylabel='Coefficient value')

    # Set formatting so it looks nice
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    return ax  """

'''# 3D plot for two variables
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
df=pd.read_csv("Dataset/data_cleaning.csv")
b = df['OEP']
d = df['political_efficacy']

B, D = np.meshgrid(b, d)
nu = np.sqrt( 1 + (2*D*B)**2 ) / np.sqrt( (1-B**2)**2 + (2*D*B)**2)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(B, D, nu)
plt.xlabel('b')
plt.ylabel('d')
plt.show()

plt.contourf(B, D, nu)
plt.colorbar()
plt.xlabel('b')
plt.ylabel('d')
#plt.show()


import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
df = pd.read_csv("Dataset/data_cleaning.csv")
b = df['OEP']
d = df['political_efficacy']
B, D = np.meshgrid(b, d)
nu = np.sqrt(1 + (2 * D * B) ** 2) / np.sqrt((1 - B ** 2) ** 2 + (2 * D * B) ** 2)
fig= plt.figure()
ax = Axes3D(fig)
ax.plot_surface(B, D, nu)
plt.xlabel('b')
plt.ylabel('d')
#plt.savefig("static/exam.png")
#plt.show()

import plotly.express as px
fig = px.density_contour(df, x="three_usage", y="OEP")
fig.update_traces(contours_coloring="fill", contours_showlabels = True)
fig=fig.show()
#print(fig)

idv=['informational_use','SIU','entertainment_use','age','education','income','location']
a=len(idv)
#print('length')
#print(a)


# Creating Income Plot
import seaborn as sns
import matplotlib.pyplot as plt
df2=pd.read_csv("Dataset/data_cleaning.csv")
carrier_count = df2['education'].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
positions = (0,1,2,3,4,5,6)
labels = ('Primary School','Middle School','High School','Bachelor Degree','Master Degree','PhD Degree','Others')
plt.xticks(positions, labels, rotation='vertical')
plt.title('Education')
plt.ylabel('Number of Occurrences', fontsize=6)
plt.xlabel('Education',fontsize=6)
# Pad margins so that markers don't get clipped by the axes
plt.margins(0.01)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.3)
#plt.savefig("static/education.png",dpi=72) '''

'''# Creating Age
import seaborn as sns
import matplotlib.pyplot as plt
df2=pd.read_csv("Dataset/data_cleaning.csv")
carrier_count = df2['age'].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
positions = (0,1,2,3,4,5,6)
labels = ('Primary School','Middle School','High School','Bachelor Degree','Master Degree','PhD Degree','Others')
#plt.xticks(positions, labels, rotation='vertical')
plt.title('Age')
plt.ylabel('Number of Occurrences', fontsize=6)
plt.xlabel('Age',fontsize=6)
# Pad margins so that markers don't get clipped by the axes
plt.margins(0.01)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.3)
plt.savefig("static/age.png",dpi=72)


import seaborn as sns
import matplotlib.pyplot as plt
df2=pd.read_csv("Dataset/data_cleaning.csv")
carrier_count = df2['income'].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
positions = (0,1,2,3,4)
labels = ('Primary School','Middle School','High School','Bachelor Degree','Master Degree')
#plt.xticks(positions, labels, rotation='vertical')
plt.title('Income')
plt.ylabel('Number of Occurrences', fontsize=6)
plt.xlabel('Income',fontsize=6)
# Pad margins so that markers don't get clipped by the axes
plt.margins(0.01)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.3)
plt.show() 
dv=['aa','bb','cc','dd']
if 'aa' and 'bb' and 'ff' in dv:
    print('Ture')
else:
    print('False') 


df3=pd.read_csv("Dataset/new_data.csv")
X=df3[['informational_use','SIU','entertainment_use','age','education','income','location']]
y=df3['OFEP']
X = sm.add_constant(X)
mlr_results = sm.OLS(y, X).fit()
df = pd.read_html(mlr_results.summary().tables[1].as_html(),header=0,index_col=0)[0]
print(df)
a_1 = df['coef'].values[0]
a_2 = df['coef'].values[1]
a_3 = df['coef'].values[2]
a_4 = df['coef'].values[3]

d_1 = df['std err'].values[0]
d_2 = df['std err'].values[1]
d_3 = df['std err'].values[2]
d_4 = df['std err'].values[3]

f_1 = df['P>|t|'].values[0]
f_2 = df['P>|t|'].values[1]
f_3 = df['P>|t|'].values[2]
f_4 = df['P>|t|'].values[3]

print(a_1)
print(a_2)
print(a_3)
print(a_4)
print(d_1)
print(d_2)
print(d_3)
print(d_4)

print(f_1)
print(f_2)
print(f_3)
print(f_4)
#df=pd.read_csv("Dataset/new_data_2.csv")
#print(df.isna().sum())
#print(df.describe())
#df = df.dropna()
#df.to_csv("Dataset/new_data.csv")

df=pd.read_csv("Dataset/new_data.csv")
print(df.isna().sum())
print(df.describe()) 

dv=['aa','bb','cc','dd']
idv='OEP'

if 'OEP' in idv:
    if 'aa' and 'bb' and 'ff' in dv:
        print('Ture')
    else:
        print('False')


import pandas as pd
from sklearn import datasets
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
from IPython.core.display import HTML


df=pd.read_csv("Dataset/new_data.csv")
X= df['threeusage', 'gender', 'age']
y=df['OEP']

est = sm.OLS(endog=y, exog=sm.add_constant(X).fit())


stargazer = Stargazer([est])

HTML(stargazer.render_html()) 

import pandas as pd
from sklearn import datasets
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
from IPython.core.display import HTML

diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data)
#print(df)
df.columns = ['Age', 'Sex', 'BMI', 'ABP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
df['target'] = diabetes.target

est = sm.OLS(endog=df['target'], exog=sm.add_constant(df[df.columns[0:4]])).fit()
est2 = sm.OLS(endog=df['target'], exog=sm.add_constant(df[df.columns[0:6]])).fit()

stargazer = Stargazer([est, est2])

#print(stargazer.render_latex(
HTML(stargazer.render_html())

#with open('reg_table.html', 'w') as f:
 #   f.write(stargazer.render_html())
from statsmodels.iolib.summary2 import summary_col

df = pd.read_csv("Dataset/new_data.csv")
X = df[['three_usage','age','gender']]
y = df['OEP']
X = sm.add_constant(X)
mlr_results = sm.OLS(y, X).fit()
aa=summary_col([mlr_results],stars=True,float_format='%0.2f',
                  model_names=['p4\n(0)','p4\n(1)','p4\n(2)'],
                  info_dict={'N':lambda x: "{0:d}".format(int(x.nobs))}) '''


import pandas as pd
from sklearn import datasets
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
from IPython.core.display import HTML
df = pd.read_csv("Dataset/new_data.csv")
X = df[['political_efficacy','education','income','location']]
y = df['OEP']
X = sm.add_constant(X)
mlr_results1 = sm.OLS(y, X).fit()

X = df[['political_efficacy','gender','age','education','income','location']]
y = df['OFEP']
X = sm.add_constant(X)
mlr_results2 = sm.OLS(y, X).fit()

stargazer = Stargazer([mlr_results1, mlr_results2])
with open('political_efficacy-to_ofe&ofep.doc', 'w') as f:
    f.write(stargazer.render_html())

''' import seaborn as sns
df_pie=pd.read_csv("Dataset/new_data.csv")
corr = df_pie[['OFEP', 'three_usage', 'gender','age','education','income',location']].corr()
print('Pearson correlation coefficient matrix of each variables:\n', corr)

# Generate a mask for the diagonal cell
mask = np.zeros_like(corr, dtype=np.bool)
np.fill_diagonal(mask, val=True)

# Initialize matplotlib figure
fig, ax = plt.subplots(figsize=(4, 3))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True, sep=100)
cmap.set_bad('grey')

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=.5)
fig.suptitle('Pearson correlation coefficient matrix', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=10)
fig.tight_layout() '''




#Data Normalizaation
data=pd.read_csv('Dataset/example.csv')

#print(data.head())

print(data['political_efficacy'].max())
print(data['political_efficacy'].min())

data['political_efficacy']=data['political_efficacy']/data['political_efficacy'].max()
#print(data.head())

#pd.get_dummies(data,columns=['gender'])

'''bins=np.linspace(min(data['age']),max(data['age']),4)
print(bins)
bin_name=['young','medum','older']
data['age_group']=pd.cut(data['age'],bins,labels=bin_name,include_lowest=True)
print(data.head())

data.to_csv('Dataset/cleaning.csv')'''
# age grounp
import seaborn as sns
import matplotlib.pyplot as plt
df2=pd.read_csv("Dataset/cleaning.csv")
carrier_count = df2['age_group'].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
positions = (0,1,2,3,4,5,6)
labels = ('Primary School','Middle School','High School','Bachelor Degree','Master Degree','PhD Degree','Others')
#plt.xticks(positions, labels, rotation='vertical')
plt.title('Age')
plt.ylabel('Number of Occurrences', fontsize=6)
plt.xlabel('Age',fontsize=6)
# Pad margins so that markers don't get clipped by the axes
plt.margins(0.01)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.3)
plt.savefig("static/age_2.png",dpi=72)