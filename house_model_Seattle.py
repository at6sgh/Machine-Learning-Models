# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 16:48:57 2019

@author: Agaat Traczyńska
"""

# %%
#zadanie 1

import pandas as pd
url ='C:/Users/gagat/Documents/Lingaro/house.csv'
df = pd.read_csv(url, sep=',', na_values='')

# %%
df.head()
df.info()
df.describe()


# %%
# Dodaje nową zmienną Distanece która jest wyliczona odległocią wkm od centrum Miasta Seattle na podstawie wspłrzędnych domu

# Współrzędne Centrum - Seattle Pike Place Market
latCentrum= 47.610136
longCentrum = -122.342057


Centrum = (latCentrum, longCentrum)

# tworzą dataframe z long i lat na potrzeby policzenia odleglosci
latitude = df['lat'].values
longitude = df['long'].values
data = {'latitude': latitude, 'longitude': longitude}
location = pd.DataFrame(data)
location.head()

#importujemy pakiet liczacy odleglosc od centrum
from geopy.distance import geodesic

#liczymy odleglosc od centrum
distance = []
for i in range(21613):
    distance.append(geodesic(location.iloc[i,:], Centrum).kilometers)
    
df['distance'] = distance
df.describe()

# %%
#Sprawdzamy liczebnoć i czy nie ma jaki braków danych
df.yr_built.value_counts(dropna=False)
df.condition.value_counts(dropna=False)
df.view.value_counts(dropna=False)
df.bedrooms.value_counts(dropna=False)
df.bathrooms.value_counts(dropna=False)
df.waterfront.value_counts(dropna=False)
df.grade.value_counts(dropna=False)
df.floors.value_counts(dropna=False)
# %%
#Sprawdzam normalnoc rozkładu poszczególnych zmiennych ilosciowych
import matplotlib.pyplot as plt
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(10,8))
df.plot(y='distance', kind='hist', bins =30, range=(0,80), ax=ax)
plt.xlabel('Distance')
plt.ylabel('Density')
plt.title('Histogram of Distance Distribution')
plt.savefig('C:/Users/gagat/Documents/Lingaro/histDistance.png')
plt.show()
#normalny

fig, ax = plt.subplots(figsize=(10,8))
df.plot(y='condition', kind='hist', bins =6, range=(0,6), ax=ax)
plt.xlabel('Condition')
plt.ylabel('Density')
plt.title('Histogram of Condition Distribution')
plt.savefig('C:/Users/gagat/Documents/Lingaro/histCondition.png')
plt.show()
#brak normalnego rozkładu
#Sprawdzam czy pomoże logarytm ze zmiennej, ale nie polepsza rozkładu
fig, ax = plt.subplots(figsize=(10,8))
df.plot(y='condition', kind='hist', bins =6, range=(0,6), ax=ax, logx=True)

fig, ax = plt.subplots(figsize=(10,8))
df.plot(y='view', kind='hist', bins =6, range=(0,6), ax=ax)
#Zmieniamy na zmienną binarną 1-ładny widok z okien, 0-brak ładnego widoku

fig, ax = plt.subplots(figsize=(10,8))
df.plot(y='waterfront', kind='hist', bins =2, range=(0,1), ax=ax)
#usówamy tą zmienną gdyż za mało obserwacji ma wartoć 1, tylko 0.009%, a powinno być minimum 10%
# oraz zawiera się ona w zmiennej view, czyli ładny widok z okna

fig, ax = plt.subplots(figsize=(10,8))
df.plot(y='grade', kind='hist', bins =13, range=(0,13), ax=ax)
#normalny

#Bedrooms
fig, ax = plt.subplots(figsize=(10,8))
df.plot(y='bedrooms', kind='hist', bins =11, range=(0,11), ax=ax)
#normalny

#Bathrooms
fig, ax = plt.subplots(figsize=(10,8))
df.plot(y='bathrooms', kind='hist', bins =32, range=(0,8), ax=ax)
#skosny prawostronny

#Floors
fig, ax = plt.subplots(figsize=(10,8))
df.plot(y='floors', kind='hist', bins =6, range=(1,3.5), ax=ax)
#brak normalnego rozkładu
#Sprawdzam czy pomoże logarytm ze zmiennej, ale nie polepsza rozkładu
fig, ax = plt.subplots(figsize=(10,8))
df.plot(y='floors', kind='hist', bins =6, range=(1,3.5), ax=ax, logx=True)

#Price 
fig, ax = plt.subplots(figsize=(10,8))
df.plot(y='price', kind='hist', bins =50, range=(0,2000000), ax=ax)
#normalny

#Sqft living
fig, ax = plt.subplots(figsize=(10,8))
df.plot(y='sqft_living', kind='hist', bins =100, range=(0,8000), ax=ax)
#przybliżony normalny

#sqft_lot
fig, ax = plt.subplots(figsize=(10,8))
df.plot(y='sqft_lot', kind='hist', bins =50, range=(0,8000), ax=ax)
#brak normalnego rozkładu

#sqft_above
fig, ax = plt.subplots(figsize=(10,8))
df.plot(y='sqft_above', kind='hist', bins =50, range=(0,6000), ax=ax)
#normalny rozkładu

#sqft_basement
fig, ax = plt.subplots(figsize=(10,8))
df.plot(y='sqft_basement', kind='hist', bins =50, range=(0,2000), ax=ax)
#brak normalnego rozkładu

# %%
#wybieram zmienne do analizy
cols = ['price', 'bedrooms','bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'view', 'condition', 'grade', 'sqft_above', 'sqft_basement','yr_built','distance']
df2=df[cols] 

# Poprawiam kilka zmiennych
# year built z daty na wiek domu
df2.yr_built = 2015 - df.yr_built
# zmienną view zmieniam na binarną: ładny widok z okien i jego brak
df2.view.value_counts(dropna=False)
df2.view[df['view'] == 2] =1
df2.view[df['view'] == 3] =1
df2.view[df['view'] == 4] =1
df2.view[df['waterfront'] ==1] =1

df2.describe()
# %%


# %%
#identyfikacja obserwacji odstających
import seaborn as sns
import matplotlib.pyplot as plt

g1 = sns.lmplot(x='price', y='distance', data=df2, palette='Set1')
g1.set_xticklabels(rotation=30)
plt.title('Scatter plot of Price and Distance')
plt.savefig('C:/Users/gagat/Documents/Lingaro/Price_Dist_scatter.png')
plt.show()


g2=sns.lmplot(x='price', y='sqft_living', data=df2, palette='Set1')
g2.set_xticklabels(rotation=30)
plt.title('Scatter plot of Price and Living Space Area')
plt.savefig('C:/Users/gagat/Documents/Lingaro/Price_Sqft_scatter.png')
plt.show()

# usuwamy obsewacje odstajace ze zbyt wysoką ceną
df2 = df2[df2.price<4000000]
# usuwamy obsewacje ze zbyt dużą powierzchnią sqrt feet area > 10000
df2 = df2[df2.sqft_living<10000]


fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df2.corr(), square=True, cmap='RdYlGn')
plt.title('Heatmap of Correlation')
plt.savefig('C:/Users/gagat/Documents/Lingaro/Heatmap.png')
plt.show()

# %%
#Lasso Regression
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


X= df2.drop(['price'], axis=1).values

y= df2['price'].values
y=y.reshape(-1,1)


# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

df_columns = df.drop(['id', 'date', 'price', 'waterfront','yr_renovated', 'zipcode', 'lat', 
              'long', 'sqft_living15', 'sqft_lot15'], axis=1).columns
# Plot the coefficients
fig, ax = plt.subplots(figsize=(10,6))
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.title('Lasso Regression Feature Sellection')
plt.savefig('C:/Users/gagat/Documents/Lingaro/LassoFetureSellection5.png')
plt.show()

#Wybieramy zmienne do modelu regresji linowej za pomocą regresji Lasso: 
#bathrooms, bedrooms, sqft_living, floors, grade, view, condition, yr_built, distance

#%%
#Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np

#tylko wybrane zmienne
X= df2.drop(['price','floors','sqft_lot', 'sqft_above', 'sqft_basement' ], axis=1).values
y= df2['price'].values
y=y.reshape(-1,1)

# %%
#Cross validacja, Wybieramy najlepsze alpha

#Funkcja rysująca wykres sredniego R2 od wartoci alpha
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.savefig('C:/Users/gagat/Documents/Lingaro/AlphaSellection.png')
    plt.show()

#Inicjacja wartoci alpha na liscie    
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)    

#Na podstawie wykresu wybieramy alpha równe 0.1, gdyż jest wystarczające by znormalizować zmienne, a 
#daje nam jednoczenie wysoki wyniki CV score
# %%
# FINALNY MODEL
ridge = Ridge(alpha = 0.1, normalize=True)

#By default, scikit-learn's cross_val_score() function uses R2 as the metric of choice for regression
cv_results = cross_val_score(ridge, X,y, cv=10)
# cross validacją sprawdzana jest skutecznosć, wychodzi CV score srednio 67%, co jest dobrym wynikiem
print('Cross validation by default uses R2 as the metric of choice for regression')
print(cv_results)
print("Average 10-Fold CV Score: {}".format(np.mean(cv_results)))

#Dzielimy zbiory danych i ternujemy model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=11)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
ridge_coef = ridge.coef_
print(ridge_coef)
print(ridge.intercept_)
print(ridge.score(X_test, y_test))

columns_sel = df.drop(['id', 'date', 'price', 'floors','waterfront','yr_renovated', 'zipcode', 'lat', 
              'long', 'sqft_living15', 'sqft_lot15', 'sqft_lot', 'sqft_above', 'sqft_basement'], axis=1).columns
ridge_coef=np.transpose(ridge_coef)

#Wykres wartoć parametrów regresji Ridge dla zmiennych objasniajacych

plt.figure(6)
plt.plot(range(len(columns_sel)), ridge_coef)
plt.xticks(range(len(columns_sel)), columns_sel.values, rotation=60)
plt.margins(0.02)
plt.savefig('C:/Users/gagat/Documents/Lingaro/RidgeFetureSellection3.png')
plt.show()

#%%
# Kod do wydruku tabeli parametrów wraz ze statystykami i wartociami p-value

from scipy import stats
from sklearn.metrics import mean_squared_error

params = np.append(ridge.intercept_,ridge.coef_)
y_pred = ridge.predict(X_test)
ridge_coef = ridge.coef_
print(ridge_coef)

newX = pd.DataFrame({"Constant":np.ones(len(X_test))}).join(pd.DataFrame(X_test))
#MSE = (sum((y_test-predictions)**2))/(len(newX)-len(newX.columns))
MSE = mean_squared_error(y_test, y_pred)
print("MSE: {}".format(MSE))
 
var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params/ sd_b

p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

sd_b = np.round(sd_b,3)
ts_b = np.round(ts_b,3)
p_values = np.round(p_values,3)
params = np.round(params,4)

myDF3 = pd.DataFrame()
myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilites"] = [params,sd_b,ts_b,p_values]
print(myDF3)

# %%
# Compute and print R^2 and RMSE
print("R^2: {}".format(ridge.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
#Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). 
#Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how spread out these residuals are.
#In other words, it tells you how concentrated the data is around the line of best fit