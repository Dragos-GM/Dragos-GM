import pandas as pd 
#pandas: For data manipulation and analysis.
import matplotlib.pyplot as plt
#matplotlib.pyplot: For plotting and visualization.
from statsmodels.tsa.seasonal import seasonal_decompose
#statsmodels: For time series analysis and seasonal decomposition.
import numpy as np
#numpy: For numerical operations and array handling.
import seaborn as sns
#seaborn: For creating high-quality plots and data visualizations.
import scipy.stats as stats
#scipy.stats: For statistical functions and distributions.
from sklearn.model_selection import train_test_split
#sklearn.model_selection: For splitting the data into training and testing sets.
from sklearn.linear_model import LinearRegression
#sklearn.linear_model: For linear regression model fitting and evaluation.
from sklearn.metrics import mean_squared_error, r2_score
#sklearn.metrics: For evaluating model performance using mean squared error and R-squared score.
from efficient_apriori import apriori
#apriori: For association rule mining.

# create the variable "data" that will read the excel file using the pandas library
data = pd.read_excel('Data.xlsx') 

#Create a variable called "missing" because i want to reuse it
missing = "The number of missing data points (sum)"
# Check for missing values
print("\033[91m" + missing + "\033[0m")
print(data.isnull().sum())

# Calculate Profit(£) based on Revenue(£) and Cost(£)
data['Profit(£)'] = data['Profit(£)'].fillna(data['Revenue(£)'] - data['Cost(£)'])
# Calculate Revenue based on profit and Cost
data['Revenue(£)'] = data['Revenue(£)'].fillna(data['Cost(£)'] + data['Profit(£)'])
# Calculate Cost based on Revenue and Profit
data['Cost(£)'] = data['Cost(£)'].fillna(data['Revenue(£)'] - data['Profit(£)'])


# Check for missing values after filling
print("\033[92m" + missing + "\033[0m")
print(data.isnull().sum())

# Dropping rows with missing values
data.dropna(inplace=True)

print("\033[93m" + missing + "final" + "\033[0m")
print(data.isnull().sum())

# To get unique values of Country(UK)
unique_country = data['Country(UK)'].unique()
print("Unique values of Country(UK):")
print(unique_country)

# To get unique values of Confectionary
unique_confectionary = data['Confectionary'].unique()
print("Unique values of Confectionary:")
print(unique_confectionary)