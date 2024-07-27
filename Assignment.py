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


# Display the first and last few rows of the data set
#print(data.head(), "\n\n", data.tail())

# Check for missing values
print(data.isnull().sum())

columns_to_fill = ['Profit(£)', 'Revenue(£)', 'Cost(£)']

# Fill missing values using a loop
for column in columns_to_fill:
    data[column] = data['Profit(£)'].fillna(data['Revenue(£)'] - data['Cost(£)'])
    data[column] = data['Revenue(£)'].fillna(data['Cost(£)'] + data['Profit(£)'])
    data[column] = data['Cost(£)'].fillna(data['Revenue(£)'] - data['Profit(£)'])


print(data.isnull().sum())