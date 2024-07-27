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

##### Data cleaning module####
# Calculate Profit(£) based on Revenue(£) and Cost(£)
data['Profit(£)'] = data['Profit(£)'].fillna(data['Revenue(£)'] - data['Cost(£)'])
# Calculate Revenue based on profit and Cost
data['Revenue(£)'] = data['Revenue(£)'].fillna(data['Cost(£)'] + data['Profit(£)'])
# Calculate Cost based on Revenue and Profit
data['Cost(£)'] = data['Cost(£)'].fillna(data['Revenue(£)'] - data['Profit(£)'])

# Dropping rows with missing values
data.dropna(inplace=True)

# Replace 'Caramel Nut' with 'Caramel nut'
data['Confectionary'] = data['Confectionary'].replace('Caramel nut', 'Caramel Nut')

# Replace 'Choclate Chunk' with 'Chocolate Chunk'
data['Confectionary'] = data['Confectionary'].replace('Choclate Chunk', 'Chocolate Chunk')
##### End of the data cleaning module####


# creating a variable profitByCountry 
profitByCountry = data.groupby('Country(UK)')['Profit(£)'].sum().sort_values(ascending=False)

# creating a variable profitByConfectionary
profitByConfectionary = data.groupby('Confectionary')['Profit(£)'].sum().sort_values(ascending=False)

plt.figure(figsize=(12,8))
sns.scatterplot(data=data, x="Revenue(£)", y="Profit(£)", hue="Confectionary", s=100)
plt.title('Revenue vs Profit by Confectionary')
plt.xlabel('Revenue(£)')
plt.ylabel('Profit(£)')
plt.legend(title='Confectionary', bbox_to_anchor = (1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,8))
sns.scatterplot(data=data, x="Revenue(£)", y="Profit(£)", hue="Country(UK)", s=100)
plt.title('Revenue vs Profit by Country(UK)')
plt.xlabel('Revenue(£)')
plt.ylabel('Profit(£)')
plt.legend(title='Country(UK)', bbox_to_anchor = (1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()