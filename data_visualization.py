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

# creating the data visualization with subplots
# Set the style of the plots
sns.set_style("whitegrid")
# Create the figure and axes for subplots 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Distributiion of Units Sold
ax = axes[0, 0] # Select the first subplot at coordonates 0,0
sns.histplot(data['Units Sold'], kde=True, color='blue', ax=ax) # Plot a histogram with KDE
ax.set_title('Distribution of units sold')  # Set the title for the subplot

# Distributiion of Revenue
ax = axes[0, 1]
sns.histplot(data['Revenue(£)'], kde=True, color='red', ax=ax)
ax.set_title('Distribution of Revenue(£)')


# Distributiion of Cost
ax = axes[0, 2]
sns.histplot(data['Cost(£)'], kde=True, color='Orange', ax=ax)
ax.set_title('Distribution of Cost(£)')

# Distributiion of Profit
ax = axes[1, 0]
sns.histplot(data['Profit(£)'], kde=True, color='green', ax=ax)
ax.set_title('Distribution of Profit(£)')

# Distributiion of Profit by country
ax = axes[1, 1]
sns.barplot(x=profitByCountry.index, y=profitByCountry, color='magenta', ax=ax) # Plot a bar chart 
ax.set_title('Distribution of Country(UK)') # Set the title for the subplot
ax.set_xlabel('Country(UK)') # Set the x-axis label
ax.set_ylabel('Profit(£)') # Set the y-axis label
ax.tick_params(axis='x', rotation=90) # Rotate x-axis tick labels

# Distributiion of Profit by confectionary
ax = axes[1, 2]
sns.barplot(x=profitByConfectionary.index, y=profitByConfectionary, color='cyan', ax=ax)
ax.set_title('Distribution of profit By Confectionary')
ax.set_xlabel('Confectionary')
ax.set_ylabel('Profit(£)')
ax.tick_params(axis='x', rotation=90)

# Adjust the spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()