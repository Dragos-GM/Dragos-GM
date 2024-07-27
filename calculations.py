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


# Dropping rows with missing values
data.dropna(inplace=True)

# calculation 1 - profit maergin index in "%"
data['index']=((data['Profit(£)']/data['Revenue(£)'])*100)
# Grouping by 'Confectionary' and 'Country(UK)' and getting unique values of 'index'
unique_results = data.groupby(['Confectionary', 'Country(UK)'])['index'].unique()

print(unique_results)

# Calculating profit margin by country
pCountryIndex=data.groupby('Country(UK)')['index'].mean().sort_values(ascending=False)
# Calculating the profit margin by confectionary
pConfectionaryIndex=data.groupby('Confectionary')['index'].mean().sort_values(ascending=False)

print('\n', pCountryIndex)
print('\n', pConfectionaryIndex)

# Plotting the results
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Subplot 1 - Bar chart for profit by country
axs[0, 0].bar(pCountryIndex.index, pCountryIndex, color='blue')
axs[0, 0].set_title('Profit by country based on profit margin')
axs[0, 0].set_xlabel('Country(UK)')
axs[0, 0].set_ylabel('Profit(£)')
axs[0, 0].tick_params(axis='x', rotation=45)
axs[0, 0].grid(True)

# Subplot 2 - KDE plot for profit by country
sns.kdeplot(data=data, x='index', hue='Country(UK)', ax=axs[0, 1])
axs[0, 1].set_title('Profit by country - KDE Plot')
axs[0, 1].set_xlabel('Profit Margin Index (%)')
axs[0, 1].set_ylabel('Density')

# Subplot 3 - Bar chart for profit by confectionary
axs[1, 0].bar(pConfectionaryIndex.index, pConfectionaryIndex, color='green')
axs[1, 0].set_title('Profit by confectionary based on profit margin')
axs[1, 0].set_xlabel('Confectionary')
axs[1, 0].set_ylabel('Profit(£)')
axs[1, 0].tick_params(axis='x', rotation=45)
axs[1, 0].grid(True)

# Subplot 4 - KDE plot for profit by confectionary
sns.kdeplot(data=data, x='index', hue='Confectionary', ax=axs[1, 1])
axs[1, 1].set_title('Profit by confectionary - KDE Plot')
axs[1, 1].set_xlabel('Profit Margin Index (%)')
axs[1, 1].set_ylabel('Density')

# Adjusting the spacing between subplots
plt.tight_layout()

# Displaying the plot
plt.show()
print('\n\n\n\n')
data['z-score']=stats.zscore(data['index'])


avgZscore=data['z-score'].mean()
print('\n', f"The avarage Z-score is: {avgZscore}%")
print('\n', f"The avarage Z-score is: {avgZscore:.2f}%")

# Visualizing the z-score

sns.scatterplot(data=data, x='Units Sold', y='z-score', hue='Confectionary', style='Confectionary')
plt.title('Z-score >> Profit margin vs Units sold')
plt.xlabel('Units Sold')
plt.ylabel('Z-score of Profit margin')
plt.tight_layout()
plt.show()