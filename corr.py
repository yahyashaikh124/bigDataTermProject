import pandas as pd
import numpy as np

df = pd.read_csv("./archive/ecommerce_customer_data_large.csv")

df.isnull().sum()
df.fillna(0, inplace=True)

df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')

df['year'] = df['Purchase Date'].dt.year
df['month'] = df['Purchase Date'].dt.month
df['day'] =  df['Purchase Date'].dt.day
df['dayOfweek'] = df['Purchase Date'].dt.dayofweek
#df['date_yy_mm'] = pd.to_datetime(df['Purchase Date']).dt.strftime('%Y-%m')

df['revenue'] = df['Product Price'] * df['Quantity']

df_encoded = pd.get_dummies(df, columns=['Payment Method', 'Product Category', 'Gender'])

numeric_columns = ['Product Price', 'Quantity', 'Total Purchase Amount', 'Customer Age', 'Returns', 'Age', 'Churn', 'year', 'month' , 'dayOfweek']

numeric_columns += [col for col in df_encoded.columns if col.startswith('Payment Method_') 
                    or col.startswith('Product Category_') 
                    or col.startswith('Gender_')]

correlation_matrix = df_encoded[numeric_columns].corr()

churn_correlation = correlation_matrix['Churn'].sort_values(ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'churn_correlation' is your correlation series
#churn_correlation = churn_correlation

# Convert the dictionary into a Pandas Series
churn_corr_series = pd.Series(churn_correlation[1:])

# Sort values to visualize better
churn_corr_series = churn_corr_series.sort_values()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Plot a horizontal barplot with Seaborn
sns.barplot(x=churn_corr_series.values, y=churn_corr_series.index, palette='coolwarm')

# Add labels and title
plt.title('Correlation of Features with Churn', fontsize=16)
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.ylabel('Features', fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()


