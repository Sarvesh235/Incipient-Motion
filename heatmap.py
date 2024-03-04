import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_excel('Incipient motion.xlsx')

# Separate the data into input X and output Y
data = data.dropna()
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# Generate correlation matrix
corr_matrix = data.corr()

# Plot the heat map
plt.figure(figsize=(10, 8))
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 12 
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

plt.title('Correlation Heatmap')
plt.show()

# Rest of the code for model training and evaluation...