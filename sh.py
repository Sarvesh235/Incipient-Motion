import shap
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt  # Import Matplotlib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor,BaggingRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# Load your data from the file "data1" using pandas
data = pd.read_excel("Incipient motion.xlsx")  # Replace with the actual file name

# Assuming the last column is the target variable and the rest are features
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
#br and gbr:best
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

gbr_model = DecisionTreeRegressor()

gbr_model.fit(X_train, y_train)

# Create a SHAP explainer
explainer1 = shap.Explainer(gbr_model)

shap_values1 = explainer1(X_test)

shap_values= shap_values1

# Configure Matplotlib to use "Times New Roman" font
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 12 # Adjust the font size as needed

# Plot SHAP summary plot
shap.summary_plot(shap_values, X_test)

# Optionally, save the plot to an image file
plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
