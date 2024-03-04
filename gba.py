import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor,RandomForestRegressor,AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.tree import ExtraTreeRegressor,DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

import pandas as pd
df = pd.read_excel('Incipient motion.xlsx')
df = df.dropna()
X=df.iloc[:,:-1]
y= df.iloc[:,-1]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define the base regression model (you can use other models as well)
base_regressor = GradientBoostingRegressor()

# Define the BaggingRegressor using the base regression model
bagging_regressor = GradientBoostingRegressor()

# Train the BaggingRegressor
bagging_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = bagging_regressor.predict(X_test)
dc = pd.DataFrame(y_pred)
dc.to_excel("pr.xlsx")
# Evaluate the performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse= np.sqrt(mse)
mae = mean_absolute_error(y_pred,y_test)
rrse = np.sqrt(mse)/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
rae= mae/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
vaf=100 * (1 - (np.var(y_test - y_pred) / np.var(y_test)))
kge = 1 - np.sqrt((np.corrcoef(y_test, y_pred)[0, 1]- 1)**2 + (np.std(y_pred) / np.std(y_test) - 1)**2 + (np.mean(y_pred) / np.mean(y_test) - 1)**2)

print(r2,rmse,mse,mae,rrse,rae,vaf,kge)
errors=pd.DataFrame([r2,rmse,mse,mae,rrse,rae,vaf,kge])
# errors.to_excel("metr.xlsx")

import joblib
joblib.dump(bagging_regressor,"./static/gba.joblib")


import shap
explainer1 = shap.Explainer(bagging_regressor)

shap_values1 = explainer1(X_test)

shap_values= shap_values1
import matplotlib.pyplot as plt
# Configure Matplotlib to use "Times New Roman" font
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 12 # Adjust the font size as needed

# Plot SHAP summary plot
shap.summary_plot(shap_values, X_test)

# Show the plot
plt.show()