#PSO_ADABOOST
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
# Load the dataset from an Excel file
df = pd.read_excel('Incipient motion.xlsx')
df = df.dropna()
X=df.iloc[:,:-1]
y= df.iloc[:,-1]

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)
# Define the objective function\
rf = XGBRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
dc = pd.DataFrame(y_pred) 
# dc.to_excel("pr.xlsx")
print(y_train)
def objective_function(x):
    # Create a Random Forest model with the given hyperparameters
    r2= r2_score(y_pred,y_test)
    mae = mean_absolute_error(y_pred,y_test)
    mse = mean_squared_error(y_pred,y_test)
    rrse = np.sqrt(mse)/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
    rae= mae/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
    vaf=100 * (1 - (np.var(y_test - y_pred) / np.var(y_test)))
    kge = 1 - np.sqrt((np.corrcoef(y_test, y_pred)[0, 1]- 1)**2 + (np.std(y_pred) / np.std(y_test) - 1)**2 + (np.mean(y_pred) / np.mean(y_test) - 1)**2)

    return kge
# PSO parameters
num_particles = 30
num_dimensions = 7 # Assuming you have 8 to 9 features in your dataset
max_iter = 10
c1 = 2.0
c2 = 2.0
w = 0.7

# Initialize particle positions and velocities.....
particles = np.random.rand(num_particles, num_dimensions)
velocities = np.random.rand(num_particles, num_dimensions)

# Initialize particle best positions and global best....
pbest = particles.copy()
pbest_fitness = np.zeros(num_particles)
gbest = np.zeros(num_dimensions)
gbest_fitness = float('inf')
gb =[]
# Main PSO loop....
for i in range(max_iter):
    for j in range(num_particles):
        fitness = objective_function(particles[j])
        # print(particles[j])

        # Update particle best if a better solution is found.....
        if fitness < pbest_fitness[j]:
            pbest[j] = particles[j]
            pbest_fitness[j] = fitness

        # Update global best if a better solution is found....
        if fitness < gbest_fitness:
            gbest = particles[j]
            gbest_fitness = fitness

    for j in range(num_particles):
        # Update particle velocities and positions
        r1, r2 = np.random.rand(), np.random.rand()
        velocities[j] = w * velocities[j] + c1 * r1 * (pbest[j] - particles[j]) + c2 * r2 * (gbest - particles[j])
        particles[j] = particles[j] + velocities[j]

    print(f"Iteration {i}: Best fitness = {gbest_fitness}")
    gb.append(gbest_fitness)


import numpy as np
res= np.mean(gb)
print(f"Optimal solution found at x = {gbest}, with a fitness value of {res} of ADABOOST MOdel")

import joblib
joblib.dump(rf,"./static/pso.joblib")


import shap
explainer1 = shap.Explainer(rf)

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