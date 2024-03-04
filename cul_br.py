#cultural algorithm with knn

import numpy as np
from sklearn.datasets import load_iris
from xgboost import XGBClassifier

# from sklearn.neural_network import 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor,AdaBoostRegressor,BaggingRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import accuracy_score,r2_score,mean_absolute_error,mean_squared_error

# Define the fitness function using KNN
knn = BaggingRegressor()
def fitness_function(X_train, y_train,X_test,ytest, solution, n_neighbors):
    knn.fit(X_train * solution, y_train)
    y_pred = knn.predict(X_test * solution)
    return r2_score(ytest, y_pred)

# Initialize the population and cultural space
population_size = 20
n_features = 7  # Number of features in the dataset
population = np.random.randint(2, size=(population_size, n_features))
cultural_knowledge = np.random.randint(2, size=n_features)

# Load a dataset and split it into training and testing sets
import pandas as pd
df = pd.read_excel("Incipient motion.xlsx")
df=df.dropna()
X = df.iloc[:,:-1]

y = df['tb']
from sklearn import preprocessing
from sklearn import utils

#convert y values to categorical values
# lab = preprocessing.LabelEncoder()
# y_transformed = lab.fit_transform(y)
# print(y_transformed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
df = pd.DataFrame(y_test)
# print(y_test)
df.to_excel("Act.xlsx")
# Parameters
max_generations = 100
mutation_rate = 0.1
knowledge_sharing_rate = 0.2
n_neighbors = 3  # K for KNN

# Main loop
for generation in range(max_generations):
    # Evaluate fitness for each individual in the population
    fitness_values = [fitness_function(X_train, y_train,X_test,y_test, individual, n_neighbors) for individual in population]

    # Cultural knowledge transfer
    best_individual = population[np.argmax(fitness_values)]
    if np.random.rand() < knowledge_sharing_rate:
        cultural_knowledge = best_individual.copy()

    # Mutation
    for i in range(population_size):
        if np.random.rand() < mutation_rate:
            random_bit = np.random.randint(n_features)
            population[i, random_bit] = 1 - population[i, random_bit]  # Flip the bit

    # Recalculate fitness
    fitness_values = [fitness_function(X_train, y_train, X_test,y_test,individual, n_neighbors) for individual in population]

    # Select parents based on fitness for reproduction
    parent_indices = np.argsort(fitness_values)[-2:]

    # Reproduction (crossover)
    crossover_point = np.random.randint(1, n_features)
    new_individual = np.concatenate((population[parent_indices[0], :crossover_point],
                                    population[parent_indices[1], crossover_point:]))
    population[np.random.choice(range(population_size))] = new_individual

# Select the best individual
best_individual = population[np.argmax(fitness_values)]
knn.fit(X_train * best_individual, y_train)
y_pred = knn.predict(X_test * best_individual)
dc = pd.DataFrame(y_pred)
dc.to_excel("pr.xlsx")
r2=r2_score(y_test, y_pred)
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mse)
rrse = rmse/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
rae= mae/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
vaf=100 * (1 - (np.var(y_test - y_pred) / np.var(y_test)))
kge = 1 - np.sqrt((np.corrcoef(y_test, y_pred)[0, 1]- 1)**2 + (np.std(y_pred) / np.std(y_test) - 1)**2 + (np.mean(y_pred) / np.mean(y_test) - 1)**2)

print(r2,rmse,mse,mae,rrse,rae,vaf,kge)
errors=pd.DataFrame([r2,rmse,mse,mae,rrse,rae,vaf,kge])
errors.to_excel("metr.xlsx")

import joblib
joblib.dump(knn,"./static/ca.joblib")

import shap
import matplotlib.pyplot as plt
explainer1 = shap.Explainer(knn)

shap_values1 = explainer1(X_test)

shap_values= shap_values1

# Configure Matplotlib to use "Times New Roman" font
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 12 # Adjust the font size as needed

# Plot SHAP summary plot
shap.summary_plot(shap_values, X_test)

# Show the plot
plt.show()
