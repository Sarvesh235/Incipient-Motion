import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor,AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


# Assume X_train, y_train are your Pandas DataFrame and labels
# For illustration, let's create a dummy DataFrame
import pandas as pd
df = pd.read_excel('Incipient motion.xlsx')
df = df.dropna()
X=df.iloc[:,:-1]
y= df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Function to convert boolean array to index array
def boolean_to_index(boolean_array):
    return np.where(boolean_array)[0]

# Define the Invasive Weed Optimization algorithm
def invasive_weed_optimization(X_train, y_train, population_size, generations):
    n_features = X_train.shape[1]

    # Initialize the population with random binary vectors
    population = np.random.randint(2, size=(population_size, n_features))

    for generation in range(generations):
        # Evaluate the fitness of each individual in the population
        fitness = []
        for individual in population:
            selected_features = boolean_to_index(individual)
            X_selected = X_train.iloc[:, selected_features]
            X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                X_selected, y_train, test_size=0.2, random_state=42
            )

            # Use a simple Decision Tree classifier as an example
            model = DecisionTreeRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc=r2_score(y_test,y_pred)
            fitness.append(acc)

        # Select the top individuals for reproduction
        selected_indices = np.argsort(fitness)[-population_size:]

        # Reproduction: Create a new population by copying top individuals
        new_population = population[selected_indices]

        # Mutation: Randomly flip some bits in the selected individuals
        mutation_rate = 0.1
        mutation_mask = np.random.rand(population_size, n_features) < mutation_rate
        new_population = np.logical_xor(new_population, mutation_mask).astype(int)

        population = new_population

    # Select the best individual from the final population
    best_individual = population[np.argmax(fitness)]

    return best_individual

# Set IWO parameters
population_size = 10
generations = 5

# Apply IWO to select features
selected_features = invasive_weed_optimization(X_train, y_train, population_size, generations)

# Apply bagging with the selected features
base_classifier = DecisionTreeRegressor()
bagging_classifier = DecisionTreeRegressor()

X_selected = X_train.iloc[:, boolean_to_index(selected_features == 1)]
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_selected, y_train, test_size=0.2, random_state=42
)

bagging_classifier.fit(X_train, y_train)
y_pred = bagging_classifier.predict(X_test)
accuracy = r2_score(y_test, y_pred)
dc = pd.DataFrame(y_pred)
dc.to_excel("pr.xlsx")
dx = pd.DataFrame(y_test)
dx.to_excel("ax.xlsx")
r2= r2_score(y_pred,y_test)
mae = mean_absolute_error(y_pred,y_test)
mse = mean_squared_error(y_pred,y_test)
rmse = np.sqrt(mse)
rrse = np.sqrt(mse)/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
rae= mae/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
vaf=100 * (1 - (np.var(y_test - y_pred) / np.var(y_test)))
kge = 1 - np.sqrt((np.corrcoef(y_test, y_pred)[0, 1]- 1)**2 + (np.std(y_pred) / np.std(y_test) - 1)**2 + (np.mean(y_pred) / np.mean(y_test) - 1)**2)
print(r2,rmse,mse,mae,rrse,rae,vaf,kge)
errors=pd.DataFrame([r2,rmse,mse,mae,rrse,rae,vaf,kge])
errors.to_excel("metr.xlsx")

import joblib
joblib.dump(bagging_classifier,"./static/iwo.joblib")

import shap
explainer1 = shap.Explainer(bagging_classifier)

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

