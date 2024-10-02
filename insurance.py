import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

pd.set_option('display.max_columns', None)

# Load the data from the CSV file
df = pd.read_csv('./data/insurance.csv')

# Convert categorical variables
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Feature scaling
scaler = StandardScaler()
df_encoded[['age', 'bmi', 'children']] = scaler.fit_transform(df_encoded[['age', 'bmi', 'children']])

# Preview the modified dataframe
print(df_encoded.head())

# Define the feature matrix and target vector
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
#model = LinearRegression()
#model = DecisionTreeRegressor()
#model = RandomForestRegressor()
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Linear Regression RMSE:", root_mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))



def display_decisiontree(model):

    # Print the tree as text
    tree_text = export_text(model, feature_names=['age', 'bmi', 'sex_male'])
    print(tree_text)

    # Plot the decision tree
    plt.figure(figsize=(15, 10))
    plot_tree(model, feature_names=['age', 'bmi', 'sex_male'], filled=True)
    plt.show()

def display_partial_dependence(model, X_train):
    # Partial dependence plot for 'age' and 'bmi'
    features_to_plot = ['age', 'bmi']
    
    # Create and plot the partial dependence display
    display = PartialDependenceDisplay.from_estimator(model, X_train, features_to_plot, grid_resolution=50)
    display.plot()
    plt.show()

def display_scatterplot(y_test, y_pred):
    # Assuming you have trained a linear regression model and made predictions
    # X_test: your test data (features)
    # y_test: actual values from the test data
    # y_pred_lr: predicted values from the model (Linear Regression predictions)

    # Create scatter plot for actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7, label='Predicted vs Actual')

    # Plot a line of perfect prediction (y = x)
    max_value = max(max(y_test), max(y_pred))
    min_value = min(min(y_test), min(y_pred))
    plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', label='Perfect Prediction')

    # Set plot labels and title
    plt.xlabel('Actual Charges')
    plt.ylabel('Predicted Charges')
    plt.title('Actual vs. Predicted Charges (Linear Regression)')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

#display_decisiontree(model)
#display_partial_dependence(model, X_train)
display_scatterplot(y_test, y_pred)