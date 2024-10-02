import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
import sys

pd.set_option('display.max_columns', None)

def display_data_set(data):
    print(data.head())
    print("---------- ")
    print(data.describe())

def display_scatter_plot(y_test, y_pred):
    # Plot actual vs predicted exam scores with different colors
    plt.figure(figsize=(8, 6))
    # Plot the actual exam scores
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Exam Scores')
    # Plot the predicted exam scores
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Exam Scores')
    plt.xlabel('Sample Index')
    plt.ylabel('Exam Score')
    plt.title('Actual vs. Predicted Exam Scores')
    plt.legend()
    plt.show()

def display_pairplot(data, hue_field):
    sns.pairplot(data, hue=hue_field)
    plt.show()

def plot_training_history(history):
    # Plot training & validation loss values
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def convert_categories_to_numbers(data):
    categorical_columns = ['Learning_Disabilities', 'Parental_Education_Level',
                           'Distance_from_Home', 'Gender', 'Motivation_Level',
                           'Parental_Involvement', 'Access_to_Resources',
                           'Extracurricular_Activities', 'Internet_Access',
                           'Family_Income', 'Teacher_Quality', 'School_Type',
                           'Peer_Influence']
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        data[f'{column}'] = label_encoder.fit_transform(data[column])

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return [X_train_scaled, X_test_scaled]

# Load the data from the CSV file
data = pd.read_csv('./data/StudentPerformanceFactors.csv')

convert_categories_to_numbers(data)

# Define training and testing data sets
X = data.drop(columns=['Exam_Score'])  # Input is all but target
y = data['Exam_Score']                 # Target: Exam Score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data (this step is important for neural networks)
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

# Define the neural network
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1))  # Output layer
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled,
                    y_train,
                    epochs=50,
                    batch_size=128,
                    validation_data=(X_test_scaled, y_test),
                    verbose=1)

# Predict on the test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model using mean squared error
mse = root_mean_squared_error(y_test, y_pred)
print(f"Root Mean Squared Error: {mse}")

