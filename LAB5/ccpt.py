import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# Load the data
data_path = "files/2020_bn_nb_data.txt"  # Update this path to your file location
df = pd.read_csv(data_path, sep="\t")

# Define features and target
features = df.columns[:-1]  # All columns except 'QP'
target = "QP"

# Extract data and target
X = df[features]
y = df[target]

# Adjust the encoder to handle unknown categories
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)

# List to store accuracy for each run
nb_accuracies = []

# Perform 20 random splits for training and testing
for _ in range(20):
    # Split the data into 70% train and 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=None)
    
    # Fit the encoder on training data to determine categories
    ordinal_encoder.fit(X_train)
    
    # Transform training and testing data with the learned categories
    X_train_enc = ordinal_encoder.transform(X_train)
    X_test_enc = ordinal_encoder.transform(X_test)
    
    # Replace NaNs with the maximum category index + 1
    max_value = np.nanmax(X_train_enc) + 1
    
    # Replace NaNs with max_value in training data
    X_train_enc = np.nan_to_num(X_train_enc, nan=max_value)
    
    # Ensure no feature index exceeds the number of classes in training
    for feature_idx in range(X_train_enc.shape[1]):
        max_index = int(X_train_enc[:, feature_idx].max())
        X_test_enc[:, feature_idx] = np.clip(X_test_enc[:, feature_idx], 0, max_index)
    
    # Create and fit the Naive Bayes model
    model = CategoricalNB()
    model.fit(X_train_enc, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test_enc)
    accuracy = accuracy_score(y_test, y_pred)
    nb_accuracies.append(accuracy)

# Calculate mean and standard deviation of accuracy
nb_mean_accuracy = np.mean(nb_accuracies)
nb_std_accuracy = np.std(nb_accuracies)

print(f"Naive Bayes Mean Accuracy: {nb_mean_accuracy}")
print(f"Naive Bayes Std Deviation: {nb_std_accuracy}")
