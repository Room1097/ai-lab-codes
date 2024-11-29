import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork  # Updated to BayesianNetwork as per the warning
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination
from sklearn.naive_bayes import CategoricalNB
import numpy as np
from tqdm import tqdm

# Disable tqdm output
tqdm._instances.clear()

# Debug: Starting script execution
print("Script execution started.")

# Load data
print("Loading data...")
try:
    data = pd.read_csv('2020_bn_nb_data.txt', sep='\t')
    print("Data loaded successfully. Data shape:", data.shape)
except FileNotFoundError:
    print("Error: File '2020_bn_nb_data.txt' not found.")
    exit()

# Debug: Check initial data
print("Initial data preview:")
print(data.head())

# Debug: Check unique values before encoding
print("Unique grade values before encoding:")
print(data.iloc[:, :-1].stack().unique())

# Encode grades and qualification status
grade_map = {'AA': 10, 'AB': 9, 'BB': 8, 'BC': 7, 'CC': 6, 'CD': 5, 'DD': 4, 'F': 0}
data.replace(grade_map, inplace=True)

# Debug: Check unique values after encoding
print("Unique grade values after encoding:")
print(data.iloc[:, :-1].stack().unique())

qualification_map = {'y': 1, 'n': 0}
data['QP'].replace(qualification_map, inplace=True)

print("Unique values in 'QP' column after encoding:", data['QP'].unique())

# Ensure all data is numeric
print("Converting data to numeric types...")
data = data.apply(pd.to_numeric, errors='coerce')
print("Conversion completed.")

# Check for missing or invalid values
if data.isnull().values.any():
    print("Warning: Missing or invalid values detected!")
    print(data.isnull().sum())  # Print counts of NaN values for each column

    # Option: Drop rows with missing values
    data.dropna(inplace=True)
    print("Rows with missing values dropped. New data shape:", data.shape)

# Check if data is empty
if data.empty:
    print("Error: Processed data is empty. Check the input file and encoding steps.")
    exit()

# Split data into features and target
X = data.iloc[:, :-1]  # Course grades
y = data.iloc[:, -1]   # Internship qualification status

# Debug: Check shapes of X and y
print("Feature set shape:", X.shape)
print("Target set shape:", y.shape)

# Ensure that X and y are not empty
if X.empty or y.empty:
    print("Error: Features or target set is empty after processing. Exiting.")
    exit()

# Repeated experiments
results = {'naive_bayes': [], 'bayesian_network': []}
print("Starting repeated experiments...")

for i in range(20):
    print(f"\nExperiment {i+1} started.")
    
    # Split data
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=np.random.randint(1000))
    print("Data split completed. Training set size:", X_train.shape, "Testing set size:", X_test.shape)

    # Debug: Check data types
    print("Training data types:")
    print(X_train.dtypes)
    print("Training target unique values:", y_train.unique())

    # Naive Bayes Classifier
    print("Training Naive Bayes classifier...")
    nb_model = CategoricalNB()
    nb_model.fit(X_train, y_train)
    print("Naive Bayes classifier trained.")
    
    print("Making predictions with Naive Bayes classifier...")
    y_pred_nb = nb_model.predict(X_test)
    nb_accuracy = accuracy_score(y_test, y_pred_nb)
    results['naive_bayes'].append(nb_accuracy)
    print(f"Naive Bayes accuracy for Experiment {i+1}: {nb_accuracy}")

    # Bayesian Network Classifier
    print("Learning Bayesian Network structure...")
    hc = HillClimbSearch(pd.concat([X_train, y_train], axis=1))
    model = hc.estimate(scoring_method=BicScore(pd.concat([X_train, y_train], axis=1)))

    # Use BayesianNetwork instead of BayesianModel
    bn_model = BayesianNetwork(model.edges())
    bn_model.fit(pd.concat([X_train, y_train], axis=1))

    # Debug: Check columns in the Bayesian Network model and test set
    print("Columns used in the Bayesian Network model:", X_train.columns)
    print("Test set columns:", X_test.columns)
    print("Model structure (nodes):", bn_model.nodes())

    print("Making predictions with Bayesian Network...")
    inference = VariableElimination(bn_model)

    # Ensure that all the columns in the test set are used in inference
    y_pred_bn = [inference.map_query(variables=[y.name], evidence=dict(row)) for _, row in X_test.iterrows()]
    y_pred_bn = [pred[y.name] for pred in y_pred_bn]
    bn_accuracy = accuracy_score(y_test, y_pred_bn)
    results['bayesian_network'].append(bn_accuracy)
    print(f"Bayesian Network accuracy for Experiment {i+1}: {bn_accuracy}")

print("\nRepeated experiments completed.")

# Report average results
print("\nCalculating average results...")
print("Naive Bayes Average Accuracy:", np.mean(results['naive_bayes']))
print("Bayesian Network Average Accuracy:", np.mean(results['bayesian_network']))

# Perform a specific query to predict grade in PH100
print("\nPredicting grade in PH100 based on given evidence...")
evidence = {'EC100': grade_map['DD'], 'IT101': grade_map['CC'], 'MA101': grade_map['CD']}
query_result = inference.map_query(variables=['PH100'], evidence=evidence)

# Convert predicted grade back to the original grade system
predicted_grade = [grade for grade, value in grade_map.items() if value == query_result['PH100']][0]
print(f"The predicted grade in PH100 is: {predicted_grade}")
