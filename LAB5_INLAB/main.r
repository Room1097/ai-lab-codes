# Load required libraries
library(bnlearn)
library(e1071)
library(caret)

# Read the dataset and preprocess it
grades_data <- read.table("2020_bn_nb_data.txt", header = TRUE)

# Convert character columns to factors
grades_data[] <- lapply(grades_data, function(col) {
  if (is.character(col)) as.factor(col) else col
})

# Validate the presence of the target variable
if (!"QP" %in% colnames(grades_data)) {
  stop("Error: 'QP' variable is missing in the dataset.")
}

# Learn the Bayesian Network structure
learned_bn <- hc(grades_data)
plot(learned_bn)

# Fit the Bayesian Network with the data
fitted_bn <- bn.fit(learned_bn, grades_data)

# Print the nodes of the network
cat("Nodes in the Bayesian Network:", nodes(learned_bn), "\n")

# Predict PH100 grade for specific inputs
input_data <- data.frame(
  EC100 = factor("DD", levels = levels(grades_data$EC100)),
  IT101 = factor("CC", levels = levels(grades_data$IT101)),
  MA101 = factor("CD", levels = levels(grades_data$MA101))
)
predicted_grade <- predict(fitted_bn, node = "PH100", data = input_data, method = "exact")
cat("Predicted grade for PH100 (EC100: DD, IT101: CC, MA101: CD):", predicted_grade, "\n")

# Initialize random seed and set parameters
set.seed(123)
num_trials <- 20
nb_accuracies <- numeric(num_trials)

# Evaluate Naive Bayes Classifier
for (trial in 1:num_trials) {
  # Split dataset into training and testing sets
  train_indices <- sample(1:nrow(grades_data), size = 0.7 * nrow(grades_data))
  train_set <- grades_data[train_indices, ]
  test_set <- grades_data[-train_indices, ]
  
  # Train and evaluate Naive Bayes model
  nb_model <- naiveBayes(QP ~ ., data = train_set)
  nb_predictions <- predict(nb_model, newdata = test_set)
  nb_confusion <- confusionMatrix(nb_predictions, test_set$QP)
  nb_accuracies[trial] <- nb_confusion$overall['Accuracy']
}

avg_nb_accuracy <- mean(nb_accuracies, na.rm = TRUE)
cat("Average accuracy of Naive Bayes model over", num_trials, "trials:", avg_nb_accuracy, "\n")

# Summarize data for specific variables
cat("Summary of MA101:\n")
print(summary(grades_data$MA101))
cat("Summary of IT101:\n")
print(summary(grades_data$IT101))
cat("Summary of EC100:\n")
print(summary(grades_data$EC100))
# Evaluate Bayesian Network Classifier
bn_accuracies <- numeric(num_trials)

for (trial in 1:num_trials) {
  # Split dataset into training and testing sets
  train_indices <- sample(1:nrow(grades_data), size = 0.7 * nrow(grades_data))
  train_set <- grades_data[train_indices, ]
  test_set <- grades_data[-train_indices, ]
  
  # Train and evaluate Bayesian Network
  trial_bn <- hc(train_set)
  trial_bn_fit <- bn.fit(trial_bn, train_set)
  bn_predictions <- predict(trial_bn_fit, node = "QP", data = test_set, method = "exact")
  bn_confusion <- confusionMatrix(bn_predictions, test_set$QP)
  bn_accuracies[trial] <- bn_confusion$overall['Accuracy']
}

avg_bn_accuracy <- mean(bn_accuracies, na.rm = TRUE)
cat("Average accuracy of Bayesian Network model over", num_trials, "trials:", avg_bn_accuracy, "\n")
