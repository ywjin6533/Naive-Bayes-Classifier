import matplotlib.pyplot as plt
import numpy as np
from bayes import load_raw_data, training, predict


epsilons = [1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1, 10, 100]  
metrics = {"accuracy": [], "precision": [], "recall": []}

# Load training and testing data
train_file = "training.csv"
test_file = "testing.csv"
train_instances, train_labels = load_raw_data(train_file)
test_instances, test_labels = load_raw_data(test_file)

for epsilon in epsilons:
    # Train the model
    parameters = training(train_instances, train_labels)
    hyperparameter = {'epsileon': epsilon}

    # Predict using test data
    predictions = [predict(instance, parameters, hyperparameter) for instance in test_instances]

    # Evaluate performance
    correct = sum([pred == true for pred, true in zip(predictions, test_labels)])
    accuracy = correct / len(test_labels) * 100
    tp = sum([pred == 1 and true == 1 for pred, true in zip(predictions, test_labels)])
    fp = sum([pred == 1 and true == 0 for pred, true in zip(predictions, test_labels)])
    fn = sum([pred == 0 and true == 1 for pred, true in zip(predictions, test_labels)])
    precision = tp / (tp + fp) * 100 if tp + fp > 0 else 0
    recall = tp / (tp + fn) * 100 if tp + fn > 0 else 0

    # Save metrics
    metrics["accuracy"].append(accuracy)
    metrics["precision"].append(precision)
    metrics["recall"].append(recall)

    print(f"Epsilon: {epsilon:.0e}")
    print(f"  Accuracy: {accuracy:.9f}%")
    print(f"  Precision: {precision:.9f}%")
    print(f"  Recall: {recall:.9f}%")


# Plot results

plt.figure()
plt.plot(epsilons, metrics["accuracy"], marker='o')
plt.xscale("log")
plt.title("Accuracy vs Smoothing Factor (epsilon)")
plt.xlabel("Smoothing Factor (epsilon)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(epsilons, metrics["precision"], marker='o')
plt.xscale("log")
plt.title("Precision vs Smoothing Factor (epsilon)")
plt.xlabel("Smoothing Factor (epsilon)")
plt.ylabel("Precision")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(epsilons, metrics["recall"], marker='o')
plt.xscale("log")
plt.title("Recall vs Smoothing Factor (epsilon)")
plt.xlabel("Smoothing Factor (epsilon)")
plt.ylabel("Recall")
plt.grid(True)
plt.show()
