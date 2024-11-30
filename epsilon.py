import matplotlib.pyplot as plt
import numpy as np
from bayes import load_raw_data, training, predict


epsilons = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]  
metrics = {"accuracy": [], "precision": [], "recall": []}

# Load training and testing data
train_file = "training.csv"
test_file = "testing.csv"
train_instances, train_labels = load_raw_data(train_file)
test_instances, test_labels = load_raw_data(test_file)

for epsilon in epsilons:
    # Modify the Gaussian probability function dynamically
    def gaussian_probability(x, mean, std):
        if std == 0:
            return 1.0 if x == mean else 0.0
        exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2 + epsilon)))
        return (1 / (np.sqrt(2 * np.pi) * (std + epsilon))) * exponent

    # Train the model
    parameters = training(train_instances, train_labels)

    # Predict using test data
    predictions = [predict(instance, parameters) for instance in test_instances]

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

# Plot results

plt.figure()
plt.plot(epsilons, metrics["accuracy"], marker='o')
plt.xscale("log")
plt.title("Accuracy vs Smoothing Factor (epsilon)")
plt.xlabel("Smoothing Factor (epsilon)")
plt.ylabel("Accuracy")
plt.ylim(92, 94)  
plt.yticks(np.arange(92, 94, 0.2))  
plt.grid(axis='y', linestyle='--', alpha=0.7) 
plt.show()

plt.figure()
plt.plot(epsilons, metrics["precision"], marker='o')
plt.xscale("log")
plt.title("Precision vs Smoothing Factor (epsilon)")
plt.xlabel("Smoothing Factor (epsilon)")
plt.ylabel("Precision")
plt.ylim(67, 68)  
plt.yticks(np.arange(67, 68, 0.1))  
plt.grid(axis='y', linestyle='--', alpha=0.7) 
plt.show()

plt.figure()
plt.plot(epsilons, metrics["recall"], marker='o')
plt.xscale("log")
plt.title("Recall vs Smoothing Factor (epsilon)")
plt.xlabel("Smoothing Factor (epsilon)")
plt.ylabel("Recall")
plt.ylim(88, 90)  
plt.yticks(np.arange(88, 90, 0.2))  
plt.grid(axis='y', linestyle='--', alpha=0.7) 
plt.show()