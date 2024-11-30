import os
import sys
import argparse
import logging
import math

def calculate_mean(numbers):
    return sum(numbers) / len(numbers)

def calculate_std(numbers, mean):
    squared_diff_sum = sum((x - mean) ** 2 for x in numbers)
    return math.sqrt(squared_diff_sum / (len(numbers) - 1)) if len(numbers) > 1 else 0

def gaussian_probability(x, mean, std):
    if std == 0:
        return 1.0 if x == mean else 0.0
    exponent = math.exp(-((x - mean) ** 2) / (2 * (std ** 2 + 1e-6)))
    return (1 / (math.sqrt(2 * math.pi) * (std + 1e-6))) * exponent

def training(instances, labels):
    """
    Naive Bayes Classifier 학습 함수
    """
    # 클래스별로 데이터 분리
    instances_0 = [inst[1:] for inst, label in zip(instances, labels) if label == 0]  # date 제외
    instances_1 = [inst[1:] for inst, label in zip(instances, labels) if label == 1]
    
    # 사전 확률 계산
    total_instances = len(instances)
    prior_0 = len(instances_0) / total_instances
    prior_1 = len(instances_1) / total_instances
    
    # 각 특성의 평균과 표준편차 계산
    n_features = len(instances[0]) - 1  # date 제외
    mean_0 = []
    std_0 = []
    mean_1 = []
    std_1 = []
    
    for feature_idx in range(n_features):
        # 클래스 0의 통계치
        feature_values_0 = [float(inst[feature_idx]) for inst in instances_0]
        mean_0_value = calculate_mean(feature_values_0)
        mean_0.append(mean_0_value)
        std_0.append(calculate_std(feature_values_0, mean_0_value))
        
        # 클래스 1의 통계치
        feature_values_1 = [float(inst[feature_idx]) for inst in instances_1]
        mean_1_value = calculate_mean(feature_values_1)
        mean_1.append(mean_1_value)
        std_1.append(calculate_std(feature_values_1, mean_1_value))
    
    return {
        'prior_0': prior_0,
        'prior_1': prior_1,
        'mean_0': mean_0,
        'std_0': std_0,
        'mean_1': mean_1,
        'std_1': std_1
    }

def predict(instance, parameters):
    """
    학습된 Naive Bayes Classifier로 예측하는 함수
    """
    features = [float(x) for x in instance[1:]]  # date 제외
    
    # 각 클래스에 대한 로그 확률 계산
    log_prob_0 = math.log(parameters['prior_0'])
    log_prob_1 = math.log(parameters['prior_1'])
    
    # 각 특성에 대한 확률 계산
    for i, feature in enumerate(features):
        # 클래스 0에 대한 확률
        prob_0 = gaussian_probability(feature, 
                                    parameters['mean_0'][i], 
                                    parameters['std_0'][i])
        log_prob_0 += math.log(prob_0 + 1e-10)  # log(0) 방지
        
        # 클래스 1에 대한 확률
        prob_1 = gaussian_probability(feature, 
                                    parameters['mean_1'][i], 
                                    parameters['std_1'][i])
        log_prob_1 += math.log(prob_1 + 1e-10)
    
    return 1 if log_prob_1 > log_prob_0 else 0

def report(predictions, answers):
    if len(predictions) != len(answers):
        logging.error("The lengths of two arguments should be same")
        sys.exit(1)

    # accuracy
    correct = 0
    for idx in range(len(predictions)):
        if predictions[idx] == answers[idx]:
            correct += 1
    accuracy = round(correct / len(answers), 2) * 100

    # precision
    tp = 0
    fp = 0
    for idx in range(len(predictions)):
        if predictions[idx] == 1:
            if answers[idx] == 1:
                tp += 1
            else:
                fp += 1
    precision = round(tp / (tp + fp), 2) * 100

    # recall
    tp = 0
    fn = 0
    for idx in range(len(answers)):
        if answers[idx] == 1:
            if predictions[idx] == 1:
                tp += 1
            else:
                fn += 1
    recall = round(tp / (tp + fn), 2) * 100

    logging.info("accuracy: {}%".format(accuracy))
    logging.info("precision: {}%".format(precision))
    logging.info("recall: {}%".format(recall))

def load_raw_data(fname):
    instances = []
    labels = []
    with open(fname, "r") as f:
        f.readline()
        for line in f:
            tmp = line.strip().split(", ")
            tmp[1] = float(tmp[1])
            tmp[2] = float(tmp[2])
            tmp[3] = float(tmp[3])
            tmp[4] = float(tmp[4])
            tmp[5] = int(tmp[5])
            tmp[6] = int(tmp[6])
            tmp[7] = float(tmp[7])
            tmp[8] = int(tmp[8])
            instances.append(tmp[:-1])
            labels.append(tmp[-1])
    return instances, labels

def run(train_file, test_file):
    # training phase
    instances, labels = load_raw_data(train_file)
    logging.debug("instances: {}".format(instances))
    logging.debug("labels: {}".format(labels))
    parameters = training(instances, labels)

    # testing phase
    instances, labels = load_raw_data(test_file)
    predictions = []
    for instance in instances:
        result = predict(instance, parameters)

        if result not in [0, 1]:
            logging.error("The result must be either 0 or 1")
            sys.exit(1)

        predictions.append(result)
    
    # report
    report(predictions, labels)

def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", required=True, metavar="<file path to the training dataset>", help="File path of the training dataset", default="training.csv")
    parser.add_argument("-u", "--testing", required=True, metavar="<file path to the testing dataset>", help="File path of the testing dataset", default="testing.csv")
    parser.add_argument("-l", "--log", help="Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)", type=str, default="INFO")

    args = parser.parse_args()
    return args

def main():
    args = command_line_args()
    logging.basicConfig(level=args.log)

    if not os.path.exists(args.training):
        logging.error("The training dataset does not exist: {}".format(args.training))
        sys.exit(1)

    if not os.path.exists(args.testing):
        logging.error("The testing dataset does not exist: {}".format(args.testing))
        sys.exit(1)

    run(args.training, args.testing)

if __name__ == "__main__":
    main()
