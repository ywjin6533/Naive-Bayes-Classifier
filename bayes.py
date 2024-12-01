import os
import sys
import argparse
import logging
import math
from sklearn.metrics import precision_recall_fscore_support

def calculate_mean(numbers):
    return sum(numbers) / len(numbers)

def calculate_std(numbers, mean):
    squared_diff_sum = sum((x - mean) ** 2 for x in numbers)
    return math.sqrt(squared_diff_sum / (len(numbers) - 1)) if len(numbers) > 1 else 0

def gaussian_probability(x, mean, std, epsileon=1e-6):
    if std == 0:
        return 1.0 if x == mean else 0.0
    exponent = math.exp(-((x - mean) ** 2) / (2 * (std ** 2 + epsileon)))
    return (1 / (math.sqrt(2 * math.pi) * (std + epsileon))) * exponent

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

def predict(instance, parameters, hyperparameter):
    """
    학습된 Naive Bayes Classifier로 예측하는 함수
    """
    features = [float(x) for x in instance[1:]]  # date 제외
    
    # 각 클래스에 대한 로그 확률 계산
    log_prob_0 = math.log(parameters['prior_0'])
    log_prob_1 = math.log(parameters['prior_1'])
    
    if hyperparameter['epsileon'] is not None:
        epsileon = hyperparameter['epsileon']
    
    # 각 특성에 대한 확률 계산
    for i, feature in enumerate(features):
        # 클래스 0에 대한 확률
        prob_0 = gaussian_probability(feature, 
                                    parameters['mean_0'][i], 
                                    parameters['std_0'][i],
                                    epsileon)
        log_prob_0 += math.log(prob_0 + 1e-10)  # log(0) 방지
        
        # 클래스 1에 대한 확률
        prob_1 = gaussian_probability(feature, 
                                    parameters['mean_1'][i], 
                                    parameters['std_1'][i],
                                    epsileon)
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

    # precision & recall calculation (raw values)
    tp = 0
    fp = 0
    fn = 0
    for idx in range(len(predictions)):
        if predictions[idx] == 1:
            if answers[idx] == 1:
                tp += 1
            else:
                fp += 1
        elif answers[idx] == 1:  # predictions[idx] == 0 and answers[idx] == 1
            fn += 1
    
    # Calculate raw metrics
    raw_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    raw_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F1 from raw values
    f1 = 2 * (raw_precision * raw_recall) / (raw_precision + raw_recall) if (raw_precision + raw_recall) > 0 else 0
    
    # Convert to percentages for display
    precision = round(raw_precision * 100, 2)
    recall = round(raw_recall * 100, 2)
    f1 = round(f1 * 100, 2)

    logging.info("accuracy: {}%".format(accuracy))
    logging.info("precision: {}%".format(precision))
    logging.info("recall: {}%".format(recall))
    logging.info("F1-score: {}%".format(f1))

def calculate_discomfort_index(temperature, humidity):
    """불쾌지수 계산 함수"""
    return 0.81 * temperature + 0.01 * humidity * (0.99 * temperature - 14.3) + 46.3

def load_raw_data(fname, selected_features=None):
    instances = []
    labels = []
    with open(fname, "r") as f:
        headers = f.readline().strip().split(", ")
        for line in f:
            tmp = line.strip().split(", ")
            if selected_features:
                features = []
                for idx in selected_features.keys():
                    if idx == 8:  # 불쾌지수를 위한 새로운 인덱스
                        temp = float(tmp[1])  # avg temperature
                        humidity = float(tmp[4])  # avg humidity
                        di = calculate_discomfort_index(temp, humidity)
                        features.append(di)
                    elif idx in [5,6]:  # humidity max, min은 정수형
                        features.append(int(tmp[idx]))
                    else:
                        features.append(float(tmp[idx]))
                instances.append(features)
            else:  # 모든 feature 사용 (기존 방식)
                features = [float(tmp[1]), float(tmp[2]), float(tmp[3]), 
                          float(tmp[4]), int(tmp[5]), int(tmp[6]), float(tmp[7])]
                instances.append(features)
            labels.append(int(tmp[-1]))
    return instances, labels

def run_experiment(train_file, test_file, feature_set, feature_names):
    """
    특정 feature set으로 실험을 수행하는 함수
    """
    # training phase
    instances, labels = load_raw_data(train_file, feature_set)
    parameters = training(instances, labels)

    # testing phase
    instances, labels = load_raw_data(test_file, feature_set)
    predictions = []
    hyperparameter = {'epsileon': 1e-6}
    
    for instance in instances:
        result = predict(instance, parameters, hyperparameter)
        predictions.append(result)
    
    # report
    logging.info(f"\nExperiment with features: {list(feature_names.values())}")
    report(predictions, labels)

def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", required=True, metavar="<file path to the training dataset>", help="File path of the training dataset", default="training.csv")
    parser.add_argument("-u", "--testing", required=True, metavar="<file path to the testing dataset>", help="File path of the testing dataset", default="testing.csv")
    parser.add_argument("-l", "--log", help="Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)", type=str, default="INFO")
    parser.add_argument("-f", "--features", help="Feature indices to use (comma-separated). Available features: 1:avg_temp, 2:max_temp, 3:min_temp, 4:avg_humid, 5:max_humid, 6:min_humid, 7:power, 8:discomfort_index", type=str, required=True)
    
    args = parser.parse_args()
    return args

def main():
    args = command_line_args()
    logging.basicConfig(level=args.log)

    if not os.path.exists(args.training) or not os.path.exists(args.testing):
        logging.error("Dataset files do not exist")
        sys.exit(1)

    # Parse feature indices
    feature_indices = [int(x.strip()) for x in args.features.split(',')]
    feature_names = {
        1: 'avg_temperature',
        2: 'max_temperature',
        3: 'min_temperature',
        4: 'avg_humidity',
        5: 'max_humidity',
        6: 'min_humidity',
        7: 'power',
        8: 'discomfort_index'
    }
    
    selected_features = {idx: feature_names[idx] for idx in feature_indices}
    
    # 선택된 feature set으로 실험 수행
    run_experiment(args.training, args.testing, selected_features, selected_features)

if __name__ == "__main__":
    main()
