import tensorflow as tf
import numpy as np
from parse_json import *
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import KFold, cross_val_score
from random import shuffle
from xgboost import XGBClassifier
from utils import *
from visualize_feature_data import *

# tasks, labels = load_data("./data")
tasks, labels = get_filtered_data()
feature_names = ALL_FEATURES

feature_data = extract_features(tasks, feature_names)
feature_data_normalized = normalize_features(feature_data)


feature_choice = ['proportion_spent_paused', 'deletion_rate', 'characters_per_second', 'chars_per_burst', 'pause_len_stddev']
minimal_feature_choice = ['proportion_spent_paused', 'deletion_rate']
train_tasks, test_tasks, train_labels, test_labels = split_tasks(tasks, labels, p_train=0.8)

tester = TestSuite(train_tasks, train_labels)

print(tester.get_correlation_matrix(features=['deletion_rate', 'proportion_spent_paused', 'characters_per_second', 'chars_per_burst', 'pause_len_stddev', 'mean_punctuation_gap', 'punctuation_gap_stddev', 'mean_backspace_sequence_length']))

# accuracies = []
# regs = list(np.arange(0.1, 1.0, 0.01))
# for reg in regs:
#     accuracies.append(tester.get_avg_performance_neural(features=feature_choice, dropout=reg, show_plot=False, folds=10))

# val_accuracies = [a[0] for a in accuracies]
# train_accuracies = [a[1] for a in accuracies]

# plt.plot(regs, val_accuracies, marker='o', label='Validation')
# plt.plot(regs, train_accuracies, marker='o', label='Train')
# plt.xlabel(r"$\gamma$")
# plt.ylabel("Accuracy")
# plt.title("Neural Regression Accuracy vs Regularization")
# plt.legend()
# plt.show()



# baseline_full = tester.get_avg_performance_logistic(show_plot=False)
# baseline = baseline_full[0]

# feats_by_predictive_power = tester.get_features_by_predictive_power()
# for row in feats_by_predictive_power:
#     print(row)

# n = len(feats_by_predictive_power)
# for i in range(n):
#     features = [f[0] for f in feats_by_predictive_power[n - i - 1:]]
#     performance = tester.get_avg_performance_logistic(features=features, show_plot=False)[0]
#     print(f"{features}: {performance}, from baseline: {performance - baseline}")

# print("\n\n\n")

# improvements = []
# for feature in ALL_FEATURES:
#     without_feature = [f for f in ALL_FEATURES if f != feature]
#     performance = tester.get_avg_performance_logistic(features=without_feature, show_plot=False)[0]
#     print(f"without {feature}: {performance}, from baseline: {performance - baseline}")
#     improvements.append([feature, performance-baseline])

# improvements = sorted(improvements, key=lambda x: x[1])
# for row in improvements:
#     print(row)

# my_custom_feature_choice = ['proportion_spent_paused', 'punctuation_gap_stddev', 'deletion_rate']
# print("\n\n\n")


# threshold = 0.01
# custom_feature_choice = [f[0] for f in improvements if f[1] < -0.01]

# print(custom_feature_choice)

# next_custom_feature_choice = custom_feature_choice
# found_improvement = True
# while(found_improvement):
#     found_improvement = False
#     cur_performance = tester.get_avg_performance_logistic(features=custom_feature_choice, show_plot=False)[0]
#     print(f"New baseline: {cur_performance}")
#     for feature in ALL_FEATURES:
#         if feature in custom_feature_choice: continue
#         performance = tester.get_avg_performance_logistic(features=custom_feature_choice + [feature], folds=4000, show_plot=False)[0]
#         if performance - cur_performance > threshold:
#             next_custom_feature_choice.append(feature)
#             found_improvement = True
#             print(f"adding {feature} yields {performance}, improvement of {performance-cur_performance}")
#     custom_feature_choice = next_custom_feature_choice
#     print(custom_feature_choice)

# print("\n\nNot selected:")
# print([f for f in ALL_FEATURES if f not in custom_feature_choice])

# accuracies = []
# for c in np.arange(0.1, 10.1, 0.1):
#     accuracies.append(tester.get_avg_performance_logistic(features=feature_choice, C=c, show_plot=False, folds=1000))

# val_accuracies = [a[0] for a in accuracies]
# train_accuracies = [a[1] for a in accuracies]

# plt.plot(np.arange(0.1, 10.1, 0.1), val_accuracies, marker='o', label='Validation')
# plt.plot(np.arange(0.1, 10.1, 0.1), train_accuracies, marker='o', label='Train')
# plt.xlabel("C")
# plt.ylabel("Accuracy")
# plt.title("Logistic Regression Accuracy vs Regularization (C)")
# plt.legend()
# plt.show()

custom_feature_choice=['deletion_rate', 'proportion_spent_paused', 'characters_per_second', 'chars_per_burst', 'pause_len_stddev', 'mean_punctuation_gap', 'punctuation_gap_stddev', 'mean_backspace_sequence_length']

print(f"automatically selected custom_choice: {custom_feature_choice}")
print(f"custom: {tester.get_avg_performance_logistic(features=custom_feature_choice, show_plot=False)}\nbaseline {tester.get_avg_performance_logistic(show_plot=False)}")
print(f"custom: {tester.get_avg_performance_neural(features=custom_feature_choice, show_plot=False)}\nbaseline {tester.get_avg_performance_neural(show_plot=False)}")
print(f"custom: {tester.get_avg_performance_xg(features=custom_feature_choice, show_plot=False)}\nbaseline {tester.get_avg_performance_xg(show_plot=False)}")
print(f"custom: {tester.get_avg_performance_rf(features=custom_feature_choice, show_plot=False)}\nbaseline {tester.get_avg_performance_xg(show_plot=False)}")

print("Confusion matrices\n\n")
print(f"custom: {tester.get_avg_confusion_logistic(features=custom_feature_choice)}")
print(f"custom: {tester.get_avg_confusion_neural(features=custom_feature_choice)}")
print(f"custom: {tester.get_avg_confusion_xg(features=custom_feature_choice)}")
print(f"custom: {tester.get_avg_confusion_rf(features=custom_feature_choice)}")


# print("\nMy custom choice")
# print(f"custom: {tester.get_avg_performance_logistic(features=my_custom_feature_choice, show_plot=False)}\nbaseline {tester.get_avg_performance_logistic(show_plot=False)}")
# print(f"custom: {tester.get_avg_performance_neural(features=my_custom_feature_choice, show_plot=False)}\nbaseline {tester.get_avg_performance_neural(show_plot=False)}")
# print(f"custom: {tester.get_avg_performance_xg(features=my_custom_feature_choice, show_plot=False)}\nbaseline {tester.get_avg_performance_xg(show_plot=False)}")
# print(f"custom: {tester.get_avg_performance_rf(features=my_custom_feature_choice, show_plot=False)}\nbaseline {tester.get_avg_performance_xg(show_plot=False)}")

# print("\nMinimal choice")
# print(f"custom: {tester.get_avg_performance_logistic(features=['proportion_spent_paused', 'deletion_rate'], show_plot=False)}\nbaseline {tester.get_avg_performance_logistic(show_plot=False)}")
# print(f"custom: {tester.get_avg_performance_neural(features=['proportion_spent_paused', 'deletion_rate'], show_plot=False)}\nbaseline {tester.get_avg_performance_neural(show_plot=False)}")
# print(f"custom: {tester.get_avg_performance_xg(features=['proportion_spent_paused', 'deletion_rate'], show_plot=False)}\nbaseline {tester.get_avg_performance_xg(show_plot=False)}")
# print(f"custom: {tester.get_avg_performance_rf(features=['proportion_spent_paused', 'deletion_rate'], show_plot=False)}\nbaseline {tester.get_avg_performance_xg(show_plot=False)}")


train_tasks, test_tasks, train_labels, test_labels = split_tasks(tasks, labels, p_train=0.8)

train_data = normalize_features(extract_features(train_tasks, features=custom_feature_choice))
test_data = extract_features(test_tasks, features=custom_feature_choice)
test_data_normalized = normalize_features(test_data)

model = linear_model.LogisticRegression(penalty='l2', C=1.0)
model.fit(train_data, train_labels)

preds = model.predict(test_data_normalized)

fp_ids = [test_tasks[i].id for i in range(len(test_tasks)) if preds[i] == 1 and test_labels[i] == 0]
print(fp_ids)

tp = np.sum((preds == 1) & (test_labels == 1))
tn = np.sum((preds == 0) & (test_labels == 0))
fp = np.sum((preds == 1) & (test_labels == 0))
fn = np.sum((preds == 0) & (test_labels == 1))
total = tp + fp + tn + fn

print([tp/total, tn/total, fp/total, fn/total])
print(f"Accuracy: {model.score(test_data_normalized, test_labels):.3f}")

plot_errors(0, 1, test_data, test_labels, preds, feature_choice)