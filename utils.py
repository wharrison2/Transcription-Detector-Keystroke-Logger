import tensorflow as tf
import numpy as np
from parse_json import *
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import KFold, cross_val_score
from random import shuffle
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import os

FILTERED_IDS_PATH = "filtered_ids.txt"

def get_filtered_data(data_path="./data", filtered_ids_path=FILTERED_IDS_PATH):
    """
    Returns (tasks, labels) containing:
      - natural tasks whose participant IDs are in filtered_ids.txt (label 0)
      - all transcription tasks (label 1)
    """
    if os.path.exists(filtered_ids_path):
        with open(filtered_ids_path, "r") as f:
            filtered_ids = set(line.strip() for line in f if line.strip())
    else:
        filtered_ids = set()

    all_tasks, all_labels = load_data(data_path)

    tasks = []
    labels = []
    for t, l in zip(all_tasks, all_labels):
        if t.type == "transcription" or (t.type == "natural" and t.id in filtered_ids):
            tasks.append(t)
            labels.append(l)

    return tasks, np.array(labels)

def split_tasks(tasks, labels, p_train=0.8):
    indices = list(range(len(tasks)))
    shuffle(indices)

    num_train = int(len(tasks) * p_train)
    
    train_idx = indices[:num_train]
    test_idx = indices[num_train:]
    
    train_tasks = [tasks[i] for i in train_idx]
    train_labels = np.array([labels[i] for i in train_idx])
    
    test_tasks = [tasks[i] for i in test_idx]
    test_labels = np.array([labels[i] for i in test_idx])
    
    return train_tasks, test_tasks, train_labels, test_labels

def split_data(feature_data, labels, p_train=0.8):
    indices = np.arange(feature_data.shape[0])
    np.random.shuffle(indices)

    num_train = int(feature_data.shape[0] * p_train)
    
    train_idx = indices[:num_train]
    test_idx = indices[num_train:]
    
    train_data = feature_data[train_idx]
    train_labels = labels[train_idx]
    
    test_data = feature_data[test_idx]
    test_labels = labels[test_idx]
    
    return train_data, test_data, train_labels, test_labels

def plot(test_accuracies, train_accuracies, show_train):
    bins = np.histogram(train_accuracies, bins=128)[1]
    if(show_train):
        bins = np.histogram(np.hstack((train_accuracies, test_accuracies)), bins=128)[1]
        plt.hist(train_accuracies, bins=bins, label="Train set accuracy", color="red", alpha=0.5)
    plt.hist(test_accuracies, bins=bins, label="Test set accuracy", color="blue", alpha=0.5)

    plt.show()

class TestSuite():
    def __init__(self, tasks, labels):
        self.tasks, self.labels = tasks, labels

    def get_avg_performance_logistic(self, features=ALL_FEATURES, C=1.0, folds=5000, p_train=0.75, show_plot=True, show_train=True, return_lists=False):
        feature_data = extract_features(self.tasks, features=features)
        feature_data_normalized = normalize_features(feature_data)
        train_accuracies = []
        accuracies = []

        for _ in range(folds):
            train_set, test_set, train_labels, test_labels = split_data(feature_data_normalized, self.labels, p_train=p_train)

            logistic = linear_model.LogisticRegression(penalty='l2', C=C)
            logistic.fit(train_set, train_labels)

            accuracies.append(logistic.score(test_set, test_labels))
            train_accuracies.append(logistic.score(train_set, train_labels))

        if show_plot:
            plot(accuracies, train_accuracies, show_train)

        return np.mean(accuracies), np.mean(train_accuracies)
    
    def get_features_by_predictive_power(self, features=ALL_FEATURES, C=1.0, folds=500, p_train=0.75):
        features_w_importances = sorted([[feature, self.get_avg_performance_logistic(features=[feature], C=C, folds=folds, p_train=p_train, show_plot=False)[0]] for feature in features], key=lambda x: x[1])
        return features_w_importances

    def get_avg_performance_neural(self, features=ALL_FEATURES, folds=10, epochs=300, p_train=0.75, shape=[8, 4], reg_param=0.01, reg_type='l2', dropout=0.3, show_plot=True, show_train=True):
        feature_data = extract_features(self.tasks, features)
        feature_data_normalized = normalize_features(feature_data)
        train_accuracies = []
        accuracies = []
        
        reg = tf.keras.regularizers.l2(0)

        if reg_type == 'l2':
            reg = tf.keras.regularizers.l2(reg_param)
        elif reg_type == 'l1':
            reg = tf.keras.regularizers.l1(reg_param)

        for _ in range(folds):
            train_set, test_set, train_labels, test_labels = split_data(feature_data_normalized, self.labels, p_train=p_train)

            layers = [tf.keras.layers.Dropout(rate=dropout)]
            for units in shape:
                layers.append(tf.keras.layers.Dense(units=units, activation="relu", kernel_regularizer=reg))
            layers.append(tf.keras.layers.Dense(units=1, activation="sigmoid", kernel_regularizer=reg))

            neural = tf.keras.Sequential(layers)

            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

            neural.compile(optimizer="adam", loss=loss_fn, metrics=['accuracy'])

            neural.fit(train_set, train_labels, epochs=epochs, verbose=0)
            accuracies.append(neural.evaluate(test_set, test_labels, verbose=0, return_dict=True)['accuracy'])
            train_accuracies.append(neural.evaluate(train_set, train_labels, verbose=0, return_dict=True)['accuracy'])
        
        if show_plot:
            plot(accuracies, train_accuracies, show_train)

        return np.mean(accuracies), np.mean(train_accuracies)


    def get_avg_performance_xg(self, features=ALL_FEATURES, ests=3, lr=0.3, d=2, gamma=0.5, folds=500, p_train=0.75, subsam=0.8, show_plot=True, show_train=True):
        feature_data = extract_features(self.tasks, features)
        feature_data_normalized = normalize_features(feature_data)
        train_accuracies = []
        accuracies = []

        for i in range(folds):
            train_set, test_set, train_labels, test_labels = split_data(feature_data_normalized, self.labels, p_train=p_train)

            bst = XGBClassifier(n_estimators=ests, max_depth=d, learning_rate=lr, gamma=gamma, objective='binary:logistic', subsample=subsam, reg_alpha=0.1)
            bst.fit(train_set, train_labels)

            preds = bst.predict(test_set)

            scores = [1 if preds[i] == test_labels[i] else 0 for i in range(len(preds))]
            accuracy = sum(scores)/len(preds)

            train_preds=bst.predict(train_set)
            train_scores = [1 if train_preds[i] == train_labels[i] else 0 for i in range(len(train_preds))]
            train_accuracy = sum(train_scores)/len(train_preds)
            train_accuracies.append(train_accuracy)
            accuracies.append(accuracy)

        if(show_plot):
            plot(accuracies, train_accuracies, show_train)

        return np.mean(accuracies), np.mean(train_accuracies)

    def get_feature_importance_xg(self, ests=5, lr=0.3, d=3, features=ALL_FEATURES, folds=500, p_train=0.75, show_plot=True, show_train=True):
        feature_data = extract_features(self.tasks, features)
        feature_data_normalized = normalize_features(feature_data)

        feature_importances = [0] * len(features)
        for i in range(folds):
            train_set, test_set, train_labels, test_labels = split_data(feature_data_normalized, labels, p_train=0.7)

            bst = XGBClassifier(n_estimators=ests, max_depth=3, learning_rate=0.3, objective='binary:logistic')
            bst.fit(train_set, train_labels)

            feature_importances += bst.feature_importances_

        features_w_importances = sorted([[features[i], feature_importances[i]] for i in range(len(features))], key=lambda x: x[1])
        return features_w_importances
    
    def get_avg_confusion_logistic(self, features=ALL_FEATURES, C=1.0, folds=500, p_train=0.75):
        feature_data = extract_features(self.tasks, features=features)
        feature_data_normalized = normalize_features(feature_data)
        tp, tn, fp, fn = 0, 0, 0, 0
        for _ in range(folds):
            train_set, test_set, train_labels, test_labels = split_data(feature_data_normalized, self.labels, p_train=p_train)
            logistic = linear_model.LogisticRegression(penalty='l2', C=C)
            logistic.fit(train_set, train_labels)
            preds = logistic.predict(test_set)
            tp += np.sum((preds == 1) & (test_labels == 1))
            tn += np.sum((preds == 0) & (test_labels == 0))
            fp += np.sum((preds == 1) & (test_labels == 0))
            fn += np.sum((preds == 0) & (test_labels == 1))
        total = tp + tn + fp + fn
        return tp/total, tn/total, fp/total, fn/total

    def get_avg_confusion_neural(self, features=ALL_FEATURES, folds=10, epochs=300, p_train=0.75, shape=[30, 10], reg_param=0.01, reg_type='l2'):
        feature_data = extract_features(self.tasks, features)
        feature_data_normalized = normalize_features(feature_data)
        tp, tn, fp, fn = 0, 0, 0, 0
        reg = tf.keras.regularizers.l2(reg_param) if reg_type == 'l2' else tf.keras.regularizers.l1(reg_param)
        for _ in range(folds):
            train_set, test_set, train_labels, test_labels = split_data(feature_data_normalized, self.labels, p_train=p_train)
            layers = [tf.keras.layers.Dense(units=shape[0], activation="relu", input_shape=(len(features),), kernel_regularizer=reg)]
            for units in shape[1:]:
                layers.append(tf.keras.layers.Dense(units=units, activation="relu", kernel_regularizer=reg))
            layers.append(tf.keras.layers.Dense(units=1, activation="sigmoid", kernel_regularizer=reg))
            neural = tf.keras.Sequential(layers)
            neural.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
            neural.fit(train_set, train_labels, epochs=epochs, batch_size=32, verbose=0)
            preds = (neural.predict(test_set, verbose=0) > 0.5).astype(int).flatten()
            tp += np.sum((preds == 1) & (test_labels == 1))
            tn += np.sum((preds == 0) & (test_labels == 0))
            fp += np.sum((preds == 1) & (test_labels == 0))
            fn += np.sum((preds == 0) & (test_labels == 1))
        total = tp + tn + fp + fn
        return tp/total, tn/total, fp/total, fn/total

    def get_avg_confusion_xg(self, features=ALL_FEATURES, ests=5, lr=0.3, d=3, folds=500, p_train=0.75):
        feature_data = extract_features(self.tasks, features)
        feature_data_normalized = normalize_features(feature_data)
        tp, tn, fp, fn = 0, 0, 0, 0
        for _ in range(folds):
            train_set, test_set, train_labels, test_labels = split_data(feature_data_normalized, self.labels, p_train=p_train)
            bst = XGBClassifier(n_estimators=ests, max_depth=d, learning_rate=lr, objective='binary:logistic', subsample=0.8)
            bst.fit(train_set, train_labels)
            preds = bst.predict(test_set)
            tp += np.sum((preds == 1) & (test_labels == 1))
            tn += np.sum((preds == 0) & (test_labels == 0))
            fp += np.sum((preds == 1) & (test_labels == 0))
            fn += np.sum((preds == 0) & (test_labels == 1))
        total = tp + tn + fp + fn
        return tp/total, tn/total, fp/total, fn/total

    def get_avg_performance_rf(self, features=ALL_FEATURES, n_estimators=100, max_depth=None, 
                                min_samples_split=5, max_features='sqrt', folds=100, 
                                p_train=0.75, show_plot=True, show_train=True):
        feature_data = extract_features(self.tasks, features)
        feature_data_normalized = normalize_features(feature_data)
        train_accuracies = []
        accuracies = []

        for _ in range(folds):
            train_set, test_set, train_labels, test_labels = split_data(feature_data_normalized, self.labels, p_train=p_train)

            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                max_features=max_features
            )
            rf.fit(train_set, train_labels)

            preds = rf.predict(test_set)
            train_preds = rf.predict(train_set)

            accuracies.append(np.mean(preds == test_labels))
            train_accuracies.append(np.mean(train_preds == train_labels))

        if show_plot:
            plot(accuracies, train_accuracies, show_train)

        return np.mean(accuracies), np.mean(train_accuracies)


    def get_avg_confusion_rf(self, features=ALL_FEATURES, n_estimators=100, max_depth=None,
                            min_samples_split=2, max_features='sqrt', folds=500, p_train=0.75):
        feature_data = extract_features(self.tasks, features)
        feature_data_normalized = normalize_features(feature_data)
        tp, tn, fp, fn = 0, 0, 0, 0

        for _ in range(folds):
            train_set, test_set, train_labels, test_labels = split_data(feature_data_normalized, self.labels, p_train=p_train)

            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                max_features=max_features
            )
            rf.fit(train_set, train_labels)

            preds = rf.predict(test_set)
            tp += np.sum((preds == 1) & (test_labels == 1))
            tn += np.sum((preds == 0) & (test_labels == 0))
            fp += np.sum((preds == 1) & (test_labels == 0))
            fn += np.sum((preds == 0) & (test_labels == 1))

        total = tp + tn + fp + fn
        return tp/total, tn/total, fp/total, fn/total
    
    def get_correlation_matrix(self, features=ALL_FEATURES):
        feature_data = extract_features(self.tasks, features)
        feature_data_normalized = normalize_features(feature_data)
        
        corr_matrix = np.corrcoef(feature_data_normalized, rowvar=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        
        ax.set_xticks(range(len(features)))
        ax.set_yticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.set_yticklabels(features)
        
        for i in range(len(features)):
            for j in range(len(features)):
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', fontsize=7)
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        return corr_matrix