import numpy as np
from parse_json import *
import matplotlib.pyplot as plt

tasks, labels = load_data("./data")
feature_names = ALL_FEATURES

feature_data = extract_features(tasks, feature_names)
feature_data_normalized = normalize_features(feature_data)

dot_size=16

FEATURE_LABELS = {
    "keypress_gap_stddev": "Keypress Gap Std Dev (ms)",
    "deletion_rate": "Deletion Rate",
    "revision_rate": "Revision Rate",
    "mean_revision_depth": "Mean Revision Depth",
    "characters_per_second": "Characters per Second",
    "proportion_spent_paused": "Proportion of Time Paused",
    "chars_per_burst": "Characters per Burst",
    "pause_len_stddev": "Pause Length Std Dev (ms)",
    "mean_punctuation_gap": "Mean Post-Punctuation Gap (ms)",
    "punctuation_gap_stddev": "Post-Punctuation Gap Std Dev (ms)",
    "mean_backspace_sequence_length": "Mean Backspace Sequence Length",
}

def plot_features(i1, i2, data=feature_data):
    colors = ['red' if label == 1 else 'blue' for label in labels]
    scatter = plt.scatter(data[:, i1], data[:, i2], c=colors, s=dot_size)
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Transcribed'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Natural'),
    ])
    plt.title(f"{FEATURE_LABELS[feature_names[i2]]} vs {FEATURE_LABELS[feature_names[i1]]}")
    plt.xlabel(FEATURE_LABELS[feature_names[i1]])
    plt.ylabel(FEATURE_LABELS[feature_names[i2]])
    plt.show()

def plot_feature_scatter(i, data=feature_data):
    colors = ['red' if label == 1 else 'blue' for label in labels]
    plt.scatter(data[:, i], labels, c=colors, s=dot_size)
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Transcribed'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Natural'),
    ])
    plt.title(FEATURE_LABELS[feature_names[i]])
    plt.xlabel(FEATURE_LABELS[feature_names[i]])
    plt.ylabel("Label")
    plt.show()

def plot_feature_histogram(i, num_bins=20, data=feature_data):
    data_0 = data[labels == 0][:, i]
    data_1 = data[labels == 1][:, i]
    
    bins = np.linspace(data[:, i].min(), data[:, i].max(), num_bins)
    
    plt.hist(data_0, bins=bins, color='blue', alpha=0.5, label='Natural')
    plt.hist(data_1, bins=bins, color='red', alpha=0.5, label='Transcribed')
    
    plt.title(f"Distribution of {FEATURE_LABELS[feature_names[i]]}")
    plt.xlabel(FEATURE_LABELS[feature_names[i]])
    plt.ylabel("Count")
    plt.legend()
    plt.show()

def plot_errors(i1, i2, data, labels, preds, features):
    colors = []
    markers = []
    legend_elements = []

    category_styles = {
        'TP': ('green', 'o'),
        'TN': ('blue', 'o'),
        'FP': ('red', 'X'),
        'FN': ('orange', 'X'),
    }

    for label, pred in zip(labels, preds):
        if label == 1 and pred == 1:
            cat = 'TP'
        elif label == 0 and pred == 0:
            cat = 'TN'
        elif label == 0 and pred == 1:
            cat = 'FP'
        else:
            cat = 'FN'
        colors.append(category_styles[cat][0])
        markers.append(category_styles[cat][1])

    for cat, (color, marker) in category_styles.items():
        legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, label=cat, markersize=8))

    for i in range(len(data)):
        plt.scatter(data[i, i1], data[i, i2], c=colors[i], marker=markers[i], s=dot_size)

    plt.legend(handles=legend_elements)
    plt.xlabel(FEATURE_LABELS[features[i1]])
    plt.ylabel(FEATURE_LABELS[features[i2]])
    plt.title(f"{FEATURE_LABELS[features[i2]]} vs {FEATURE_LABELS[features[i1]]}")
    plt.show()