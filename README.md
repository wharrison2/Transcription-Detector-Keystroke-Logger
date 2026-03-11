# Keystroke-Based Transcription Detector

A machine learning classifier that distinguishes naturally written text from transcribed (or pasted) text using keystroke dynamics. Motivated by the limitations of existing AI detection tools, which are often inaccurate and biased against non-native English speakers.

## How It Works

Participants complete two ~50-word typing tasks in a browser — one free writing task and one transcription task — in randomized order. Keystroke timings, cursor positions, and final text are logged for each task. Statistical features are extracted from the logs and used to train a binary classifier.

## Features

| Feature | Description |
|---|---|
| Proportion of Time Paused | Fraction of task time spent in pauses (>1500ms) |
| Deletion Rate | Ratio of characters deleted to characters added |
| Characters per Burst | Mean characters typed between pauses |
| Characters per Second | Overall typing speed |
| Pause Length Std Dev | Variability in pause durations |
| Keypress Gap Std Dev | Variability in inter-keypress intervals |
| Post-Punctuation Gap (mean/std dev) | Gaps between punctuation and next word |
| Revision Rate | Rate of mid-text edits |
| Mean Revision Depth | How far back in the text edits occur |
| Mean Backspace Sequence Length | Length of consecutive deletion runs |

## Results

Trained on ~240 tasks (evenly split between transcribed and natural) using logistic regression with Monte Carlo cross-validation (5000 trials).

The strongest individual predictors were proportion of time paused (73% accuracy alone) and deletion rate (69%). A greedy forward feature selection procedure consistently selected: **proportion spent paused, deletion rate, characters per second, characters per burst, and pause length standard deviation**.

| | Predicted Natural | Predicted Transcribed |
|---|---|---|
| **Actual Natural** | 0.404 | 0.213 |
| **Actual Transcribed** | 0.021 | 0.362 |

- **Train accuracy:** 79%
- **Validation accuracy:** 77%
- **Test accuracy:** 77%

Most errors are false positives — naturally written responses that resemble transcription in their pacing and deletion patterns (e.g. rushed or carelessly written text).

## Models

Three classifiers were evaluated:
- **Logistic Regression** (scikit-learn) — primary model, C=1.0
- **Neural Network** (TensorFlow)
- **Gradient Boosting** (XGBoost)

All three achieved similar validation and test accuracy. Deep learning and XGBoost showed overfitting on the current dataset size.

## Setup

```bash
pip install scikit-learn xgboost tensorflow numpy matplotlib
```

Place participant JSON files in a `data/` directory, then:

```python
from parse_json import load_data, extract_features, normalize_features

tasks, labels = load_data("./data")
feature_data = extract_features(tasks)
```

## Data Collection

Open `index.html` in a browser (or serve it statically). Update the `SUBMIT_URL` variable at the top of the script to point to your backend. The FastAPI backend is not included in this repository.

