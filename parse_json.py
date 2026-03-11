from pathlib import Path
import json
import numpy as np

ALL_FEATURES = [
    "keypress_gap_stddev",
    "deletion_rate",
    "revision_rate",
    "mean_revision_depth",
    "characters_per_second",
    "proportion_spent_paused",
    "chars_per_burst",
    "pause_len_stddev",
    "mean_punctuation_gap",
    "punctuation_gap_stddev",
    "mean_backspace_sequence_length"
]

MIN_PAUSE_LENGTH_MS = 1500
REVISION_TOLERANCE = 2
BACKSPACE_SEQUENCE_TOLERANCE_MS = 500

PUNCTUATION_KEYS = [
    ".",
    ",",
    "?",
    "!",
    ":",
    ";",
]

class Task:
    def __init__(self, task_data, id):
        self.id = id
        self.data = task_data
        self.keystrokes = task_data["keystrokes"]
        self.type = task_data["type"]

        self.char_strokes = [s for s in self.keystrokes if self.adds_char(s)]
        self.down_strokes = [s for s in self.keystrokes if s["type"] == "keydown"]
        
        timestamps = np.array([s["t"] for s in self.char_strokes])
        self.gaps_ms = np.diff(timestamps)
        self.punctuation_gaps = self.get_punctuation_gaps()

    def get_num_pauses(self):
        return np.size(self.gaps_ms[self.gaps_ms > MIN_PAUSE_LENGTH_MS])

    def adds_char(self, stroke):
        if stroke["type"] == "keyup":
            return False
        if len(stroke["key"]) != 1:
            return False
        return True

    def get_keypress_gap_stddev_ms(self):
        return np.std(self.gaps_ms) if len(self.gaps_ms) > 0 else 0.0

    def get_deletion_rate(self):
        chars_added = len(self.char_strokes)
        chars_deleted = sum(1 for s in self.keystrokes if s["type"] != "keyup" and s["code"] == "Backspace")

        if chars_added == 0:
            print("TERRIBLE THINGS HAVE HAPPENED")
            return 0
        
        return chars_deleted / chars_added

    def get_revision_rate(self):
        chars_total = len(self.char_strokes)
        revisions = 0

        for stroke in self.keystrokes:
            if stroke["type"] == "keydown" and stroke["text_length"] - stroke["cursor"] > REVISION_TOLERANCE:
                revisions += 1

        return revisions / chars_total if chars_total > 0 else 0
    
    def get_mean_revision_depth(self):
        revision_depths = []

        for stroke in self.keystrokes:
            if stroke["type"] == "keydown" and stroke["text_length"] - stroke["cursor"] > REVISION_TOLERANCE:
                revision_depths.append(stroke["text_length"] - stroke["cursor"])

        return sum(revision_depths) / len(revision_depths) if len(revision_depths) > 0 else 0

    def get_characters_per_second(self):
        chars_added = len(self.char_strokes)
        if chars_added == 0 or len(self.keystrokes) == 0:
            return 0

        last_char_time_ms = self.char_strokes[-1]["t"]

        total_time_s = (last_char_time_ms - self.keystrokes[0]["t"]) / 1000
        return chars_added / total_time_s if total_time_s > 0 else 0

    def get_proportion_spent_paused(self):
        time_paused = np.sum(self.gaps_ms[self.gaps_ms > MIN_PAUSE_LENGTH_MS])

        last_char_time_ms = self.char_strokes[len(self.char_strokes) - 1]["t"]

        total_time = (last_char_time_ms - self.keystrokes[0]["t"])

        return time_paused / total_time if total_time > 0 else 0
    
    def get_chars_per_burst(self):
        pause_inds = np.array(np.where(self.gaps_ms > MIN_PAUSE_LENGTH_MS)[0])
        pause_inds = np.append(pause_inds, len(self.char_strokes))
        if pause_inds[0] != 0:
            pause_inds = np.insert(pause_inds, 0, 0)
        return np.mean(np.diff(pause_inds))

    def get_pause_len_stddev(self):
        return np.std(self.gaps_ms[self.gaps_ms > MIN_PAUSE_LENGTH_MS]) if self.get_num_pauses() > 1 else 0
    
    def get_punctuation_gaps(self):
        punctuation_inds = [i for i in range(1, len(self.keystrokes) - 1) if self.adds_char(self.keystrokes[i]) and self.keystrokes[i]["key"] in PUNCTUATION_KEYS and self.keystrokes[i+1]["key"] not in PUNCTUATION_KEYS and self.keystrokes[i+1]["code"] != "Backspace"]
        gaps = []
        for i in range(len(punctuation_inds)):
            keystroke_ind = punctuation_inds[i]
            next_char_ind = keystroke_ind + 1

            while next_char_ind < len(self.keystrokes) and (
                not self.adds_char(self.keystrokes[next_char_ind]) or 
                self.keystrokes[next_char_ind]["key"] == " " or 
                self.keystrokes[next_char_ind]["key"] in PUNCTUATION_KEYS
            ):
                next_char_ind += 1

            if(next_char_ind == len(self.keystrokes)): break

            gaps.append(self.keystrokes[next_char_ind]["t"] - self.keystrokes[keystroke_ind]["t"])
        
        return gaps
    
    def get_mean_punctuation_gap(self):
        return sum(self.punctuation_gaps) / len(self.punctuation_gaps) if len(self.punctuation_gaps) > 0 else 0

    def get_punctuation_gap_stddev(self):
        return np.std(np.array(self.punctuation_gaps)) if len(self.punctuation_gaps) > 0 else 0
            
    def get_mean_backspace_sequence_length(self):
        backspace_inds = [i for i in range(len(self.down_strokes)) if self.down_strokes[i]["code"] == "Backspace"]
        sequence_lens=[]
        cur_sequence_len = 1
        for i in range(1, len(backspace_inds)):
            if backspace_inds[i] - backspace_inds[i-1] == 1 or (self.down_strokes[backspace_inds[i]]["t"] - self.down_strokes[backspace_inds[i-1]]["t"]) < BACKSPACE_SEQUENCE_TOLERANCE_MS:
                cur_sequence_len += 1
            else:
                sequence_lens.append(cur_sequence_len)
                cur_sequence_len = 0

        if cur_sequence_len > 0: sequence_lens.append(cur_sequence_len)
        return sum(sequence_lens) / len(sequence_lens) if len(sequence_lens) > 0 else 0



def load_data(data_path_string):
    data_path = Path(data_path_string)
    tasks = []
    labels = []
    for file_path in data_path.iterdir():
        if file_path.is_file():
            with open(file_path, 'r') as file:
                cur_file_json = json.load(file)
                cur_id = cur_file_json["participant_id"]
                for task_data in cur_file_json["tasks"]:
                    task = Task(task_data, cur_id)
                    tasks.append(task)
                    if task.type == "transcription":
                        labels.append(1)
                    else:
                        labels.append(0)

    return tasks, np.array(labels)

def extract_features(tasks, features=ALL_FEATURES):
    feature_functions = {
        "keypress_gap_stddev": "get_keypress_gap_stddev_ms",
        "deletion_rate": "get_deletion_rate",
        "revision_rate": "get_revision_rate",
        "mean_revision_depth": "get_mean_revision_depth",
        "characters_per_second": "get_characters_per_second",
        "proportion_spent_paused": "get_proportion_spent_paused",
        "chars_per_burst": "get_chars_per_burst",
        "pause_len_stddev": "get_pause_len_stddev",
        "mean_punctuation_gap": "get_mean_punctuation_gap",
        "punctuation_gap_stddev": "get_punctuation_gap_stddev",
        "mean_backspace_sequence_length": "get_mean_backspace_sequence_length"
    }

    m = len(tasks)
    k = len(features)
    feature_data = np.zeros((m, k))
    
    for feature_num in range(k):
        feature = features[feature_num]
        feature_fn_name = feature_functions.get(feature)

        if not feature_fn_name:
            raise RuntimeError(f"Requested feature '{feature}' not found")

        for task_num in range(m):
            task = tasks[task_num]
            feature_fn = getattr(task, feature_fn_name)
            feature_data[task_num][feature_num] = feature_fn()
            
    return feature_data
        
def normalize_features(feature_data):
    data_normalized = feature_data.copy()
    for i in range(data_normalized.shape[1]):
        data_normalized[:, i] -= np.mean(data_normalized[:, i])
        if(sigma := np.std(data_normalized[:, i])):
            data_normalized[:, i] /= sigma 
    return data_normalized