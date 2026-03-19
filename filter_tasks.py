import sys
import os
import tty
import termios
from parse_json import load_data

FILTERED_IDS_PATH = "filtered_ids.txt"
REJECTED_IDS_PATH = "rejected_ids.txt"


def load_ids(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_ids(ids, path):
    with open(path, "w") as f:
        for id_ in sorted(ids):
            f.write(id_ + "\n")


def get_keypress():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def review_tasks(data_path="./data", filtered_ids_path=FILTERED_IDS_PATH, rejected_ids_path=REJECTED_IDS_PATH):
    tasks, labels = load_data(data_path)
    accepted = load_ids(filtered_ids_path)
    rejected = load_ids(rejected_ids_path)

    natural_tasks = [
        (task, label) for task, label in zip(tasks, labels)
        if task.type == "natural" and task.id not in accepted and task.id not in rejected
    ]

    if not natural_tasks:
        print("No new free-writing tasks to review.")
        return

    decisions = {}  # index -> True (accept) / False (reject)

    i = 0
    while i < len(natural_tasks):
        task, label = natural_tasks[i]
        os.system("clear")

        print(f"Free-writing task {i + 1} / {len(natural_tasks)}")
        print(f"Participant ID : {task.id}")
        print()
        print(f"  {task.data['final_text']}")
        print()
        print(f"  [Enter] or [y]  accept")
        print(f"  [n]             reject")
        print(f"  [b]             go back")
        print(f"  [q]             quit and save")
        print()

        key = get_keypress()

        if key in ("\r", "\n", "y", "Y"):
            decisions[i] = True
            print(f"  ✓ Accepted")
            i += 1
        elif key in ("n", "N"):
            decisions[i] = False
            print(f"  ✗ Rejected")
            i += 1
        elif key in ("b", "B"):
            if i > 0:
                i -= 1
            else:
                print("  Already at the first task.")
        elif key in ("q", "Q"):
            break

    for idx, decision in decisions.items():
        task = natural_tasks[idx][0]
        if decision:
            accepted.add(task.id)
        else:
            rejected.add(task.id)

    save_ids(accepted, filtered_ids_path)
    save_ids(rejected, rejected_ids_path)
    print(f"\nSaved {len(accepted)} accepted ID(s) to {filtered_ids_path}")
    print(f"Saved {len(rejected)} rejected ID(s) to {rejected_ids_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "./data"
    review_tasks(data_path)