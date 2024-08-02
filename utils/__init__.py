from utils.plot_utils import *

HINT_MAPPING = {
    "none": "No Hint",
    "testcases": "Test Cases",
    "conceptualhint": "Conceptual",
    "detailedfix": "Detailed Fix",
}
TASK_NUM_TO_NAME = {
    "1": "Sum Positives",
    "2": "Count Pos Neg",
    "3": "Average Rainfall",
    "4": "Palindrome",
    "5": "Fibonacci",
}

HINT_ORDER = list(HINT_MAPPING.values())
TASK_ORDER = list(TASK_NUM_TO_NAME.values())
STIMULUS_ORDER = ["Python", "Text"]
PART_ORDER = ["Confused", "Clear"]
