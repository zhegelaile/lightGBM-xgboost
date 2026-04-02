import os

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(BASE_DIR, "metadata_dictionary.json")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "train_data_new.csv")

# =============================================================================
# 1. 输入特征定义 (Input Features)
# =============================================================================
INPUT_FEATURES = [
    {"name": "project_type",   "type": "categorical"},
    {"name": "tool_type",      "type": "categorical"},
    {"name": "surface_attr",   "type": "categorical"},
    {"name": "tool_diameter",  "type": "numeric"},
    {"name": "tool_length",    "type": "numeric"},
    {"name": "machining_area", "type": "numeric"},
    {"name": "ref_lines",      "type": "numeric"},
]

# =============================================================================
# 2. 输出目标定义 (Output Targets) - 保持 18 个目标
# =============================================================================
OUTPUT_TARGETS = [
    {"name": "step_distance",  "type": "regression"},
    {"name": "drive_method",   "type": "classification", "num_class": 6},
    {"name": "path_process",   "type": "classification", "num_class": 2},
    {"name": "path_direction", "type": "classification", "num_class": 2},
    {"name": "step_direction", "type": "classification", "num_class": 2},
    {"name": "sorting_method", "type": "classification", "num_class": 2},
    {"name": "tool_axis_direction", "type": "classification", "num_class": 3},
    {"name": "region_sorting", "type": "classification", "num_class": 3},
    {"name": "ignore_hole_diameter", "type": "classification", "num_class": 2},
    {"name": "path_interpolation", "type": "classification", "num_class": 4},
    {"name": "link_motion_interpolation", "type": "classification", "num_class": 4},
    {"name": "entry_type", "type": "classification", "num_class": 4},
    {"name": "exit_type", "type": "classification", "num_class": 4},
    {"name": "approach_setting", "type": "classification", "num_class": 2},
    {"name": "retract_setting", "type": "classification", "num_class": 2},
    {"name": "rapid_approach_setting", "type": "classification", "num_class": 2},
    {"name": "rapid_retract_setting", "type": "classification", "num_class": 2},
    {"name": "long_short_link_threshold", "type": "regression"},
]

def get_input_names(): return [f["name"] for f in INPUT_FEATURES]
def get_target_names(): return [f["name"] for f in OUTPUT_TARGETS]
def get_categorical_inputs(): return [f["name"] for f in INPUT_FEATURES if f["type"] == "categorical"]
