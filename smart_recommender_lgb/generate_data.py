import pandas as pd
import numpy as np
import json

def generate_logical_data(n_samples=2000):
    np.random.seed(42)
    
    # --- 业务枚举 ---
    project_types = ['PolishProject', 'OutLineProject', 'PlaneMillProject', 'HighRoughProject', 
                     'FreePlaneMillProject', 'SidePlaneMillProject', 'HighShapeProject']
    tool_types = ['END_MILL', 'BALL_KNIFE', 'TAPERED_KNIFE', 'NOSE_KNIFE', 'T_KNIFE', 'R_KNIFE']
    
    surface_attrs = [
        'Plane_NoHole_Shallow', 'Plane_NoHole_Steep', 'Plane_HasHole_Shallow', 'Plane_HasHole_Steep',
        'Periodic_NoHole_Shallow', 'Periodic_NoHole_Steep', 'Periodic_HasHole_Shallow', 'Periodic_HasHole_Steep',
        'MicroCurved_NoHole_Shallow', 'MicroCurved_NoHole_Steep', 'MicroCurved_HasHole_Shallow', 'MicroCurved_HasHole_Steep'
    ]
    
    # --- 输出目标枚举 ---
    drive_methods = [f'drive_{i}' for i in range(6)]
    path_processes = ['one_way', 'zigzag']
    path_directions = ['climb', 'conventional']
    step_directions = ['near_to_far', 'far_to_near']
    sorting_methods = ['direction_first', 'step_first']
    
    # 新增枚举
    tool_axis_directions = ['axis_0', 'axis_1', 'axis_2']
    region_sortings = ['region_0', 'region_1', 'region_2']
    bool_labels = ['False', 'True']
    interpolation_types = ['interp_0', 'interp_1', 'interp_2', 'interp_3']
    entry_exit_types = ['type_0', 'type_1', 'type_2', 'type_3']

    # 生成并保存元数据
    metadata = {
        "project_type": {val: i for i, val in enumerate(project_types)},
        "tool_type": {val: i for i, val in enumerate(tool_types)},
        "surface_attr": {val: i for i, val in enumerate(surface_attrs)},
        "target_drive_method": {val: i for i, val in enumerate(drive_methods)},
        "target_path_process": {val: i for i, val in enumerate(path_processes)},
        "target_path_direction": {val: i for i, val in enumerate(path_directions)},
        "target_step_direction": {val: i for i, val in enumerate(step_directions)},
        "target_sorting_method": {val: i for i, val in enumerate(sorting_methods)},
        "target_tool_axis_direction": {val: i for i, val in enumerate(tool_axis_directions)},
        "target_region_sorting": {val: i for i, val in enumerate(region_sortings)},
        "target_ignore_hole_diameter": {val: i for i, val in enumerate(bool_labels)},
        "target_path_interpolation": {val: i for i, val in enumerate(interpolation_types)},
        "target_link_motion_interpolation": {val: i for i, val in enumerate(interpolation_types)},
        "target_entry_type": {val: i for i, val in enumerate(entry_exit_types)},
        "target_exit_type": {val: i for i, val in enumerate(entry_exit_types)},
        "target_approach_setting": {val: i for i, val in enumerate(bool_labels)},
        "target_retract_setting": {val: i for i, val in enumerate(bool_labels)},
        "target_rapid_approach_setting": {val: i for i, val in enumerate(bool_labels)},
        "target_rapid_retract_setting": {val: i for i, val in enumerate(bool_labels)},
    }
    with open("metadata_dictionary.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    # --- 生成特征 ---
    proj_idx = np.random.randint(0, len(project_types), n_samples)
    tool_idx = np.random.randint(0, len(tool_types), n_samples)
    surf_idx = np.random.randint(0, len(surface_attrs), n_samples)
    tool_diameter = np.random.uniform(2.0, 20.0, n_samples)
    tool_length = np.random.uniform(10.0, 100.0, n_samples)
    machining_area = np.random.uniform(100.0, 5000.0, n_samples)
    ref_lines = np.random.randint(0, 3, n_samples)

    # --- 建立逻辑关联 ---

    # 1. step_distance
    step_distance = tool_diameter * (0.4 + 0.05 * surf_idx) + (proj_idx * 0.2) + np.random.normal(0, 0.05, n_samples)
    
    # 2. drive_method
    drive_idx = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if proj_idx[i] <= 1: drive_idx[i] = np.random.choice([0, 1])
        elif proj_idx[i] == 3: drive_idx[i] = 3
        elif proj_idx[i] == 6: drive_idx[i] = 5
        else: drive_idx[i] = np.random.choice([2, 4])

    # 3. path_process
    path_process_idx = (machining_area > 2000).astype(int)

    # 4. path_direction
    path_direction_idx = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if tool_idx[i] == 1: # BALL_KNIFE
            path_direction_idx[i] = np.random.choice([0, 1], p=[0.3, 0.7])
        else:
            path_direction_idx[i] = np.random.choice([0, 1], p=[0.9, 0.1])

    # 5. step_direction: 选取 Steep 类型 (索引 1, 3, 5, 7, 9, 11)
    step_direction_idx = (surf_idx % 2 == 1).astype(int)

    # 6. sorting_method
    sorting_method_idx = (ref_lines >= 1).astype(int)

    # 7. tool_axis_direction: 依赖于工程类型
    tool_axis_idx = (proj_idx % 3)

    # 8. region_sorting: 依赖于加工面积
    region_sorting_idx = np.where(machining_area < 1500, 0, np.where(machining_area < 3500, 1, 2))

    # 9. ignore_hole_diameter: 选取 HasHole 类型 (索引 2, 3, 6, 7, 10, 11)
    has_hole_indices = [2, 3, 6, 7, 10, 11]
    ignore_hole_idx = (np.isin(surf_idx, has_hole_indices)).astype(int)

    # 10 & 11. interpolation: 依赖于工具直径
    path_interp_idx = (tool_diameter // 5).clip(0, 3).astype(int)
    link_interp_idx = (tool_diameter // 6).clip(0, 3).astype(int)

    # 12 & 13. entry/exit: 依赖于工具类型
    entry_type_idx = (tool_idx % 4)
    exit_type_idx = ((tool_idx + 1) % 4)

    # 14-17. settings: 简单逻辑
    approach_idx = (proj_idx > 3).astype(int)
    retract_idx = (proj_idx > 3).astype(int)
    rapid_approach_idx = (machining_area > 3000).astype(int)
    rapid_retract_idx = (machining_area > 3000).astype(int)

    # 18. long_short_link_threshold: 依赖于工具长度
    threshold = (tool_length * 0.5) + np.random.normal(0, 1, n_samples)
    threshold = np.maximum(threshold, 0.1)

    data = {
        'project_type': [project_types[i] for i in proj_idx],
        'tool_type': [tool_types[i] for i in tool_idx],
        'surface_attr': [surface_attrs[i] for i in surf_idx],
        'tool_diameter': tool_diameter,
        'tool_length': tool_length,
        'machining_area': machining_area,
        'ref_lines': ref_lines,
        
        'step_distance': step_distance,
        'drive_method': drive_idx,
        'path_process': path_process_idx,
        'path_direction': path_direction_idx,
        'step_direction': step_direction_idx,
        'sorting_method': sorting_method_idx,
        
        'tool_axis_direction': tool_axis_idx,
        'region_sorting': region_sorting_idx,
        'ignore_hole_diameter': ignore_hole_idx,
        'path_interpolation': path_interp_idx,
        'link_motion_interpolation': link_interp_idx,
        'entry_type': entry_type_idx,
        'exit_type': exit_type_idx,
        'approach_setting': approach_idx,
        'retract_setting': retract_idx,
        'rapid_approach_setting': rapid_approach_idx,
        'rapid_retract_setting': rapid_retract_idx,
        'long_short_link_threshold': threshold
    }
    
    df = pd.DataFrame(data)
    df.to_csv('train_data_new.csv', index=False)
    print(f"Generated {n_samples} samples with updated surface attributes.")

if __name__ == "__main__":
    generate_logical_data()
