import json
import xgboost as xgb
import numpy as np
import os
import schema_config as cfg

def verify_xgboost():
    # 1. 加载元数据
    with open(cfg.METADATA_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    # 2. 模拟输入 (Test input)
    test_input = {
        "project_type": "PolishProject",
        "tool_type": "END_MILL",
        "surface_attr": "Plane_NoHole_Shallow",
        "tool_diameter": 10.0,
        "tool_length": 50.0,
        "machining_area": 3000.0,
        "ref_lines": 2
    }
    
    # 3. 自动根据 Schema 编码特征向量
    input_vector = []
    for feature in cfg.INPUT_FEATURES:
        name = feature["name"]
        val = test_input.get(name)
        
        if feature["type"] == "categorical":
            # 类别型: 查表映射
            input_vector.append(meta[name][val])
        else:
            # 数值型: 直接添加
            input_vector.append(float(val))
    
    input_data = np.array([input_vector])
    
    print("--- 自动化 Schema 推理验证 ---")
    print("-" * 30)

    # 4. 自动根据 Schema 推理所有目标
    for target in cfg.OUTPUT_TARGETS:
        name = target["name"]
        task_type = target["type"]
        model_path = os.path.join(cfg.BASE_DIR, f"model_{name}.json")
        
        if not os.path.exists(model_path): continue

        if task_type == "regression":
            bst = xgb.XGBRegressor()
        else:
            bst = xgb.XGBClassifier()
            
        bst.load_model(model_path)
        pred = bst.predict(input_data)
        
        # 5. 解码输出
        if task_type == "regression":
            print(f"[{name:15s}]: {float(pred[0]):.4f} (Float)")
        else:
            idx = int(pred[0])
            # 从元数据反向解码
            reverse_map = {v: k for k, v in meta[f"target_{name}"].items()}
            label = reverse_map.get(idx, f"ID:{idx}")
            print(f"[{name:15s}]: {idx} (Enum: {label})")

    print("-" * 30)
    print("验证完成。")

if __name__ == "__main__":
    verify_xgboost()
