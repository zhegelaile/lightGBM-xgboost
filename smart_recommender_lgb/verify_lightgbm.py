import json
import lightgbm as lgb
import numpy as np
import os
import schema_config as cfg

def verify_lightgbm():
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
    
    # 3. 编码特征向量
    input_vector = []
    for feature in cfg.INPUT_FEATURES:
        name = feature["name"]
        val = test_input.get(name)
        if feature["type"] == "categorical":
            input_vector.append(meta[name][val])
        else:
            input_vector.append(float(val))
    
    input_data = np.array([input_vector])
    
    print("--- LightGBM 推理验证 (Python 端) ---")
    print("-" * 30)

    # 4. 推理所有目标
    for target in cfg.OUTPUT_TARGETS:
        name = target["name"]
        task_type = target["type"]
        txt_path = os.path.join(cfg.BASE_DIR, f"model_{name}.txt")
        
        if not os.path.exists(txt_path): continue

        # 加载 LightGBM Booster
        bst = lgb.Booster(model_file=txt_path)
        
        # 对于分类任务，lgb.Booster.predict 默认返回概率或 raw score
        # 这里的 predict 返回的是概率数组
        pred_prob = bst.predict(input_data)
        
        if task_type == "regression":
            val = float(pred_prob[0])
            print(f"[{name:15s}]: {val:.4f} (Float)")
        else:
            # 获取概率最大的索引
            if len(pred_prob.shape) > 1 and pred_prob.shape[1] > 1:
                idx = int(np.argmax(pred_prob[0]))
            else:
                # 二分类返回单列概率
                idx = int(pred_prob[0] > 0.5)
                
            reverse_map = {v: k for k, v in meta[f"target_{name}"].items()}
            label = reverse_map.get(idx, f"ID:{idx}")
            print(f"[{name:15s}]: {idx} (Enum: {label})")

    print("-" * 30)
    print("LightGBM 验证完成。请使用生成的 .bin 文件部署到 C++ 端。")

if __name__ == "__main__":
    verify_lightgbm()
