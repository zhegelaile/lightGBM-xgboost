import pandas as pd
import numpy as np
import lightgbm as lgb
import treelite
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import schema_config as cfg

def train_and_export_lgb():
    # 1. 加载数据与元数据
    if not os.path.exists(cfg.TRAIN_DATA_PATH):
        print("Data not found. Please run generate_data.py first.")
        return
        
    df = pd.read_csv(cfg.TRAIN_DATA_PATH)
    with open(cfg.METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print("Loaded metadata for LightGBM training.")

    # 2. 自动编码类别型输入特征
    for col in cfg.get_categorical_inputs():
        df[col] = df[col].map(metadata[col])

    # 3. 准备数据集
    X = df[cfg.get_input_names()]
    X_train, X_test = train_test_split(X, test_size=0.15, random_state=42)

    # 4. 循环训练所有目标模型
    for target in cfg.OUTPUT_TARGETS:
        name = target["name"]
        task_type = target["type"]
        y_train = df.loc[X_train.index, name]
        y_test = df.loc[X_test.index, name]

        if task_type == "regression":
            model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                importance_type='split',
                n_jobs=-1,
                random_state=42
            )
        else:
            num_class = target["num_class"]
            model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                importance_type='split',
                n_jobs=-1,
                random_state=42,
                objective="multiclass" if num_class > 2 else "binary"
            )

        model.fit(X_train, y_train)
        
        # 验证
        y_pred = model.predict(X_test)
        if task_type == "regression":
            score = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"[{name:15s}] LGBM RMSE: {score:.4f}")
        else:
            score = accuracy_score(y_test, y_pred)
            print(f"[{name:15s}] LGBM ACC:  {score:.4f}")

        # 5. 导出模型 (Bin & TXT)
        bin_path = os.path.join(cfg.BASE_DIR, f"model_{name}.bin")
        txt_path = os.path.join(cfg.BASE_DIR, f"model_{name}.txt")
        
        # Treelite 导出 (核心：保持 C++ 兼容性)
        tl_model = treelite.frontend.from_lightgbm(model.booster_)
        tl_model.serialize(bin_path)
        
        # LightGBM 原生导出
        model.booster_.save_model(txt_path)

    print("\nAll LightGBM models exported (TXT & BIN).")

if __name__ == "__main__":
    train_and_export_lgb()
