import pandas as pd
import numpy as np
import xgboost as xgb
import treelite
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import schema_config as cfg

def train_and_export():
    # 1. 加载数据与元数据
    df = pd.read_csv(cfg.TRAIN_DATA_PATH)
    with open(cfg.METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print("Loaded metadata from schema.")

    # 2. 自动编码类别型输入特征
    for col in cfg.get_categorical_inputs():
        df[col] = df[col].map(metadata[col])

    # 3. 准备数据集
    X = df[cfg.get_input_names()]
    X_train, X_test = train_test_split(X, test_size=0.15, random_state=42)

    # 4. 循环训练所有目标模型 (Drove by Schema)
    for target in cfg.OUTPUT_TARGETS:
        name = target["name"]
        task_type = target["type"]
        y_train = df.loc[X_train.index, name]
        y_test = df.loc[X_test.index, name]

        if task_type == "regression":
            model = xgb.XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=8, subsample=0.8,
                colsample_bytree=0.8, objective="reg:squarederror", n_jobs=-1
            )
        else:
            num_class = target["num_class"]
            model = xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=8, subsample=0.8,
                colsample_bytree=0.8, n_jobs=-1,
                objective="multi:softprob" if num_class > 2 else "binary:logistic",
                num_class=num_class if num_class > 2 else None
            )

        model.fit(X_train, y_train)
        
        # 验证与性能输出
        y_pred = model.predict(X_test)
        if task_type == "regression":
            score = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"[{name:15s}] Regression RMSE: {score:.4f}")
        else:
            score = accuracy_score(y_test, y_pred)
            print(f"[{name:15s}] Classifier ACC:  {score:.4f}")

        # 5. 导出模型 (Bin & JSON)
        bin_path = os.path.join(cfg.BASE_DIR, f"model_{name}.bin")
        json_path = os.path.join(cfg.BASE_DIR, f"model_{name}.json")
        
        # Treelite
        tl_model = treelite.frontend.from_xgboost(model.get_booster())
        tl_model.serialize(bin_path)
        # XGBoost Native
        model.save_model(json_path)

    print("\nTraining and all models exported to script directory.")

if __name__ == "__main__":
    train_and_export()
