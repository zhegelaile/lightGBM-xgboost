# CAM 智能加工参数推荐系统 (CAM Smart Recommender)

本项目为基于 C++ 的桌面端 CAM 软件提供智能参数推荐功能。采用 Python 端的 XGBoost 训练方案，并通过 Treelite 实现跨平台、高性能的 C++ 部署。

## 1. 整体架构 (Architecture)
- **数据层**: 处理 1000+ 条历史加工数据，包含混合型输入。
- **训练层**: 6 个独立任务（XGBRegressor + XGBClassifier），任务解耦。
- **桥接层**: `metadata_dictionary.json` 存储特征映射关系，是 Python 与 C++ 的共同语言。
- **部署层**: Treelite 序列化模型 -> 编译为原生 C++ 动态链接库 (.dll/.so)。

## 2. 文件功能清单 (File Manifest)

| 文件名 | 类型 | 功能说明 |
| :--- | :--- | :--- |
| `GEMINI.md` | 文档 | 核心需求、输入特征规范、输出目标定义的“最高指令”。 |
| `README.md` | 文档 | 项目整体原理、功能介绍及使用指南。 |
| `generate_data.py` | 脚本 | **数据生成器**。包含物理规律的模拟数据，确保模型能学习到真实的关联关系（如：步距与刀具直径成正比）。 |
| `train_xgboost.py` | 核心 | **训练中心**。执行 Ordinal Encoding、保存映射字典、训练 6 个独立模型、导出 Treelite 二进制文件及 JSON 模型。 |
| `verify_xgboost.py` | 验证 | **端到端测试**。演示如何从人类可读参数（字符串）到特征编码，再到推理结果解码的全过程。 |
| `metadata_dictionary.json` | 配置 | **核心元数据**。定义了字符串枚举（如：加工类型、刀具类型）与整数之间的映射关系。 |
| `model_*.bin` | 权重 | **Treelite 序列化模型**。用于 C++ 端 Treelite C API 的推理加载。 |
| `model_*.json` | 权重 | **XGBoost 原生模型**。用于 Python 端的快速验证与再训练。 |

## 3. 运行指南 (Getting Started)

### 第一步：安装依赖 (Dependencies)
```bash
pip install xgboost treelite treelite_runtime pandas numpy scikit-learn
```

### 第二步：准备数据 (Data Preparation)
运行脚本生成带有物理逻辑的模拟数据（用于开发测试）：
```bash
python generate_data.py
```

### 第三步：训练模型 (Model Training)
执行训练脚本，会自动生成映射字典和模型文件：
```bash
python train_xgboost.py
```

### 第四步：本地验证 (Verification)
修改 `verify_xgboost.py` 中的输入参数并运行，观察模型预测是否符合工业常识：
```bash
python verify_xgboost.py
```

## 4. 核心设计原则 (Design Principles)
1. **禁止 One-Hot 编码**: 针对 10 类刀具，C++ 端只需要输入一个 float 整数，而非 10 个维度的稀疏向量。
2. **任务解耦**: 步距 (Regression) 与排序方式 (Classification) 独立训练，提升精度。
3. **物理安全网 (Fallback)**: C++ 端在调用推理结果后，必须通过传统边界检查，确保预测值不超出机床物理极限。
