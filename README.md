# CAM 智能加工参数推荐系统 (CAM Smart Recommender)

本项目是一个基于机器学习的辅助制造 (CAM) 参数推荐系统。它通过分析历史加工数据，自动推荐切削步距、排序方式、进退刀设置等核心参数，旨在提高工艺工程师的工作效率并降低工艺不确定性。

## 🌟 项目亮点
- **双引擎支持**: 提供基于 **XGBoost** 和 **LightGBM** 的两种训练方案，可根据精度和性能需求灵活选择。
- **跨平台 C++ 部署**: 模型训练完成后可导出为二进制格式，支持在 C++ 桌面端应用（如 NX 插件、独立 CAM 软件）中高性能调用。
- **物理规律模拟**: 内置 `generate_data.py` 脚本，可生成符合机械加工物理逻辑的模拟数据，用于模型原型开发与验证。
- **解耦设计**: 针对不同类型的参数（分类 vs 回归）建立独立模型，确保推荐精度。

## 📂 目录结构
- `smart_recommender/`: 基于 **XGBoost** 的实现方案。
- `smart_recommender_lgb/`: 基于 **LightGBM** 的实现方案。
- `metadata_dictionary.json`: 定义了字符串参数（如刀具类型、加工方式）与模型输入特征之间的映射关系，是 Python 与 C++ 的“共同语言”。

## 🛠️ 环境准备
建议使用 Python 3.8+ 环境，安装以下依赖：
```bash
pip install xgboost lightgbm pandas numpy scikit-learn treelite treelite_runtime
```

## 🚀 快速开始

### 1. 生成训练数据
进入对应目录，生成带有物理逻辑的模拟数据集：
```bash
python generate_data.py
```

### 2. 训练模型
执行训练脚本。程序会自动处理特征编码、训练多个独立任务模型，并导出权重文件：
- **XGBoost 版本**: `python train_xgboost.py`
- **LightGBM 版本**: `python train_lightgbm.py`

### 3. 模型验证
运行验证脚本，模拟从“人类可读参数”到“模型预测”的完整流程：
- `python verify_xgboost.py` 或 `python verify_lightgbm.py`

## 🔧 核心推荐任务
系统当前支持以下 6 类关键参数的智能推荐：
1. **步距 (Step Distance)**: 回归模型，根据刀具直径和加工余量预测。
2. **切削方向 (Path Direction)**: 分类模型（顺铣/逆铣）。
3. **区域排序 (Region Sorting)**: 分类模型（层优先/深度优先）。
4. **进刀方式 (Entry Type)**: 分类模型（螺旋/斜插/直接）。
5. **退刀方式 (Exit Type)**: 分类模型。
6. **连接运动 (Link Motion)**: 分类模型。

## 📦 部署说明 (C++ Integration)
1. **依赖库**: 部署在 C++ 端所需的 Treelite 库已包含在 `deps/treelite-4.7.0.rar` 中。请解压该文件并按照库的文档进行配置。
2. **模型导出**: 训练脚本会生成 `.bin` (Treelite) 或 `.txt` (LGBM) 文件。
3. **C++ 调用**: 推荐使用 Treelite C API 或 LightGBM C++ API 加载模型。
4. **预处理**: C++ 端需读取 `metadata_dictionary.json`，确保输入特征的编码与训练时完全一致。

---
*本项目由 Gemini CLI 协助构建与维护。*
