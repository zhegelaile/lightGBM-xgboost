# 业务数据字典与特征规范 (Data Schema)

## 1. 输入特征 (Input Features) - 共 7 个维度
字符串枚举特征保持原始字符串形式，但在模型训练前通过 **Ordinal Encoding** 转换为从 0 开始的整数。

- **工程类型 (Project Type)**:
    - `PolishProject` = 0
    - `OutLineProject` = 1
    - `PlaneMillProject` = 2
    - `HighRoughProject` = 3
    - `FreePlaneMillProject` = 4
    - `SidePlaneMillProject` = 5
    - `HighShapeProject` = 6
- **工具类型 (Tool Type)**:
    - `END_MILL` = 0
    - `BALL_KNIFE` = 1
    - `TAPERED_KNIFE` = 2
    - `NOSE_KNIFE` = 3
    - `T_KNIFE` = 4
    - `R_KNIFE` = 5
- **加工面属性 (Surface Attribute)**: 
    - `Plane_NoHole_Shallow` = 0
    - `Plane_NoHole_Steep` = 1
    - `Plane_HasHole_Shallow` = 2
    - `Plane_HasHole_Steep` = 3
    - `Periodic_NoHole_Shallow` = 4
    - `Periodic_NoHole_Steep` = 5
    - `Periodic_HasHole_Shallow` = 6
    - `Periodic_HasHole_Steep` = 7
    - `MicroCurved_NoHole_Shallow` = 8
    - `MicroCurved_NoHole_Steep` = 9
    - `MicroCurved_HasHole_Shallow` = 10
    - `MicroCurved_HasHole_Steep` = 11
- **连续数值特征**: `tool_diameter`, `tool_length`, `machining_area` (均为 float)
- **离散整数特征**: `ref_lines` (取值为 `0, 1, 2`)

## 2. 输出目标 (Output Targets) - 共 6 个独立任务
分类任务在 CSV 中直接存储为 **Ordinal Encoding (从 0 开始的整数)**。

- **目标 1：加工步距 (Step Distance)** -> 连续值回归
- **目标 2：驱动方式 (Drive Method)** -> 6 分类 (`0 ~ 5`)
- **目标 3：轨迹工艺 (Path Process)** -> 2 分类 (`0, 1`)
- **目标 4：轨迹方向 (Path Direction)** -> 2 分类 (`0, 1`)
- **目标 5：步进方向 (Step Direction)** -> 2 分类 (`0, 1`)
- **目标 6：排序方式 (Sorting Method)** -> 2 分类 (`0, 1`)
- **目标 7：刀轴方向 (Tool Axis Direction)** -> 3 分类 (`0 ~ 2`)
- **目标 8：区域排序 (Region Sorting)** -> 3 分类 (`0 ~ 2`)
- **目标 9：忽略孔径 (Ignore Hole Diameter)** -> 2 分类 (`0, 1`, Bool)
- **目标 10：加工路径插补 (Path Interpolation)** -> 4 分类 (`0 ~ 3`)
- **目标 11：链接运动插补 (Link Motion Interpolation)** -> 4 分类 (`0 ~ 3`)
- **目标 12：切入类型 (Entry Type)** -> 4 分类 (`0 ~ 3`)
- **目标 13：切出类型 (Exit Type)** -> 4 分类 (`0 ~ 3`)
- **目标 14：接近设置 (Approach Setting)** -> 2 分类 (`0, 1`, Bool)
- **目标 15：离开设置 (Retract Setting)** -> 2 分类 (`0, 1`, Bool)
- **目标 16：快速接近设置 (Rapid Approach Setting)** -> 2 分类 (`0, 1`, Bool)
- **目标 17：快速离开设置 (Rapid Retract Setting)** -> 2 分类 (`0, 1`, Bool)
- **目标 18：长短链接阈值 (Long Short Link Threshold)** -> 连续值回归 (Float > 0)

## 3. C++ 数据结构与推理约束
- **输入映射**: `CamSmartRecommender` 负责根据 `metadata_dictionary.json` 将 C++ 业务层传入的字符串枚举（如 `"PolishProject"`) 转换为模型可识别的 `float`。
- **模型文件**: 每个目标对应一个 `.bin` (Treelite) 和 `.json` (XGBoost) 文件。
