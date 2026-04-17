# 混合动机推演模式（Hybrid Motivation Inference）

## 概述

本模块实现了宏观周期先验（M_r）与短期行为惯性（M_e）的上下文自适应融合，用于生成更准确的活动轨迹。

## 核心设计思想

- **M_r 定位**：提供"基准锚点"。反映长周期规律（周末/节假日/季节/历史高相似日），适合应对分布外（OOD）场景或数据稀疏期。
- **M_e 定位**：提供"动态微调"。反映近 k 日行为偏移（如临时加班、连续出行偏好衰减），适合常态周期内的平滑过渡。
- **融合策略**：设计**情境感知门控机制（Context-Aware Gating）**，根据当前日期的`时间新颖度 `、` 近期数据密度`、` 行为波动率`动态分配权重 α ∈ [0,1]，再通过结构化 Prompt 交由 LLM 进行动机对齐与冲突消解。

## 形式化融合公式

### (1) 上下文门控向量构建
```
c_d = [近期周期偏移，近期数据完整性，短期行为波动率]^T
```

其中：
- **近期周期偏移**：最近 k 天中周末的比例
- **近期数据完整性**：近期签到记录覆盖时间窗口的比例
- **短期行为波动率**：近期空间步长距离的方差

### (2) 自适应权重计算
```
α_d = σ(w^T * c_d + b) = 1 / (1 + exp(-(w^T * c_d + b)))
```

- α_d → 1 表示高度依赖检索先验（如：近期无数据/波动大/遇节假日）
- α_d → 0 表示高度依赖演化惯性（如：数据密集/行为稳定/常规工作日）

### (3) 动机融合函数
```
M̂_d = F_LLM(M_r^(d), M_e^(d), α_d | Task)
```

其中 F_LLM 为结构化融合 Prompt，负责权重引导、逻辑对齐与冲突裁决。

## 使用方式

### 命令行方式

```bash
# 使用混合动机模式 (mode=3)
python generate.py --dataset 2019 --mode 3
```

### 前端 UI 方式

在 Streamlit 前端中，选择"混合动机"模式即可。

### 编程方式

```python
from engine.trajectory_generate import mob_gen
from engine.agent import Person

# 创建 Person 对象
P = Person(name="user_id", model=YourLLM(), person_id="user_id")
# ... 加载数据 ...

# 配置融合参数
fusion_config = {
    "k": 7,                     # 近期天数窗口
    "weight_params": None,       # 使用默认权重参数
    "use_mlp": False,            # 不使用 MLP
    "use_heuristic": False,      # 使用 LLM 融合
    "heuristic_config": None     # 启发式配置
}

# 执行轨迹生成
mob_gen(P, mode=3, scenario_tag="normal", critic_check=True, 
        g_days=14, fusion_config=fusion_config)
```

## 文件结构

```
engine/
├── motivation_fusion.py          # 混合动机融合核心模块
├── trajectory_generate.py        # 轨迹生成主逻辑（已更新支持 mode=3）
└── prompt_template/
    ├── hybrid_motivation_infer.txt  # 混合动机推断 prompt 模板
    └── prompt_paths.py              # prompt 路径配置（已更新）

generate.py                       # 主入口脚本（已更新支持 mode=3）
front/
├── page5.py                      # 前端 UI（已更新支持混合模式）
└── tools.py                      # 前端工具函数（已更新）

evaluate.py                       # 评估脚本（已更新支持 mode=3）
```

## 模式对比

| 模式 | 名称 | 描述 |
|------|------|------|
| 0 | 基于检索 (llm_l) | 使用检索到的历史相似日轨迹推断动机 |
| 1 | 基于演化 (llm_e) | 使用最近 k 天的行为惯性推断动机 |
| 2 | 无动机 (llm_nm) | 不使用动机驱动 |
| 3 | 混合动机 (llm_hybrid) | **新增** 融合检索和演化两种动机源 |

## 配置参数说明

### fusion_config 字典

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| k | int | 7 | 近期天数窗口大小 |
| weight_params | dict | None | 权重参数 {"w": [w1, w2, w3], "b": b}，None 使用默认值 |
| use_mlp | bool | False | 是否使用 MLP 计算权重 |
| mlp_weights | list | None | MLP 权重列表（每层一个数组） |
| mlp_biases | list | None | MLP 偏置列表（每层一个数组） |
| use_heuristic | bool | False | 是否使用启发式融合（不调用 LLM） |
| heuristic_config | dict | None | 启发式融合配置 |

### heuristic_config 字典（当 use_heuristic=True 时使用）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| high_alpha_threshold | float | 0.7 | 高α阈值，超过此值优先使用 M_r |
| low_alpha_threshold | float | 0.3 | 低α阈值，低于此值优先使用 M_e |

## 日志输出

混合模式会输出以下关键日志：
- `mode3 M_r_demo`: 检索到的 demo 轨迹
- `mode3 M_e_demo`: 演化使用的 demo 轨迹
- `mode3 alpha`: 计算得到的融合权重
- `mode3 M_r_motivation`: 检索动机
- `mode3 M_e_motivation`: 演化动机
- `mode3 fused_motivation`: 融合后的动机

## 注意事项

1. **数据连续性**：混合模式依赖于近期轨迹数据的连续性，如果数据不连续，门控向量的计算可能不准确。

2. **计算开销**：混合模式需要调用 LLM 三次（M_r 推断、M_e 推断、动机融合），Token 消耗约为单一模式的 3 倍。

3. **权重调优**：默认权重参数为启发式设置，如需更高精度，可在小规模验证集上训练 MLP 或使用网格搜索优化。

4. **结果保存**：混合模式的结果保存在 `./result/{scenario_tag}/generated/llm_hybrid/{person_id}/` 目录下。

## 未来优化方向

1. **MLP 权重训练**：在小规模验证集上训练 2 层 MLP 来学习最优权重参数。

2. **动态 k 值**：根据数据密度动态调整近期窗口大小 k。

3. **多源融合**：扩展支持更多动机源（如季节性、特殊事件等）。
