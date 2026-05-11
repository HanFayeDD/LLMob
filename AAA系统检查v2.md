## 基于大语言模型API的个性化城市居民活动轨迹生成与分析
220111011 付龙飞 
指导老师：余剑峤

### 一.系统概述
本系统实现了一个耦合模式锚定、动机驱动与反思校验的活动轨迹生成架构，
核心目标是对城市居民日常活动轨迹进行建模、生成与多维评价。
主要函数入口包含以下三个部分：
- `generate.py`活动轨迹生成，包含模式锚定（P）、动机驱动（M）与反思校验（C）三部分
- `evaluate.py`活动轨迹评估，包含四个客观评估指标与一个主观评估指标
- `main.py`基于streamlit搭建的可视化面板函数入口

| 活动轨迹生成与评估模块 | 对应代码模块 | 核心函数/类 |
|---|---|---|
| 活动模式的识别与提取 | [engine/persona_identify.py](engine/persona_identify.py) | `identify()`, `score_from_rating()` |
| 基于演化的动机驱动 | [engine/trajectory_generate.py](engine/trajectory_generate.py) | `mob_gen()` |
| 基于检索的动机驱动 | [engine/utilities/retrieval_helper.py](engine/utilities/retrieval_helper.py) | `TemporalRetriever`, `DeepModel` |
| 混合动机推演 | [engine/motivation_fusion.py](engine/motivation_fusion.py) | `build_context_gating_vector()`, `compute_adaptive_weight()` |
| 客观评估指标 | [evaluate.py](evaluate.py) | `Evaluation` 类 |
| 主观评估指标 | [evaluate.py](evaluate.py) | `llm_as_judge_preference_rate()` |

### 二、核心模块解析
#### 2.1 数据加载流程
入口 [generate.py](generate.py#L78) 第 78-98 行 加载每个用户的 `.pkl` 数据文件，包含：
- `train_routine_list`：用于训练的历史轨迹
- `test_routine_list`：待生成的测试轨迹
- `attribute`：用户个性化属性（Persona），初始为None。在活动模式的识别与提取阶段进行填充
- `domain_knowledge`：领域知识（通勤模式、偏好等）
- `activity_area`/`area_freq`：活动区域及频率

#### 2.2 活动轨迹生成
##### 2.2.1 活动模式的识别与提取
在候选角色集合中，依据用户的历史轨迹选出拟定候职业以及个性化模式描述

##### 2.2.2 动机驱动的活动轨迹生成
- 基于演化的动机驱动方式
  基于最近连续几天的轨迹推断短期行为惯性动机。
- 基于检索的动机驱动方式
  基于TemporalRetriever 检索最相似日期的轨迹，推断宏观周期性动机。涉及到轻量化多层感知机的训练。
- 混合动机推演
  实现在 [engine/trajectory_generate.py](engine/trajectory_generate.py#L142) 第 142-218 行：
  - 检索动机 M_r 计算：基于 `TemporalRetriever` 检索最相似日期的轨迹，推断宏观周期性动机。
  - 演化动机 M_e 计算：基于最近一天的轨迹推断短期行为惯性动机。
  - 自适应权重计算：
    - `build_context_gating_vector()`（第 20-126 行）构建三维上下文门控向量
    - `compute_adaptive_weight()`（第 154-199 行）通过 sigmoid 函数计算自适应权重 α ∈ [0,1]
      - α → 1：高度依赖检索先验（如近期无数据、波动大、遇节假日）
      - α → 0：高度依赖演化惯性（如数据密集、行为稳定、常规工作日）
  - 动机融合：
    基于固定阈值进行动机的选择或者融合

##### 2.2.3 硬性与软性校验结合的反思机制
[engine/trajectory_generate.py](engine/trajectory_generate.py#L246) 第 246-333 行实现了自主反思循环。主要实现思路为：硬性与软性校验相结合的级联架构。
1. **硬约束检查**（时间地点）（第 285-289 行）：
   - `valid_place_return()`：检查地点是否在推荐区域中
   - `valid_time_return()`：检查时间格式是否合法
2. **软约束检查**（第 299-307 行）：
   只有当开始critic且硬约束检查通过时才进行软约束检查
   - `semantic_critic()`（第 14-37 行）：使用 LLM 检查轨迹的语义逻辑合理性
3. **错误反馈与重试**（第 318-333 行）：当检查不通过时，生成反思指令，引导 LLM 自我修正
**数据流图：**
```
Persona + 历史轨迹 → 动机推断 → 自适应融合(M_r + M_e) 
    → Prompt构建 → LLM生成 → [硬约束检查] 
    → [语义Critic检查] → 通过 → 保存结果
                   ↓ 失败
              反思指令 → 重新生成(最多5次)
```

#### 2.3 评估模块
评估模块包含四个客观评估指标和一个主观评估指标。
##### 2.3.1 客观评估指标
1. SD (Spatial Distance, [distance_one_step](evaluate.py#L223))
   - 含义: 评估轨迹中相邻点之间地理距离的分布相似性。
   - 计算过程:
     对每个轨迹，计算相邻点之间的距离（使用 geodistance 函数，基于经纬度计算球面距离，单位为公里）。将距离值分箱（bins=11，范围 0-10km），生成直方图分布。使用 JSD 比较真实数据和生成数据的距离分布。
2. SI (Spatial-temporal Interval, [duration_jsd](evaluate.py#L244))
   - 含义: 评估轨迹中相邻点之间时间间隔（持续时间）的分布相似性。
   - 计算过程:
     对每个轨迹，计算相邻点的时间差（分钟），乘以10并四舍五入（duration 函数）。 将时间间隔分箱（bins=12，范围 0-12），生成直方图分布。使用 JSD 比较真实数据和生成数据的时间间隔分布。
3. DARD (Dynamic Activity Representation Distance, [st_act_jsd_v2](evaluate.py#L296))
   - 含义: 评估时间-活动类型的联合分布相似性。
   - 计算过程:为每个轨迹点创建键：str(时间间隔) + '_' + str(活动ID)（活动ID 通过 p2id 映射）。将所有键映射到唯一ID，生成分布（bins=1000，归一化到 0-1）。使用 JSD 比较真实数据和生成数据的联合分布。
4. STVD (Spatial-temporal Variation Distance, [st_loc_jsd_v3](evaluate.py#L438))
   - 含义: 评估时间-地理位置的联合分布相似性。
   - 计算过程:为每个轨迹点创建键：str(时间间隔) + '_' + str(纬度) + '_' + str(经度)。将所有键映射到唯一ID，生成分布（bins=400，归一化到 0-1）。使用 JSD 比较真实数据和生成数据的联合分布。
##### 2.3.2 主观评估指标
基于llm-as-a-judge计算偏好率。
输入是生成轨迹、真实轨迹以及用户模型，输出是llm认为哪条轨迹更符合当前日期以及用户模型的偏好。
偏好率即为所有轨迹中，llm认为生成轨迹更好的占比。

### 三、代码系统设计
- LLM API封装与继承设计
  - 

- [限流器](engine\utilities\api_limiter.py#L8)
  滑动窗口，通过`max_calls`以及`period`两个参数进行限流，使用装饰器。
