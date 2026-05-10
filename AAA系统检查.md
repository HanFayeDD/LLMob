# 代码/系统检查展示报告

## 一、系统概述

本系统实现了一个**动机驱动的个体移动轨迹生成与评估框架**，核心目标是对城市居民日常活动轨迹进行建模、生成与多维评价。

### 论文创新点与代码对应关系

| 论文创新点 | 对应代码模块 | 核心函数/类 |
|---|---|---|
| 个性化角色建模（Persona） | [engine/persona_identify.py](engine/persona_identify.py) | `identify()`, `score_from_rating()` |
| 时间感知的轨迹检索 | [engine/utilities/retrieval_helper.py](engine/utilities/retrieval_helper.py) | `TemporalRetriever`, `DeepModel` |
| 动机驱动的轨迹生成 | [engine/trajectory_generate.py](engine/trajectory_generate.py) | `mob_gen()` |
| 混合动机推演（Hybrid） | [engine/motivation_fusion.py](engine/motivation_fusion.py) | `build_context_gating_vector()`, `compute_adaptive_weight()` |
| 多维评估指标 | [evaluate.py](evaluate.py) | `Evaluation` 类 |
| LLM 语义评估 | [evaluate.py](evaluate.py) | `llm_as_judge_one_day()`, `llm_as_judge_preference_rate()` |

---

## 二、核心模块解析

### 模块一：轨迹生成模块（[generate.py](generate.py) → [engine/trajectory_generate.py](engine/trajectory_generate.py)）

#### 2.1 数据加载流程

入口 [generate.py](generate.py) 第 78-98 行 加载每个用户的 `.pkl` 数据文件，包含：
- `train_routine_list`：历史训练轨迹
- `test_routine_list`：待生成的测试轨迹
- `attribute`：用户个性化属性（Persona）
- `domain_knowledge`：领域知识（通勤模式、偏好等）
- `activity_area`/`area_freq`：活动区域及频率

#### 2.2 四种生成模式

[engine/trajectory_generate.py](engine/trajectory_generate.py) 第 43-382 行 的 `mob_gen()` 函数支持四种动机推断模式：

| mode | 名称 | 动机来源 | 说明 |
|---|---|---|---|
| 0 | 基于检索（Retrieval） | 历史同期最相似日期 | 利用 `TemporalRetriever` 找到日期特征最相似的历史轨迹 |
| 1 | 基于演化（Evolving） | 近期行为惯性 | 使用最近一天的轨迹作为参考 |
| 2 | 无动机（No Motivation） | 无 | 直接使用历史轨迹作为 baseline |
| **3** | **混合动机（Hybrid）** | **检索 + 演化 + 自适应融合** | **本论文核心创新** |

#### 2.3 混合动机推演（Mode 3）详解

这是论文的核心创新，实现在 [engine/trajectory_generate.py](engine/trajectory_generate.py) 第 142-218 行：

**Step 1 - 检索动机 M_r 计算**（第 144-148 行）：使用 mode 0 的 prompt 模板，基于 `TemporalRetriever` 检索最相似日期的轨迹，推断宏观周期性动机。

**Step 2 - 演化动机 M_e 计算**（第 149-152 行）：使用 mode 1 的 prompt 模板，基于最近一天的轨迹推断短期行为惯性动机。

**Step 3 - 自适应权重计算**（[engine/motivation_fusion.py](engine/motivation_fusion.py) 第 16-199 行）：
- `build_context_gating_vector()`（第 20-126 行）构建三维上下文门控向量：
  1. **近期周期偏移**：最近 k 天中周末占比
  2. **近期数据完整性**：签到记录覆盖时间窗口的比例
  3. **短期行为波动率**：空间步长距离方差
- `compute_adaptive_weight()`（第 154-199 行）通过 sigmoid 函数计算自适应权重 α ∈ [0,1]
  - α → 1：高度依赖检索先验（如近期无数据、波动大、遇节假日）
  - α → 0：高度依赖演化惯性（如数据密集、行为稳定、常规工作日）

**Step 4 - 动机融合**（第 186-205 行）：支持两种融合策略：
- **LLM 驱动融合**（[engine/motivation_fusion.py](engine/motivation_fusion.py) 第 202-254 行）：通过结构化 prompt 引导 LLM 分析两个动机源的冲突与对齐
- **启发式融合**（第 257-293 行）：基于阈值快速融合，不调用 LLM

#### 2.4 轨迹生成与 Reflexion 校正机制

[engine/trajectory_generate.py](engine/trajectory_generate.py) 第 246-333 行 实现了自主反思循环：

1. **硬约束检查**（第 285-289 行）：
   - `valid_place_return()`：检查地点是否在推荐区域中
   - `valid_time_return()`：检查时间格式是否合法
2. **软约束检查**（第 299-307 行）：
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

---

### 模块二：评估模块（[evaluate.py](evaluate.py)）

#### 3.1 四大核心评估指标

`Evaluation` 类（[evaluate.py](evaluate.py) 第 199-662 行）实现了四个基于 JSD（Jensen-Shannon Divergence）的评估指标：

**① SD（空间距离分布）** - [evaluate.py 第 223-242 行](evaluate.py#L223-L242)
- 计算轨迹中相邻点的球面距离（Haversine 公式，[evaluate.py 第 63-70 行](evaluate.py#L63-L70)）
- 分箱范围 0-10km，计算真实轨迹与生成轨迹的距离分布 JSD

**② SI（停留间隔分布）** - [evaluate.py 第 244-263 行](evaluate.py#L244-L263)
- 计算相邻点的时间间隔（分钟）
- 分箱范围 0-240 分钟（步长 10 分钟），比较分布 JSD

**③ DARD（时空-活动联合分布）** - [evaluate.py 第 296-369 行](evaluate.py#L296-L369)
- 为每个轨迹点创建键：`时间间隔_活动类型ID`
- 归一化到 0-1，1000 个 bins，计算 JSD
- 包含 Top-10 高频时空活动模式的可视化输出

**④ STVD（时空-位置联合分布）** - [evaluate.py 第 400-522 行](evaluate.py#L400-L522)
- 为每个轨迹点创建键：`时间间隔_纬度_经度`
- 支持 3 种精度版本（v1 精确匹配、v2 排序优化、v3 浮点精度控制）
- 400 个 bins，计算 JSD

#### 3.2 LLM as Judge

- **逐日评分**（[evaluate.py 第 664-678 行](evaluate.py#L664-L678)）：使用 `llm_as_judge_one_day()` 让 LLM 对每条生成轨迹打分
- **偏好率评估**（[evaluate.py 第 681-727 行](evaluate.py#L681-L727)）：使用 `llm_as_judge_preference_rate()` 比较真实轨迹与生成轨迹哪个更符合用户个性化特征

#### 3.3 可视化辅助

每个指标计算过程中自动生成 KDE 分布图和分箱柱状图（[evaluate.py 第 524-637 行](evaluate.py#L524-L637)），方便直观对比生成轨迹与真实轨迹的分布差异。

---

### 三、系统演示流程（基于 [main.py](main.py)）

#### 演示路径设计

**Step 1 → 原始数据集展示（Page 1）**

打开 [front/page1.py](front/page1.py)，操作：
1. 下拉选择数据集（2019/2021/20192021）
2. 选择一个个体 ID
3. 观察原始轨迹的 ArcLayer 弧线图 + Heatmap 热力图

目的：展示数据的真实分布，为后续对比建立 baseline。

**Step 2 → 轨迹生成工具（Page 5）— 核心演示**

打开 [front/page5.py](front/page5.py)，操作：
1. 选择数据集（如 2019）
2. 选择模式：**混合动机（Hybrid）**——论文核心方法
3. 选择模型（如 gemini-2.5-flash-lite）
4. 选择个体 ID
5. **开启 Critic** 开关（展示 Reflexion 校正能力）
6. 点击"开始生成"
7. 观察进度条：动机挖掘 → 动机融合 → 轨迹生成 → 硬性校验 → 软性校验

重点展示：
- 生成完毕后，对比**真实轨迹 vs 生成轨迹**的地图
- **雷达图**展示活动类型分布对比
- 分时段（06-11、11-16、16-24）的轨迹分布对比

**Step 3 → 生成轨迹展示（Page 2）**

打开 [front/page2.py](front/page2.py)，对比不同模型/模式的结果：
- 选择 "normal" 数据集
- 对比 `gemini_PMC`、`gp3.5turbo_PMC` 等不同模型的生成效果
- 观察分时段雷达图，展示生成轨迹在时间维度上的准确性

**Step 4 → 个体追踪（Page 3）**

打开 [front/page3.py](front/page3.py)，展示：
1. 选中某个个体的 Persona 识别结果（[front/page3.py 第 26-27 行](front/page3.py#L26-L27)）
2. 展开"活动模式识别"查看 LLM 识别的个性化特征
3. 展开"活动轨迹生成"查看生成过程中的 LLM 对话历史
4. 展开"LLM 轨迹评分"查看打分结果

**Step 5 → 评估指标回顾**

切换到 [evaluate.py](evaluate.py)，展示命令行的评估输出：
- 运行 `python evaluate.py --dataset 2019 --mode 3`
- 展示 SD / SI / DARD / STVD 四个指标的数值
- 展示 KDE 分布图和分箱柱状图
- 若启用 `--llmjudge` 或 `--preference`，展示 LLM 评分和偏好率结果

---

### 四、总结

#### 代码健壮性
1. **多层校验机制**：硬约束（地点合法性、时间格式）+ 软约束（语义 Critic）+ Reflexion 自我修正
2. **完善的错误处理**：JSON 解析失败回退、LLM 调用异常捕获、最大重试次数兜底
3. **数据一致性保障**：评估模块中 `assert` 确保生成数据与真实数据长度一致

#### 工程工作量
- **核心引擎**：7 个 Python 模块（agent、trajectory_generate、persona_identify、motivation_fusion、retrieval_helper、process_tools 等）
- **可视化平台**：6 个 Streamlit 页面 + pydeck 地图可视化 + 雷达图 + 热力图
- **评估系统**：4 个定量指标 + LLM 定性评估
- **Prompt 模板**：11 个精心设计的 prompt 模板（critic、init、eval、motivation infer 等）

#### 对论文实验的支撑
- 四个 JSD 指标提供**定量评估**，覆盖空间、时间、活动、位置四个维度
- LLM as Judge 提供**定性评估**，反映轨迹的逻辑合理性
- 偏好率评估验证生成轨迹的**个性化程度**
- 可视化工具辅助**实验分析**与结果展示
