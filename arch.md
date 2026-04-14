# 架构梳理

## 入口
- `generate.py`：批量读取用户样本，组装 `Person`，补全 persona，生成 mobility trajectory。

## 主流程
1. 解析参数：`dataset / mode / seed / critic / loadpersona / identify / model`
2. 初始化日志：`utils/logger.py:init_log()`
3. 根据 `dataset` 选择用户 id 列表与场景标签
4. 逐个用户读取 `data/{dataset}/{id}.pkl`
5. 构造 `Person`（见 `engine/agent.py`）并填充属性：
   - `train_routine_list`
   - `test_routine_list`
   - `attribute`
   - `cat / domain_knowledge / neg_routines / activity_area / area_freq / loc_cat`
6. persona 处理：
   - `loadpersona=1`：从 `persona_result/results{dataset}.txt` 读取
   - 否则调用 `engine/persona_identify.py:identify()` 生成并落盘
7. 若 `identify=0`，清空 `P.attribute`
8. `mode=0` 时初始化检索器 `P.init_retriever()`
9. 调用 `engine/trajectory_generate.py:mob_gen()` 生成轨迹

## 核心模块
- `engine/agent.py`
  - 定义 `Person` 数据对象
  - 持有 LLM、轨迹数据、persona、区域统计、retriever

- `engine/persona_identify.py`
  - 基于 `domain_knowledge + activity_area + neg_routines`
  - 先生成候选 role，再为每个 role 生成 attribute
  - 用正负样本打分，选出最终 persona

- `engine/trajectory_generate.py`
  - 主函数 `mob_gen(person, mode, scenario_tag, critic_check, ...)`
  - 先推断 motivation，再生成每日 plan
  - 校验分两层：
    - 硬校验：地点/时间合法性
    - 软校验：`semantic_critic()` 用 LLM 做语义检查
  - 失败时走自修正重试；最终结果写入 `result/.../results.pkl`

- `engine/llm_configs/poe_api.py`
  - `PoeAPI`：LLM 适配层
  - 负责模型切换、chat completion、限流

- `engine/llm_configs/gpt_structure.py`
  - prompt 组装：`generate_prompt()`
  - LLM 执行：`execute_prompt()`
  - 调用日志写入：`set_current_log_file()` + `log_execution`

- `utils/logger.py`
  - 初始化全局日志输出到 `log/`

## 数据流
`pkl样本 -> Person -> persona(读取或识别) -> retriever(可选) -> motivation -> daily plan -> critic/校验 -> result`
