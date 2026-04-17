"""
混合动机推演模式（Hybrid Motivation Inference）

本模块实现了宏观周期先验（M_r）与短期行为惯性（M_e）的上下文自适应融合。
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging

# ==============================================================================
# [新增] 混合动机融合模块
# ==============================================================================

def build_context_gating_vector(
    train_routine_list: List[str],
    query_date: str,
    k: int = 7
) -> np.ndarray:
    """
    构建上下文门控向量 c_d，用于自适应权重计算。
    
    门控向量包含三个维度：
    1. 近期周期偏移：最近k天中周末的比例
    2. 近期数据完整性：近期签到记录覆盖时间窗口的比例
    3. 短期行为波动率：近期空间步长距离的方差
    
    Args:
        train_routine_list: 训练轨迹列表，格式如 ["2019-01-01: Home at 08:00, Work at 09:00", ...]
        query_date: 查询日期（字符串格式 "YYYY-MM-DD"）
        k: 近期天数窗口，默认7天
    
    Returns:
        np.ndarray: 3维门控向量 [近期周期偏移, 近期数据完整性, 短期行为波动率]
    """
    # 解析所有日期
    dates = []
    routines = {}
    for routine in train_routine_list:
        parts = routine.split(": ")
        if len(parts) >= 2:
            date_str = parts[0].split(" ")[-1].strip()
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                dates.append(date_obj)
                routines[date_str] = routine
            except ValueError:
                continue
    
    if not dates:
        # 如果没有轨迹数据，返回中性值
        return np.array([0.5, 0.5, 0.5])
    
    # 获取查询日期
    try:
        query_date_obj = datetime.strptime(query_date, "%Y-%m-%d")
    except ValueError:
        return np.array([0.5, 0.5, 0.5])
    
    # 1. 近期周期偏移：最近k天中周末的比例
    recent_weekend_count = 0
    recent_days = 0
    for i in range(1, k + 1):
        check_date = query_date_obj - timedelta(days=i)
        check_date_str = check_date.strftime("%Y-%m-%d")
        if check_date_str in routines:
            recent_days += 1
            if check_date.weekday() >= 5:  # 周末
                recent_weekend_count += 1
    
    # 如果没有近期数据，使用中性值0.5
    recent_periodic_offset = recent_weekend_count / recent_days if recent_days > 0 else 0.5
    
    # 2. 近期数据完整性：近期签到记录覆盖时间窗口的比例
    # 计算从 query_date - k 天到 query_date - 1 天的时间窗口
    window_start = query_date_obj - timedelta(days=k)
    window_end = query_date_obj - timedelta(days=1)
    window_size = (window_end - window_start).days + 1  # 完整窗口天数
    
    # 统计窗口内的实际签到天数
    actual_days = 0
    for check_date in [query_date_obj - timedelta(days=i) for i in range(1, k + 1)]:
        check_date_str = check_date.strftime("%Y-%m-%d")
        if check_date_str in routines:
            actual_days += 1
    
    # 数据完整性比例
    data_completeness = actual_days / window_size if window_size > 0 else 0.5
    
    # 3. 短期行为波动率：近期空间步长距离的方差
    # 计算相邻轨迹之间的空间距离变化
    distances = []
    sorted_dates = sorted([d for d in dates if d < query_date_obj], reverse=True)[:k]
    
    for i in range(len(sorted_dates) - 1):
        date1 = sorted_dates[i]
        date2 = sorted_dates[i + 1]
        routine1 = routines.get(date1.strftime("%Y-%m-%d"), "")
        routine2 = routines.get(date2.strftime("%Y-%m-%d"), "")
        
        # 提取两个轨迹的空间中心点
        center1 = extract_trajectory_center(routine1)
        center2 = extract_trajectory_center(routine2)
        
        if center1 and center2:
            # 计算欧氏距离
            dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            distances.append(dist)
    
    # 计算方差（如果距离数据不足，使用默认值）
    if len(distances) >= 2:
        spatial_variance = float(np.var(distances))
        # 归一化到 [0, 1] 范围（假设最大合理距离为100公里）
        spatial_variance = min(spatial_variance / 100.0, 1.0)
    else:
        spatial_variance = 0.5
    
    # 构建门控向量
    context_vector = np.array([
        recent_periodic_offset,
        data_completeness,
        spatial_variance
    ])
    
    return context_vector


def extract_trajectory_center(routine: str) -> Optional[Tuple[float, float]]:
    """
    从轨迹字符串中提取空间中心点（所有地点坐标的平均值）。
    
    Args:
        routine: 轨迹字符串，格式如 "Home at 08:00, Work at 09:00"
    
    Returns:
        tuple: (纬度, 经度) 或 None（如果无法提取）
    """
    import re
    
    # 匹配地点坐标模式：地点名 (纬度, 经度)
    pattern = r'\(([\d.]+),\s*([\d.]+)\)'
    matches = re.findall(pattern, routine)
    
    if not matches:
        return None
    
    lats = [float(m[0]) for m in matches]
    lngs = [float(m[1]) for m in matches]
    
    return (np.mean(lats), np.mean(lngs))


def compute_adaptive_weight(
    context_vector: np.ndarray,
    weight_params: Optional[Dict] = None,
    use_mlp: bool = False,
    mlp_weights: Optional[np.ndarray] = None,
    mlp_biases: Optional[np.ndarray] = None
) -> float:
    """
    计算自适应权重 α_d ∈ [0, 1]。
    
    α_d → 1 表示高度依赖检索先验（如：近期无数据/波动大/遇节假日）
    α_d → 0 表示高度依赖演化惯性（如：数据密集/行为稳定/常规工作日）
    
    Args:
        context_vector: 上下文门控向量 c_d (3维)
        weight_params: 权重参数字典 {"w": [w1, w2, w3], "b": b}
        use_mlp: 是否使用MLP计算权重
        mlp_weights: MLP权重列表（每层一个数组）
        mlp_biases: MLP偏置列表（每层一个数组）
    
    Returns:
        float: 自适应权重 α_d
    """
    if use_mlp and mlp_weights is not None and mlp_biases is not None:
        # 使用MLP计算权重
        x = context_vector.astype(np.float32)
        for i, (w, b) in enumerate(zip(mlp_weights, mlp_biases)):
            x = np.dot(x, w) + b
            if i < len(mlp_weights) - 1:
                x = np.tanh(x)  # 隐藏层使用tanh
        # 输出层使用sigmoid
        alpha = 1.0 / (1.0 + np.exp(-x))
        return float(alpha)
    else:
        # 使用线性加权 + sigmoid
        if weight_params is None:
            # 默认参数：三个维度权重相等
            w = np.array([1, -1.5, 1])
            b = -0
        else:
            w = np.array(weight_params.get("w", [1, -1.5, 1]))
            b = weight_params.get("b", -0)
        
        z = np.dot(w, context_vector) + b
        alpha = 1.0 / (1.0 + np.exp(-z))
        return float(alpha)


def hybrid_motivation_fusion(
    M_r: str,
    M_e: str,
    alpha: float,
    query_date: str,
    date_weekday: str,
    use_heuristic: bool = False,
    heuristic_config: Optional[Dict] = None
) -> str:
    """
    LLM驱动的动机融合函数，负责权重引导、逻辑对齐与冲突裁决。
    
    Args:
        M_r: 检索动机先验（宏观周期）
        M_e: 演化动机推断（短期惯性）
        alpha: 自适应权重
        query_date: 查询日期
        date_weekday: 星期几
        use_heuristic: 是否使用启发式融合（不调用LLM）
        heuristic_config: 启发式融合配置
    
    Returns:
        str: 融合后的动机描述
    """    
    # 构建LLM融合Prompt
    fusion_prompt = f"""You are a motivation fusion expert. Your task is to combine two sources of motivation:

1. **Retrieval-based Motivation (M_r)**: Reflects long-term periodic patterns (weekends/holidays/seasons).
   "{M_r}"

2. **Evolving-based Motivation (M_e)**: Reflects recent behavior trends (last k days).
   "{M_e}"

**Context Information:**
- Query Date: {query_date}
- Day of Week: {date_weekday}
- Fusion Weight α (closer to 1 → prefer M_r; closer to 0 → prefer M_e): {alpha:.3f}

**Instructions:**
1. Analyze both motivations and identify potential conflicts or alignments.
2. Use the weight α to guide your fusion strategy:
   - If α > 0.7: Prioritize M_r, use M_e for fine-tuning
   - If 0.3 ≤ α ≤ 0.7: Balance both motivations
   - If α < 0.3: Prioritize M_e, use M_r for context awareness
3. Output a unified, coherent motivation sentence that integrates both sources.

**Output Format (JSON):**
{{
    "analysis": "Brief analysis of both motivations and their relationship",
    "fused_motivation": "Unified motivation sentence"
}}
"""
    return fusion_prompt


def heuristic_fusion(
    M_r: str,
    M_e: str,
    alpha: float,
    config: Optional[Dict] = None
) -> str:
    """
    启发式动机融合（不调用LLM，用于快速推理或LLM不可用时）。
    
    Args:
        M_r: 检索动机先验
        M_e: 演化动机推断
        alpha: 自适应权重
        config: 配置参数
    
    Returns:
        str: 融合后的动机描述
    """
    if config is None:
        config = {}
    
    # 阈值配置
    high_alpha_threshold = config.get("high_alpha_threshold", 0.7)
    low_alpha_threshold = config.get("low_alpha_threshold", 0.4)
    
    if alpha > high_alpha_threshold:
        # 高度依赖检索先验
        return f"{M_r} Given the current context, this long-term pattern is particularly relevant."
    elif alpha < low_alpha_threshold:
        # 高度依赖演化惯性
        return f"{M_e} Based on recent behavior trends, this is the dominant motivation."
    else:
        # 平衡融合
        return f"{M_r} Meanwhile, {M_e} This reflects both long-term patterns and recent trends."


def get_fusion_context(
    person,
    query_date: str,
    k: int = 7,
    weight_params: Optional[Dict] = None,
    use_mlp: bool = False,
    mlp_weights: Optional[List[np.ndarray]] = None,
    mlp_biases: Optional[List[np.ndarray]] = None,
    use_heuristic: bool = False,
    heuristic_config: Optional[Dict] = None
) -> Dict:
    """
    获取完整的融合上下文信息，包括门控向量、权重和融合Prompt。
    
    Args:
        person: Person对象，包含train_routine_list等属性
        query_date: 查询日期
        k: 近期天数窗口
        weight_params: 权重参数
        use_mlp: 是否使用MLP
        mlp_weights: MLP权重
        mlp_biases: MLP偏置
        use_heuristic: 是否使用启发式融合
        heuristic_config: 启发式配置
    
    Returns:
        dict: 包含context_vector, alpha, fusion_prompt等信息
    """
    # 构建门控向量
    context_vector = build_context_gating_vector(
        person.train_routine_list,
        query_date,
        k
    )
    
    # 计算自适应权重
    alpha = compute_adaptive_weight(
        context_vector,
        weight_params,
        use_mlp,
        mlp_weights,
        mlp_biases
    )
    
    # 获取日期信息
    try:
        date_obj = datetime.strptime(query_date, "%Y-%m-%d")
        weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][date_obj.weekday()]
    except:
        weekday = "weekday"
    
    # 构建融合Prompt
    fusion_prompt = hybrid_motivation_fusion(
        M_r="",
        M_e="",
        alpha=alpha,
        query_date=query_date,
        date_weekday=weekday,
        use_heuristic=use_heuristic,
        heuristic_config=heuristic_config
    )
    
    return {
        "context_vector": context_vector,
        "alpha": alpha,
        "weekday": weekday,
        "fusion_prompt_template": fusion_prompt
    }


# ==============================================================================
# [新增] Critic模块：用于动机融合结果检查
# ==============================================================================

def fusion_critic(llm, fusion_result: str, date_str: str) -> Optional[str]:
    """
    使用LLM检查动机融合结果的合理性。
    
    Args:
        llm: LLM实例
        fusion_result: 融合结果字符串
        date_str: 日期字符串
    
    Returns:
        None (通过) 或 错误描述字符串 (不通过)
    """
    critic_prompt = f"""Check if the following motivation fusion result is reasonable:

Date: {date_str}
Motivation: {fusion_result}

Please evaluate:
1. Is the motivation coherent and logical?
2. Does it reflect both long-term patterns and recent trends appropriately?
3. Are there any contradictions or unrealistic elements?

Output "Pass" if reasonable, or describe the issue if not.
"""
    
    try:
        feedback = llm.execute_prompt(critic_prompt, objective="Fusion_Check", temperature=0.0)
        if "Pass" in feedback:
            return None
        else:
            return feedback
    except Exception as e:
        logging.error(f"Fusion Critic Error: {e}")
        return None
