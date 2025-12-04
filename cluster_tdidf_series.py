# %%
import os
import pickle
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ==========================================
# 1. 数据加载 (保持不变)
# ==========================================
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data    

relative_dir = r"data\2019"
# 检查目录是否存在，防止报错
if not os.path.exists(relative_dir):
    print(f"Warning: Directory {relative_dir} not found. Please ensure data exists.")
    raw_data = {}
else:
    ids = os.listdir(relative_dir)
    ids.sort(reverse=True)
    raw_data = {}
    for id in ids:
        file_path = os.path.join(relative_dir, id)
        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                # 假设 pickle 结构是 [list_of_days, ...]
                data = pickle.load(f)[0] 
                raw_data[id] = data

print(f"Loaded {len(raw_data)} users.")

# 加载映射表
map_path = r"data\location_activity_map.pkl"
if os.path.exists(map_path):
    location_category_map = load_pickle(map_path)
else:
    location_category_map = {} # 防止报错，实际运行需要文件

# %%
# ==========================================
# 2. 数据预处理：按“天”拆分样本
# ==========================================

def parse_daily_trajectory(day_record, mapping_dict):
    """
    解析单日的轨迹字符串。
    """
    daily_categories = []
    # 正则匹配 LocationName
    pattern = re.compile(r"([a-zA-Z ]+)#\d+")
    
    matches = pattern.findall(day_record)
    for loc_name in matches:
        loc_name = loc_name.strip()
        category = mapping_dict.get(loc_name, "Unknown")
        category = category.replace(" ", "_") 
        daily_categories.append(category)
    
    # 返回空格分隔的字符串，代表这一天的序列
    return " ".join(daily_categories)

# --- 核心修改：构建“每日”语料库 ---
daily_corpus = []       # 存放每一天的轨迹字符串
daily_owners = []       # 存放这一天属于哪个 User ID
daily_indices = []      # 记录这是该用户的第几天 (可选，用于调试)

for pid, trajectories in raw_data.items():
    # trajectories 是一个列表，每个元素代表一天
    for day_idx, day_record in enumerate(trajectories):
        seq = parse_daily_trajectory(day_record, location_category_map)
        
        # 过滤掉空轨迹（如果某天没有任何记录）
        if seq.strip():
            daily_corpus.append(seq)
            daily_owners.append(pid)
            daily_indices.append(day_idx)

print(f"--- 数据处理完成 ---")
print(f"总用户数: {len(raw_data)}")
print(f"总样本数 (人*天): {len(daily_corpus)}")
if len(daily_corpus) > 0:
    print(f"示例样本 ({daily_owners[0]} Day {daily_indices[0]}): {daily_corpus[0]}")

# %%
# ==========================================
# 3. 向量化：引入 N-gram 捕捉时序
# ==========================================

# ngram_range=(1, 2): 
# 不仅统计 "Home", "Work" (1-gram)
# 还统计 "Home Work", "Work Restaurant" (2-gram) -> 这捕捉了先后顺序
# use_idf=True: 降低常见地点（如大家都在睡觉的地方）的权重，突出独特行为
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=1000) 

X_daily = vectorizer.fit_transform(daily_corpus)
print(f"特征矩阵形状: {X_daily.shape}")

# %%
# ==========================================
# 4. 聚类：对“天”进行聚类
# ==========================================

# 同样使用轮廓系数寻找最佳 K
k_range = range(3, 10) 
silhouette_scores = []
models = {}

print(f"{'K值':<5} | {'Inertia':<15} | {'轮廓系数':<20}")
print("-" * 45)

# 为了速度，如果样本量极大(>10000)，可以考虑只用部分样本计算轮廓系数
sample_indices = np.random.choice(X_daily.shape[0], min(2000, X_daily.shape[0]), replace=False)
X_sample = X_daily[sample_indices]

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_daily) # 在全量数据上训练
    
    # 在采样数据上评估（计算全量轮廓系数太慢）
    labels_sample = kmeans.predict(X_sample)
    if len(set(labels_sample)) > 1: # 防止聚成一类报错
        score = silhouette_score(X_sample, labels_sample)
    else:
        score = -1
        
    silhouette_scores.append(score)
    models[k] = kmeans
    print(f"{k:<5} | {kmeans.inertia_:<15.4f} | {score:<20.4f}")

# 选择最佳 K
best_k_idx = np.argmax(silhouette_scores)
best_k = k_range[best_k_idx]
print(f"\n>>> 最佳 K 值 (基于每日行为模式): {best_k}")

# 获取最佳模型的标签（针对每一天）
final_model = models[best_k]
daily_labels = final_model.labels_

# %%
# ==========================================
# 5. 归类：多数投票 (Majority Voting)
# ==========================================
# 逻辑：如果 User A 有 20 天属于 Cluster 1，5 天属于 Cluster 2 -> User A = Cluster 1

user_cluster_counts = defaultdict(Counter)

# 统计每个用户每天的类别
for pid, label in zip(daily_owners, daily_labels):
    user_cluster_counts[pid][label] += 1

# 决定用户的最终类别
user_final_labels = {}
user_details = {} # 存储用于显示的详情

for pid, counts in user_cluster_counts.items():
    # 找到出现次数最多的 Cluster
    most_common_cluster, count = counts.most_common(1)[0]
    user_final_labels[pid] = most_common_cluster
    
    # 构造详情字符串：显示该用户的模式分布
    # e.g., "Cluster 1 (80%), Cluster 2 (20%)"
    total_days = sum(counts.values())
    detail_str = f"Total Days: {total_days}. Patterns: "
    patterns = []
    for cls, cnt in counts.most_common():
        patterns.append(f"Type {cls}: {cnt/total_days:.0%}")
    user_details[pid] = detail_str + ", ".join(patterns)

print("\n--- 用户最终归类示例 ---")
count = 0
for pid, label in user_final_labels.items():
    print(f"User {pid}: Final Cluster {label} | {user_details[pid]}")
    count += 1
    if count >= 5: break

# %%
# ==========================================
# 6. 可视化准备：计算用户的“平均向量”
# ==========================================
# 因为我们在“天”的粒度上聚类，但要在“人”的粒度上画图。
# 方案：计算每个用户所有天向量的平均值 (Centroid)，然后 PCA 降维。

# 1. 构建用户到每日向量索引的映射
user_indices_map = defaultdict(list)
for idx, pid in enumerate(daily_owners):
    user_indices_map[pid].append(idx)

# 2. 计算每个用户的平均向量
unique_users = list(user_final_labels.keys())
user_vectors = []
sorted_user_ids = []
sorted_labels = []
sorted_details = []

dense_X_daily = X_daily  # 稀疏矩阵

for pid in unique_users:
    indices = user_indices_map[pid]
    # 获取该用户所有天的向量切片
    user_days_matrix = dense_X_daily[indices]
    # 计算均值 (axis=0 沿列压缩)
    # 注意：sparse matrix mean 返回的是 matrix 对象，需要转 array
    avg_vec = user_days_matrix.mean(axis=0)
    # 转换格式
    user_vectors.append(np.asarray(avg_vec).flatten())
    
    sorted_user_ids.append(pid)
    sorted_labels.append(user_final_labels[pid])
    sorted_details.append(user_details[pid])

user_vectors = np.array(user_vectors)

# 3. PCA 降维
pca = PCA(n_components=2)
coords = pca.fit_transform(user_vectors)

# 4. 生成 JSON
visualization_data = []
for i, pid in enumerate(sorted_user_ids):
    visualization_data.append({
        "id": pid,
        "x": float(coords[i, 0]),
        "y": float(coords[i, 1]),
        "cluster": int(sorted_labels[i]),
        "details": sorted_details[i]
    })

json_output = json.dumps(visualization_data)
print(f"\n--- 生成了 {len(visualization_data)} 个用户的可视化数据 ---")

# 写入 HTML (复用原有逻辑)
try:
    template_name = 'cluster_template.html'
    if os.path.exists(template_name):
        with open(template_name, 'r', encoding='utf-8') as f:
            s = f.read()
        
        s = s.replace("<marker></marker>", str(json_output))
        
        output_filename = 'cluster_output_tdidf_series.html'
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(s)
        print(f"成功生成可视化文件: {output_filename}")
    else:
        print(f"未找到模板文件 {template_name}，仅生成了 JSON 数据。")
except Exception as e:
    print(f"生成 HTML 时出错: {e}")