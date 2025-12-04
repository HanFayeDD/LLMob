# %%
import os
import pickle
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# 引入 Gensim 用于 Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 设置随机种子以保证结果可复现
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ==========================================
# 1. 数据加载 (保持不变)
# ==========================================
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data    

relative_dir = r"data\2019"
# 检查目录是否存在
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
                data = pickle.load(f)[0] 
                raw_data[id] = data

print(f"Loaded {len(raw_data)} users.")

map_path = r"data\location_activity_map.pkl"
if os.path.exists(map_path):
    location_category_map = load_pickle(map_path)
else:
    location_category_map = {} 

# %%
# ==========================================
# 2. 数据预处理：构建 TaggedDocument
# ==========================================

def parse_daily_trajectory_list(day_record, mapping_dict):
    """
    解析单日的轨迹，返回 Token 列表 (List of strings)。
    Doc2Vec 需要 list 格式，而不是 join 后的字符串。
    """
    daily_categories = []
    pattern = re.compile(r"([a-zA-Z ]+)#\d+")
    
    matches = pattern.findall(day_record)
    for loc_name in matches:
        loc_name = loc_name.strip()
        category = mapping_dict.get(loc_name, "Unknown")
        # 将空格替换为下划线，确保 "Fast Food" 被视为一个词
        category = category.replace(" ", "_") 
        daily_categories.append(category)
    
    return daily_categories

# --- 核心修改：构建 TaggedDocument 列表 ---
# Doc2Vec 的输入必须是 TaggedDocument 对象
tagged_documents = []
doc_mapping_info = [] # 用于后续找回每一条数据属于哪个用户

print("正在预处理数据并构建 TaggedDocuments...")

for pid, trajectories in raw_data.items():
    for day_idx, day_record in enumerate(trajectories):
        tokens = parse_daily_trajectory_list(day_record, location_category_map)
        
        if tokens: # 过滤空轨迹
            # Tag 必须是唯一的，我们使用 "UserID_DayIndex"
            tag_id = f"{pid}_{day_idx}"
            
            # 创建 TaggedDocument
            td = TaggedDocument(words=tokens, tags=[tag_id])
            tagged_documents.append(td)
            
            # 记录元数据以便后续聚合
            doc_mapping_info.append({
                'pid': pid,
                'day_idx': day_idx,
                'tag_id': tag_id
            })

print(f"--- 数据处理完成 ---")
print(f"总样本数 (TaggedDocuments): {len(tagged_documents)}")
if len(tagged_documents) > 0:
    print(f"示例样本: {tagged_documents[0]}")

# %%
# ==========================================
# 3. 向量化：训练 Doc2Vec 模型
# ==========================================

print("\n开始训练 Doc2Vec 模型...")

# 参数解释：
# vector_size: 向量维度。对于简单的轨迹数据，32-64 通常足够。
# window: 上下文窗口大小。设为 5 表示模型会查看当前地点前后的 5 个地点。
# min_count: 忽略出现次数少于此值的词。设为 1 保证所有地点都被考虑。
# workers: 训练使用的 CPU 核心数。
# epochs: 迭代次数。数据量较小时建议调大。
model = Doc2Vec(
    vector_size=64, 
    window=5, 
    min_count=1, 
    workers=4, 
    epochs=50,
    seed=SEED
)

# 1. 构建词汇表
model.build_vocab(tagged_documents)

# 2. 训练模型
model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)

print("Doc2Vec 模型训练完成。")

# 3. 提取向量
# 我们需要提取出每一天对应的向量，形成矩阵 X
X_daily = []
daily_owners = [] # 保持与 X_daily 对应的用户ID顺序

for info in doc_mapping_info:
    tag = info['tag_id']
    # model.dv (Document Vectors) 存储了训练好的文档向量
    vector = model.dv[tag]
    X_daily.append(vector)
    daily_owners.append(info['pid'])

X_daily = np.array(X_daily)
print(f"特征矩阵形状: {X_daily.shape}")

# %%
# ==========================================
# 4. 聚类：对“天”进行聚类 (K-Means)
# ==========================================

k_range = range(3, 10) 
silhouette_scores = []
models = {}

print(f"\n{'K值':<5} | {'Inertia':<15} | {'轮廓系数':<20}")
print("-" * 45)

# 采样计算轮廓系数（提升速度）
sample_size = min(2000, X_daily.shape[0])
sample_indices = np.random.choice(X_daily.shape[0], sample_size, replace=False)
X_sample = X_daily[sample_indices]

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    kmeans.fit(X_daily)
    
    labels_sample = kmeans.predict(X_sample)
    if len(set(labels_sample)) > 1:
        score = silhouette_score(X_sample, labels_sample)
    else:
        score = -1
        
    silhouette_scores.append(score)
    models[k] = kmeans
    print(f"{k:<5} | {kmeans.inertia_:<15.4f} | {score:<20.4f}")

best_k_idx = np.argmax(silhouette_scores)
best_k = k_range[best_k_idx]
print(f"\n>>> 最佳 K 值: {best_k}")

final_model = models[best_k]
daily_labels = final_model.labels_

# %%
# ==========================================
# 5. 归类：多数投票 (Majority Voting)
# ==========================================
# 逻辑与之前相同：统计用户每一天属于哪个 Cluster，占比最大的即为用户类别

user_cluster_counts = defaultdict(Counter)

for pid, label in zip(daily_owners, daily_labels):
    user_cluster_counts[pid][label] += 1

user_final_labels = {}
user_details = {} 

for pid, counts in user_cluster_counts.items():
    most_common_cluster, count = counts.most_common(1)[0]
    user_final_labels[pid] = most_common_cluster
    
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
# 6. 可视化准备：计算用户的“平均向量”并 PCA
# ==========================================

# 1. 映射用户到 X_daily 的索引
user_indices_map = defaultdict(list)
for idx, pid in enumerate(daily_owners):
    user_indices_map[pid].append(idx)

unique_users = list(user_final_labels.keys())
user_vectors = []
sorted_user_ids = []
sorted_labels = []
sorted_details = []

# X_daily 已经是 numpy array (dense)，不需要像 TF-IDF 那样处理稀疏矩阵
for pid in unique_users:
    indices = user_indices_map[pid]
    user_days_matrix = X_daily[indices]
    
    # 计算该用户所有天向量的均值 (Centroid)
    avg_vec = user_days_matrix.mean(axis=0)
    
    user_vectors.append(avg_vec)
    sorted_user_ids.append(pid)
    sorted_labels.append(user_final_labels[pid])
    sorted_details.append(user_details[pid])

user_vectors = np.array(user_vectors)

# 2. PCA 降维
pca = PCA(n_components=2)
coords = pca.fit_transform(user_vectors)

# 3. 生成 JSON
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

# 4. 写入 HTML
try:
    template_name = 'cluster_template.html'
    if os.path.exists(template_name):
        with open(template_name, 'r', encoding='utf-8') as f:
            s = f.read()
        
        s = s.replace("<marker></marker>", str(json_output))
        
        output_filename = 'cluster_output_doc2vec.html'
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(s)
        print(f"成功生成可视化文件: {output_filename}")
    else:
        print(f"未找到模板文件 {template_name}，仅生成了 JSON 数据。")
except Exception as e:
    print(f"生成 HTML 时出错: {e}")