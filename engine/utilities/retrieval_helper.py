import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import os
sns.set_theme(style="darkgrid", font="Times New Roman")
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def calculate_relative_features(date1, date2):
    delta = date1 - date2
    return [
        delta.days/365.,
        int(delta.days % 7 == 0),
        date1.month == date2.month,
    ]


def encode_dates(date1, date2):
    relative_features = calculate_relative_features(date1, date2)

    return np.array(relative_features)

def input_from_date(query_date, node_date):
    '''
    返回 3 维特征：
    delta.days/365.：归一化的天数差（反映时间长短影响）。
    int(delta.days % 7 == 0)：是否为整周间隔（与星期相关的周期性）。
    date1.month == date2.month：是否同月（反映季节/月份相关行为）。
    '''
    
    f = encode_dates(query_date, node_date)
    return f


# Define a custom dataset class
class ContrastiveDataset(Dataset):
    def __init__(self, data, eval=None, num_pairs=10, class_id_map=None):
        self.data = data
        self.eval = eval
        self.num_pairs = num_pairs # 传入了3
        self.eval_pairs = [] # 每一个元素是对应一个轨迹的一个最相似轨迹和num_pairs-1个不相似轨迹的日期特征
        self.class_id_map = class_id_map # 传入地点>活动类型映射

        self._create_pairs()


    def _create_pairs(self):
        scores = np.zeros((len(self.data), len(self.data))) - 100.
        # 日期列表
        node_dates = [datetime.strptime(s.split(": ")[0].split(" ")[-1], "%Y-%m-%d") for s in self.data]
        # 轨迹列表，一个元素是一天的轨迹
        activities = [s.split(": ")[1] for s in self.data]
        l = len(self.data)
        # l = self.num_pairs
        for i in range(l):
            sample = []
            for j in range(l):
                if i != j and scores[i][j] < 0.:
                    score = self.eval(activities[i], activities[j], self.class_id_map)
                    scores[i][j] = score
                    scores[j][i] = score

            sorted_indices = sorted(range(len(self.data)), key=lambda k: scores[k].reshape(-1).tolist(), reverse=True)
            query_date = node_dates[i] # 固定第i天
            node_date = node_dates[sorted_indices[0]] # most similar / positive sample 选出最相似的一天
            input_ = input_from_date(query_date, node_date) # 得到1*3相似度矩阵
            sample.append(input_)
            
            # num_pairs 表示每个锚点（anchor）样本包含的样本数（包括 1 个正样本 + num_pairs - 1 个负样本
            for k in range(self.num_pairs-1):
                node_date = node_dates[sorted_indices[-k+1]] # least similar / negative sample
                input_ = input_from_date(query_date, node_date) 
                sample.append(input_)
            # 第 0 行是与 i 最相似的“正样本”对应的日期特征，其余行为按相似度最不相似选出的负样本的日期特征，最终存入 self.eval_pairs
            sample = np.array(sample).reshape(self.num_pairs, -1) # 转为2维3*3矩阵，每一行是一个日期相似衡量1*3矩阵
            self.eval_pairs.append(sample)

    def __len__(self):
        return len(self.eval_pairs)

    def __getitem__(self, idx):
        return self.eval_pairs[idx]

def map_traj2mat(traj, class_loc_map, act_map, interval=60):
    '''
    入参：
        一条轨迹、地点>地点类型、地点类型>数字映射    
    出参：
        矩阵形状（24，len(act_map)）计数矩阵
    '''
    # 矩阵形状（24，len(act_map)）
    mat = np.zeros((int(1440 / interval), len(act_map)))
    traj = traj.replace(":00, ", " at ")
    traj_ = traj.split(" at ")
    i = 0
    while i < len(traj_) - 1:
        loc = traj_[i].split("#")[0]
        time_str = traj_[i + 1]
        act_class = class_loc_map[loc]
        time_obj = None
        if '.' not in time_str:
            time_obj = datetime.strptime(time_str, '%H:%M')
        else:
            time_obj = datetime.strptime(time_str, '%H:%M:%S.')
        if time_obj is None:
            continue
        minutes = time_obj.hour * 60 + time_obj.minute
        mat[int(minutes / interval), act_map[act_class]] += 1
        i += 2
    return mat


def act_mat_compute(traj1, traj2, class_loc_map):
    '''
        计算两条轨迹在时间-地点类型上的相似度
        先转为二维矩阵，在进行相似度计算
    '''
    # 地点类型去重后作序列化映射
    act_map = {v: id_ for id_, v in enumerate(list(set(class_loc_map.values())))}
    # 轨迹、地点>地点类型、地点类型>数字映射
    mat1 = map_traj2mat(traj1, class_loc_map, act_map)
    mat2 = map_traj2mat(traj2, class_loc_map, act_map)
    # 计算“重合槽位数”（numerator）：满足同时两矩阵在某时间槽和类型列上“相等且至少为 1”的格子数量
    # 计算分母（denominator）：两矩阵各自非零槽位数量的较大值
    comp = np.where((mat1 == mat2) & (mat1 >= 1))[0].shape[0] / max(np.where((mat1 >= 1))[0].shape[0],
                                                                    np.where((mat2 >= 1))[0].shape[0])
    return comp
    
class NodeWithScore:
    def __init__(self, node, score, input_):
        self.node = node
        self.score = score
        self.input_ = input_


class DeepModel(nn.Module):
    """原始简单两层 MLP，保留以供对比。
    结构: Linear(3->64) -> ReLU -> Linear(64->1)
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class EnhancedDeepModel(nn.Module):
    """增强版时间相似度评分网络。

    改进点（相比 DeepModel）：
    1. 更深的网络：4 层全连接（3→64→128→64→1），提升非线性拟合能力。
    2. BatchNorm：每层后接 BatchNorm，稳定训练、加速收敛、**显著抑制 loss 震荡**。
    3. Dropout（p=0.1）：轻量正则化，防止在小数据集上过拟合。
    4. 残差连接：中间两层（64→128→64）构成残差块，缓解梯度消失，
       让浅层特征可以直接传递到深层，提升训练稳定性。
    5. GELU 激活函数：比 ReLU 更平滑，梯度过渡更自然，有助于减少训练抖动。

    参数量估算：
      - fc1: 3×64 + 64 = 256
      - fc2: 64×128 + 128 = 8,320
      - fc3: 128×64 + 64 = 8,256
      - fc_out: 64×1 + 1 = 65
      - BN 参数: ~512
      总计 ≈ 17,409 参数（原模型 ≈ 321 参数）
      依然极其轻量，1500 epoch 训练时间预计 1-3 分钟。
    """
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(EnhancedDeepModel, self).__init__()

        # 输入映射层: input_size -> hidden_size (3 -> 64)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        # 残差块: hidden_size -> hidden_size*2 -> hidden_size (64 -> 128 -> 64)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

        # 输出层: hidden_size -> output_size (64 -> 1)
        self.fc_out = nn.Linear(hidden_size, output_size)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # 输入映射
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # 残差块
        residual = x  # 保存残差 (64 维)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = x + residual  # 残差连接
        x = self.activation(x)

        # 输出
        x = self.fc_out(x)
        return x


class TemporalRetriever:
    def __init__(self, nodes, similarity_top_k, is_train=None, class_id_map=None,
                 model_type="DeepModel"):
        """
        Args:
            nodes: 轨迹节点列表。
            similarity_top_k: 检索时返回的 top-k 数量。
            is_train: 非 None 时触发训练流程。
            class_id_map: 地点→活动类型映射。
            model_type: 使用的模型类型，可选 "DeepModel" 或 "EnhancedDeepModel"，
                        默认为 "DeepModel"。
        """
        self.nodes = nodes
        self.similarity_top_k = similarity_top_k
        self.feature_size = 3
        self.model_type = model_type
        if is_train is not None:
            ## 组织正负样本训练数据集
            self.calibrate_dataset = ContrastiveDataset(nodes,
                                                        eval=act_mat_compute,
                                                        num_pairs=3,
                                                        class_id_map=class_id_map)
            # 根据 model_type 选择模型
            if self.model_type == "EnhancedDeepModel":
                self._model = EnhancedDeepModel(self.feature_size, 64, 1)
            else:
                self._model = DeepModel(self.feature_size, 64, 1)
            self.optimizer = optim.Adam(self._model.parameters(), lr=0.002)
            self.criterion = torch.nn.MSELoss()
            self.calibrate_score_function(num_epochs=1500)

    def retrieve(self, query_str):
        scored_nodes = self.get_scored_nodes(query_str)
        nodes = sorted(scored_nodes, key=lambda x: x.score, reverse=True)
        retrieved_nodes = [node.node for node in nodes[: self.similarity_top_k]]
        return retrieved_nodes

    def get_scored_nodes(self, query_date):
        _nodes = []
        query_date = datetime.strptime(query_date, "%Y-%m-%d")
        for node in self.nodes:
            node_date = datetime.strptime(node.split(": ")[0].split(" ")[-1], "%Y-%m-%d")
            score = self.get_similarity_score(query_date, node_date)
            _nodes.append(NodeWithScore(node=node, score=score, input_=input_from_date(query_date, node_date)))
        return _nodes

    def get_similarity_score(self, query_date, node_date):
        input_ = input_from_date(query_date, node_date)
        self._model.eval()
        with torch.no_grad():
            # unsqueeze 保证 batch 维度存在，兼容 BatchNorm（EnhancedDeepModel）
            score = self._model(torch.tensor(input_).float().unsqueeze(0)).squeeze(0)
        return score

    def calibrate_score_function(self, batch_size=64, num_epochs=10):
        dataloader = DataLoader(self.calibrate_dataset, batch_size=batch_size, shuffle=True)
        print(f"训练样本天数: {len(self.calibrate_dataset)}, 模型: {self.model_type}")
        loss_list = []
        for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
            self._model.train()  # 确保 BatchNorm / Dropout 处于训练模式
            epoch_loss_sum = 0.0  # 初始化当前 epoch 的总 loss
            batch_count = 0       # 记录 batch 数量
            for batch in dataloader:
                # 每个batch的形状： (B, num_pairs, feature_size) = (B, 3, 3)
                positive_pairs = batch[:, 0, :].reshape(-1, self.feature_size) ## B，3
                positive_scores = self._model(torch.tensor(positive_pairs).float()) ## B，1
                negative_pairs = batch[:, 1:, :].reshape(-1, self.feature_size) ## B*(num_pairs-1), 3  展平为2D以兼容BatchNorm
                negative_scores = self._model(torch.tensor(negative_pairs).float()).reshape(batch.shape[0], -1) ## B, 2
                logits = torch.cat([positive_scores, negative_scores], dim=1) ## B，3 正+负分数拼接
                loss = -torch.log_softmax(logits, dim=1)[:, 0]
                loss = loss.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 累加 loss
                epoch_loss_sum += loss.item()
                batch_count += 1
            # 计算并记录平均 loss
            avg_epoch_loss = epoch_loss_sum / batch_count
            loss_list.append(avg_epoch_loss)
        draw_loss_curve(loss_list)
        print("Calibration finished!")
        


def retrieve_loc(person, route):
    """_summary_
        基于类别的地点联想推荐
        为了防止 Prompt 过长或推荐过多重复信息，代码限制了每个类别最多推荐 7 个地点。
    Args:
        person (_type_): _description_
        route (_type_): _description_
                
    Returns:
        _type_: _description_
        返回一个字符串列表 area，包含了与参考路径中地点类型相似的一组候选地点。这些地点随后会被放入 Prompt 中（在 mob_gen 函数里），提示 LLM：“你可以去这些地方”。
    """
    area = []
    ## 提取地点名称
    loc_in_retrieve = route.split(": ")[1].replace(",", " at ").split(" at ")[::2]
    selected_loc_cat = {}
    for loc in loc_in_retrieve:
        loc = loc.lstrip().rstrip()
        ## 地点名称对应的类别
        c = person.loc_cat[loc.split("#")[0]]
        for k, v in person.area_freq.items():
            k = k.replace(".", "")
            if person.loc_cat[k.split("#")[0]] == c:
                if c not in selected_loc_cat:
                    selected_loc_cat[c] = 1
                else:
                    selected_loc_cat[c] += 1
                if selected_loc_cat[c] <= 7:
                    area.append(k)
                else:
                    break
    return area


def draw_loss_curve(losses: list, save_name: str = "loss_curve.png"):
    """绘制损失曲线并保存到工作根目录。

    Args:
        losses: 每个 epoch 对应的 loss 值列表。
        save_name: 保存的文件名，默认为 loss_curve.png。
    """
    epochs = list(range(1, len(losses) + 1))

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=epochs, y=losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.tight_layout()

    save_path = os.path.join(os.getcwd(), save_name)
    plt.savefig(save_path, dpi=500)
    plt.close()
    print(losses[-10:])
    print(f"Loss curve saved to {save_path}")

def draw_mat_from_traj(traj, class_loc_map, save_name="traj_heatmap.png", title=None):
    act_map = {v: id_ for id_, v in enumerate(list(set(class_loc_map.values())))}
    mat1 = map_traj2mat(traj, class_loc_map, act_map)

    # 将 act_map 按 id 顺序还原为类别列表（保证 y 轴顺序一致）
    act_list = [None] * len(act_map)
    for act, idx in act_map.items():
        act_list[idx] = act

    # 绘制热力图
    plt.figure(figsize=(12, max(4, len(act_list) * 0.4)))
    sns.heatmap(
        mat1.T,
        cmap="YlGnBu",
        cbar_kws={"label": "Count"},
        xticklabels=4,  # 只显示部分 x 轴刻度
        yticklabels=act_list,
        annot=False,
        fmt="d"
    )
    plt.xlabel("Time Slot (1h an interval)")
    plt.ylabel("Activity Type")
    plt.title(title)
    plt.tight_layout()

    save_path = os.path.join(os.getcwd(), save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Heatmap saved to {save_path}")
# ...existing code...   
    
if __name__ == "__main__":
    import pickle
    with open(r"data\2019\13.pkl", "rb") as f:
        att = pickle.load(f)
        train_routine_list = att[0]
        loc_cat = att[11]
    idx = 8
    retriever = TemporalRetriever(train_routine_list, 6, is_train=1, class_id_map=loc_cat, model_type="DeepModel")
    # print(train_routine_list[idx])
    # t = train_routine_list[idx].split(": ")[0].split("at")[-1].strip()
    # draw_mat_from_traj(train_routine_list[idx].split(": ")[1], loc_cat,
    #                    title = f"Trajectory Time-Activity Matrix Heatmap of Person 2575 on {t}")
    
