import os
import pickle
import re
from front.conversations import parse_conversation_log, draw_conversations
# import pydeck as pdk

##辅助函数
### 帮我写一个函数，递归找到输入目录下的非目录文件   
def find_files(path:str):
    files = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            files.extend(find_files(item_path))
        else:
            files.append(item_path)
    return files
### 
def parse_place_time(text:str):
    place, time = text.split(" at ")
    return (place.strip(), time.strip())

def load_loc_map():
    """_summary_
        load loc_map.pkl and reverse k and v
    """
    def get_lat_lon(text:str):
        numbers = re.findall(r'\d+\.\d+', text)
        # 确保提取到两个数字
        if len(numbers) == 2:
            latitude, longitude = float(numbers[0]), float(numbers[1])
            return (latitude, longitude)
        else:
            raise ValueError(f"{text} does not contain valid latitude and longitude.")
    locmap = pickle.load(open("data\loc_map.pkl", "rb"))
    d = dict()
    for k, v in locmap.items():
        try:
            d[v] = get_lat_lon(k)
        except:
            continue
    return d

def parser_actyls(actyls:list):
    res = []
    for ele in actyls:
        try:
            tmp = [None, []]
            p1, p2 = ele.split(": ")
            p1 = p1[-10:]
            tmp[0] = p1
            for pos in p2.split(","):
                tmp[1].append(parse_place_time(pos))
        except:
            continue
        res.append(tmp)
    return res 

## page1
def ls_dir_with_tag(tag:str):
    files = os.listdir(f'./data/{tag}')
    filtered_files = [f.split(".")[0] for f in files if f.endswith(".pkl")]
    return filtered_files

def load_pkl_from_data_dir(tag:str, id:str):
    person = pickle.load(open(f'./data/{tag}/{id}.pkl', 'rb'))
    actyls = person[0] + person[1]
    return parser_actyls(actyls)
    
## page2 
def load_pkl_from_selected_folder(tag:str, fold:str):
    base_path = f"./chathistory/{tag}/{fold}/traj/result/{tag}/"
    pkl_g = find_files(base_path + "generated")
    pkl_r = find_files(base_path + "ground_truth")
    
    if len(pkl_g) != len(pkl_r):
        raise ValueError("Generated and ground truth files are not equal in length.")

    res_r = []
    for ele in pkl_r:
        res_r.extend(list(pickle.load(open(ele, "rb")).values()))

    res_g = []
    for ele in pkl_g:
        res_g.extend(list(pickle.load(open(ele, "rb")).values()))
    
    return parser_actyls(res_r), parser_actyls(res_g)


## page3
def ls_id_from_history_dir(tag:str, fold:str):
    base_path = f"./chathistory/{tag}/{fold}/history/"
    files = os.listdir(base_path)
    filtered_files = [f.split(".")[0] for f in files if f.endswith(".txt")]
    return filtered_files

def show_identify_history(tag:str, fold:str, id:str):
    path = f"./chathistory/{tag}/identify_history/{id}.txt"
    with open(path, "r", encoding="utf-8") as f:  
        content = f.read()
    draw_conversations(content)


def show_traj_gen_history(tag:str, fold:str, id:str):
    path = f"./chathistory/{tag}/{fold}/history/{id}.txt"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    draw_conversations(content)
    
def show_llm_judge_history(tag:str, fold:str, id:str):
    path = f"./chathistory/{tag}/{fold}/llmjudge/{id}_llmjudge.txt"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    draw_conversations(content)
    
def load_pkl_from_id(tag, fold, id):
    base_path = f"./chathistory/{tag}/{fold}/traj/result/{tag}/"
    
    tmp = os.listdir(base_path + "generated")
    # 保证只有一种方式
    assert len(tmp) == 1 
    tmp = tmp[0]
    traj_g_path = os.path.join(base_path + "generated", tmp, f"{id}", "results.pkl")
    traj_g = list(pickle.load(open(traj_g_path, "rb")).values())
    
    tmp = os.listdir(base_path + "ground_truth")
    assert len(tmp) == 1 
    tmp = tmp[0]
    traj_r_path = os.path.join(base_path + "ground_truth", tmp, f"{id}", "results.pkl")
    traj_r = list(pickle.load(open(traj_r_path, "rb")).values())
    
    return parser_actyls(traj_r), parser_actyls(traj_g)
    
    
    
## page5 - 轨迹生成工具辅助函数
def get_person_choices(dataset: str, limit: int = 20):
    """获取可选的个体ID列表（前20个）"""
    from front.defines import available2019, available1921
    if dataset == "2019":
        return available2019[:limit]
    elif dataset == "20192021":
        return available1921[:limit]
    return []

def get_max_days(dataset: str, person_id: int):
    """获取测试轨迹的最大天数"""
    with open(f"./data/{dataset}/{person_id}.pkl", "rb") as f:
        data = pickle.load(f)
        # data[1] 是 test_routine_list
        return len(data[1])

def save_results_to_frontdata(real_traj: dict, gen_traj: dict, person_id: int,
                               dataset: str, mode: int, model: str):
    """保存结果到frontdata文件夹"""
    mode_name = {0: "llm_l", 1: "llm_e", 2: "llm_nm", 3: "llm_hybrid"}
    save_dir = f"./frontdata/{dataset}/{mode_name[mode]}/{model}/{person_id}"
    os.makedirs(save_dir, exist_ok=True)
    
    with open(f"{save_dir}/real_traj.pkl", "wb") as f:
        pickle.dump(real_traj, f)
    with open(f"{save_dir}/gen_traj.pkl", "wb") as f:
        pickle.dump(gen_traj, f)
    return save_dir

def load_results_from_frontdata(person_id: int, dataset: str, mode: int, model: str):
    """从frontdata加载结果"""
    mode_name = {0: "llm_l", 1: "llm_e", 2: "llm_nm", 3: "llm_hybrid"}
    save_dir = f"./frontdata/{dataset}/{mode_name[mode]}/{model}/{person_id}"
    
    real_path = f"{save_dir}/real_traj.pkl"
    gen_path = f"{save_dir}/gen_traj.pkl"
    
    if not os.path.exists(real_path) or not os.path.exists(gen_path):
        return None, None
    
    with open(real_path, "rb") as f:
        real_traj = pickle.load(f)
    with open(gen_path, "rb") as f:
        gen_traj = pickle.load(f)
    
    # 转换为可视化格式
    real_actyls = parser_actyls(list(real_traj.values()))
    gen_actyls = parser_actyls(list(gen_traj.values()))
    
    return real_actyls, gen_actyls

def has_results_in_frontdata(person_id: int, dataset: str, mode: int, model: str):
    """检查frontdata中是否存在结果"""
    mode_name = {0: "llm_l", 1: "llm_e", 2: "llm_nm", 3: "llm_hybrid"}
    save_dir = f"./frontdata/{dataset}/{mode_name[mode]}/{model}/{person_id}"
    return os.path.exists(f"{save_dir}/real_traj.pkl") and os.path.exists(f"{save_dir}/gen_traj.pkl")

def load_persona_mid_result(ds):
    """加载persona中间结果"""
    try:
        with open(rf"persona_result\results{ds}.txt", "r") as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            res = dict()
            for i in range(0, len(content), 2):
                res[int(content[i].split("-")[1])] = content[i+1]
            return res
    except Exception as e:
        print(e)
        return dict()

loc_map = load_loc_map()
