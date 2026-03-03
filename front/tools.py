import os
import pickle
import re
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

loc_map = load_loc_map()
