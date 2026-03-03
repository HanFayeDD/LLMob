import os
import pickle
import re
## 辅助函数
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
    global loc_map
    person = pickle.load(open(f'./data/{tag}/{id}.pkl', 'rb'))
    actyls = person[0] + person[1]
    return parser_actyls(actyls)
    
    
    
loc_map = load_loc_map()