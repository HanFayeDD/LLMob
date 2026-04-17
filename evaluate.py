import pickle
import numpy as np
import scipy.stats
import math
from datetime import datetime
from math import sin, cos, asin, sqrt, radians
import os
import argparse
from engine.llm_configs.poe_api import PoeAPI as LLMJudge
from engine.prompt_template.prompt_paths import *
from engine.utilities.process_tools import *
from engine.llm_configs.gpt_structure import *
from engine.utilities.retrieval_helper import *
import engine.llm_configs.gpt_structure as gpt_structure
import matplotlib.pyplot as plt 
import seaborn as sns 
plt.rcParams['font.family'] = 'Times New Roman'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sns.set_theme(style="darkgrid", font="Times New Roman")

def invert_dict(d):
    return {value: key for key, value in d.items()}


def load_pickle(file_path):
    """Utility function to load a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_persona_mid_result(ds):
    '''
    load_persona_mid_result 的 Docstring
    根据dataset分文件读取用户的个性化特征
    '''
    try:
        with open(rf"persona_result\results{ds}.txt", "r") as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            res = dict()
            for i in range(0, len(content), 2):
                if i + 1 < len(content):
                    res[int(content[i].split("-")[1])] = content[i+1]
            return res
    except Exception as e:
        print(f"Error loading persona result for dataset {ds}: {e}")
        return dict()


# Key: location name + latitude + longitude
# Value: ID in the city network
pos_map = load_pickle('./data/pos_map.pkl')

# Key: "location name + latitude + longitude" (to ensure uniqueness)
# Value: "location name + a unique ID for this location (same name locations get different IDs)"
loc_map = load_pickle('./data/loc_map.pkl')

# Key: location name, Value: category from foursquare
cat = load_pickle('./data/location_activity_map.pkl')
map_loc = invert_dict(loc_map)


def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000
    distance = round(distance / 1000, 3)
    return distance


def calculate_intervals_to_midnight(times, interval=10):
    midnight = datetime.strptime('00:00:00', '%H:%M:%S')
    intervals = []
    for time in times:
        if time.strip('.') == "24:00":
            time = "23:59"
        try:
            current_time = datetime.strptime(time.strip('.'), '%H:%M:%S')
        except:
            current_time = datetime.strptime(time.strip('.'), '%H:%M')
        time_diff_minutes = (current_time - midnight).seconds / 60
        number_of_intervals = time_diff_minutes // interval
        intervals.append(number_of_intervals)
    return intervals


def clean_traj(traj):
    acts = traj.split(": ")[-1]
    acts = acts.replace(", ", " at ").replace("Indulge in ", "").replace("Try in ", "").replace("Grab a quick bite at ",
                                                                                                "").replace("Try ", "")
    acts = acts.replace("Car Dealership", "Auto Dealership").replace("Enjoy ", "").replace("Mall", "Shopping Mall")
    acts = acts.replace("Outlet Shopping Mall", "Shopping Mall").replace("Shopping Shopping Mall", "Shopping Mall")
    acts = acts.replace("Relax at ", "").replace("Experience ", "").replace("Ramem Restaurant", "Ramen Restaurant")
    acts = acts.replace("Drop by ", "").replace("Stop by ", "").replace("End the day at ", "").replace("Visit ", "")
    acts = acts.replace("Go to ", "").replace("Sip coffee at ", "").replace("Noodle Restaurant", "Noodle House")
    acts = acts.replace("Explore ", "").replace("Visit ", "").replace("Shopping at ", "").replace("Lunch at ", "")
    acts = acts.replace("Savor ", "").replace("Discover ", "")
    return acts


def duration(p):
    d = [[i[0] - u[index][0] for index, i in enumerate(u[1:])] for u in p]
    d = [round(i * 10) for u in d for i in u]
    return d


def obtain_analysis_traj(data):
    traj_ids = []
    traj_lat_lngs = []
    traj_act_ts = []
    for d, traj in data.items():
        traj = data[d]
        if ": : " in traj:
            traj = traj.replace(": : ", ": ")
        if " :" in traj:
            traj = traj.replace(", ", "")
        traj_acts = clean_traj(traj)
        loc_times = traj_acts.split(" at ")
        locs = [] ## 地点
        times = [] ## 时间
        acts = [] ## 活动类别
        k = 0
        while k < len(loc_times):
            loc_times[k] = loc_times[k].replace(".", "")
            if k % 2 == 0:
                try:
                    clean_loc = loc_times[k]
                except:
                    clean_loc = clean_loc.split("#")[0] + str(1)
                    print(clean_loc)
                if "Home" in clean_loc or "home" in clean_loc:
                    k += 2
                    continue
                try:
                    acts.append(cat[loc_times[k].split("#")[0].strip()])
                except Exception as e:
                    print(e)
                    print(traj)
                    k += 2
                    continue
                locs.append(loc_times[k])

            else:
                times.append(loc_times[k].split(" ")[0])
            k += 1
        times_interval = calculate_intervals_to_midnight(times)
        traj_id, traj_lat_lng, traj_act_t = [], [], []
        for i in range(len(locs)):
            if "Home" in locs[i] or "home" in locs[i]:
                continue
            try:
                loc_with_lat_lng = map_loc[locs[i].strip()]
            except:
                continue
            loc_with_lat_lng_ = loc_with_lat_lng.replace(" (", ", ")
            loc_with_lat_lng_ = loc_with_lat_lng_.replace(")", "")
            lat_lng = [float(loc_with_lat_lng_.split(", ")[1]), float(loc_with_lat_lng_.split(", ")[2])]
            loc_id = pos_map[loc_with_lat_lng]
            t = times_interval[i]
            ## 添加一个轨迹点
            traj_id.append([loc_id, t]) ## 地点id、时间间隙
            traj_act_t.append([acts[i], t]) ## 活动类型、时间间隙
            traj_lat_lng.append([lat_lng[0], lat_lng[1], t]) ## 经纬度、时间间隙
        traj_ids.append(traj_id)
        traj_act_ts.append(traj_act_t) ## 添加天
        traj_lat_lngs.append(traj_lat_lng)
    return traj_ids, traj_lat_lngs, traj_act_ts


p2id = {'Travel & Transport': 0, 'Food': 1, 'Shop & Service': 2,
        'Nightlife Spot': 3, 'Arts & Entertainment': 4, 'Professional & Other Places': 5,
        'Outdoors & Recreation': 6,
        'College & University': 7, 'Residence': 8, 'Event': 9}


def transfer(data):
    '''
    transfer 的 Docstring
    传入的是一个人的所有数据
    return [时间间隔, 类型ID, (纬度, 经度)]
    :param data: 说明
    '''
    transfer_data = []
    locs_id = data[0]
    lat_lngs = data[1]
    acts = data[2]
    ## 对应此处上一个函数的返回
    for i in range(len(locs_id)): ## 天
        this_day = []
        for j in range(len(lat_lngs[i])): ## 哪一个点
            this_day.append([locs_id[i][j][1], p2id[acts[i][j][0]], [lat_lngs[i][j][0], lat_lngs[i][j][1]]])
        sorted_this_day = sorted(this_day, key=lambda x: x[0])
        transfer_data.append(sorted_this_day)
    return transfer_data


class Evaluation(object):
    
    GENERATED = "Generated"
    REAL = "Real"
    def __init__(self, args):
        self.args = args    

    def arr_to_distribution(self, arr, Min, Max, bins):
        ## 得到计数与下边界
        distribution, base = np.histogram(arr[arr <= Max], bins=bins, range=(Min, Max))
        m = np.array([len(arr[arr > Max])], dtype='int64')
        distribution = np.hstack((distribution, m))
        return distribution, base

    def get_js_divergence(self, p1, p2):
        ## TODO:figure out
        ## 长度是否需要一致
        ## 里面的位置是否很关键
        p1 = p1 / (p1.sum() + 1e-9)
        p2 = p2 / (p2.sum() + 1e-9)
        m = (p1 + p2) / 2
        js = 0.5 * scipy.stats.entropy(p1, m) + 0.5 * scipy.stats.entropy(p2, m)
        return js

    def distance_one_step(self, p1, p2):
        ## u是一天的轨迹
        ## 错开计算
        f = [geodistance(i[2][0], i[2][1], u[index][2][0], u[index][2][1]) for u in p1 for index, i in enumerate(u[1:])] 
        r = [geodistance(i[2][0], i[2][1], u[index][2][0], u[index][2][1]) for u in p2 for index, i in enumerate(u[1:])]
        self.draw_fr_distribution(f, r, "SD")
        MIN = 0
        MAX = 10
        bins = math.ceil(MAX - MIN)
        r_list, sep = self.arr_to_distribution(np.array(r), MIN, MAX, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), MIN, MAX, bins)
        print("SD")
        print(f_list)
        print(r_list)
        print(sep)
        self.draw_box_bar(f_list, r_list, sep, "SD Binning Result",
                          "Bin Lower Boundary (Km)")

        JSD = self.get_js_divergence(r_list, f_list)
        return JSD
    
    def duration_jsd(self, p1, p2):
        f = duration(p1)
        r = duration(p2)
        
        self.draw_fr_distribution(f, r, "SI")
        ## fix me
        MIN = 0
        MAX = 240
        bins = math.ceil((MAX - MIN) / 10)
        r_list, sep = self.arr_to_distribution(np.array(r), MIN, MAX, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), MIN, MAX, bins)
        print("SI")
        print(f_list)
        print(r_list)
        print(sep)
        self.draw_box_bar(f_list, r_list, sep, "SI Binning Result",
                          "Bin Lower Boundary (Min)")
        # self.draw_fr_distribution(f_list, r_list, sep, "SI_yes_bins_no_standard")
        JSD = self.get_js_divergence(r_list, f_list)
        return JSD
    
    def st_act_jsd(self, p1, p2):
        st_act_dict = {}
        for u in p1:
            for i in u:
                if str(i[0]) + '_' + str(i[1]) not in st_act_dict:
                    st_act_dict[str(i[0]) + '_' + str(i[1])] = len(st_act_dict)
        for u in p2:
            for i in u:
                if str(i[0]) + '_' + str(i[1]) not in st_act_dict:
                    st_act_dict[str(i[0]) + '_' + str(i[1])] = len(st_act_dict)
        # st_act_dict: 为每个轨迹点创建键：str(时间间隔) + '_' + str(活动ID)（活动ID 通过 p2id 映射）。将所有键映射到唯一ID
        f, r = [], []
        for u in p1:
            for i in u:
                f.append(st_act_dict[str(i[0]) + '_' + str(i[1])])
        for u in p2:
            for i in u:
                r.append(st_act_dict[str(i[0]) + '_' + str(i[1])])
        MIN = np.min(r + f)
        MAX = np.max(r + f)
        #TODO 分箱参数
        bins = 1000
        r = (np.array(r) - MIN) / (MAX - MIN)
        f = (np.array(f) - MIN) / (MAX - MIN)
        r_list, sep = self.arr_to_distribution(r, 0, 1, bins)
        f_list, _ = self.arr_to_distribution(f, 0, 1, bins)
        assert len(f_list) == len(r_list)
        assert len(sep) == len(f_list)
        JSD = self.get_js_divergence(r_list, f_list)
        return JSD
    
    def st_act_jsd_v2(self, p1, p2):
        st_act_dict = {}
        order = set()
        id2p = revert_dict(p2id)
        data_to_draw_2d = []
        
        for u in p1:
            for i in u:
                if str(i[0]) + '_' + str(i[1]) not in order:
                    order.add(str(i[0]) + '_' + str(i[1]))
        for u in p2:
            for i in u:
                if str(i[0]) + '_' + str(i[1]) not in order:
                    order.add(str(i[0]) + '_' + str(i[1]))
        
        st_act_dict = {pair: idx for idx, pair in enumerate(sorted(order))}
        # st_act_dict: 为每个轨迹点创建键：str(时间间隔) + '_' + str(活动ID)（活动ID 通过 p2id 映射）。将所有键映射到唯一ID
        f, r = [], []
        from collections import Counter
        for u in p1:
            for i in u:
                data_to_draw_2d.append([int(i[0]), i[1], Evaluation.GENERATED])
                f.append(st_act_dict[str(i[0]) + '_' + str(i[1])])
        for u in p2:
            for i in u:
                data_to_draw_2d.append([int(i[0]), i[1], Evaluation.REAL])
                r.append(st_act_dict[str(i[0]) + '_' + str(i[1])])
        
        st_act_dict_reverse = revert_dict(st_act_dict)
        fcnt = Counter(f)
        rcnt = Counter(r)
        top_n = 10
        print("*"*5 + "DARD" + "*"*5)
        print(f"keys num:{len(st_act_dict)}")
        print(f"generated:{len(f)}")
        print(f"real:{len(r)}")
        top_n_f = fcnt.most_common(top_n)
        top_n_r = rcnt.most_common(top_n)
        print("Generated")
        for ele in top_n_f:
            numid = ele[0]
            cnt = ele[1]
            strid = st_act_dict_reverse[numid]
            a, b = strid.split("_")
            a = int(float(a))
            b = id2p[int(b)]
            print(f"Numid:{numid} Time:{a//6}:{(a%6)*10} Activity Type:{b} Count:{cnt}")
        print("Real")
        for ele in top_n_r:
            numid = ele[0]
            cnt = ele[1]
            strid = st_act_dict_reverse[numid]
            a, b = strid.split("_")
            a = int(float(a))
            b = id2p[int(b)]
            print(f"Numid:{numid} Time:{a//6}:{(a%6)*10} Activity Type:{b} Count:{cnt}")
        print("*"*10)
        
        self.draw_fr_distribution(f, r, "DARD")
        self.draw_fr_distribution_2d(data_to_draw_2d, "DARD 2d", ["Time Interval Id (10 mins an interval)", "Activity Type Id"])
        
        MIN = np.min(r + f)
        MAX = np.max(r + f)
        #TODO 分箱参数
        bins = 1000
        r = (np.array(r) - MIN) / (MAX - MIN)
        f = (np.array(f) - MIN) / (MAX - MIN)
        r_list, sep = self.arr_to_distribution(r, 0, 1, bins)
        f_list, _ = self.arr_to_distribution(f, 0, 1, bins)
        # self.draw_fr_distribution(f_list, r_list, sep, "DARD")
        assert len(f_list) == len(r_list)
        assert len(sep) == len(f_list)
        JSD = self.get_js_divergence(r_list, f_list)
        return JSD

    def st_loc_jsd(self, p1, p2):
        st_act_dict = {}
        for u in p1:
            for i in u:
                if str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1]) not in st_act_dict:
                    st_act_dict[str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1])] = len(st_act_dict)
        for u in p2:
            for i in u:
                if str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1]) not in st_act_dict:
                    st_act_dict[str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1])] = len(st_act_dict)
        f, r = [], []
        for u in p1:
            for i in u:
                f.append(st_act_dict[str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1])])
        for u in p2:
            for i in u:
                r.append(st_act_dict[str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1])])
        MIN = np.min(r + f)
        MAX = np.max(r + f)
        bins = 400 #TODO 分箱参数
        r = (np.array(r) - MIN) / (MAX - MIN)
        f = (np.array(f) - MIN) / (MAX - MIN)
        r_list, sep = self.arr_to_distribution(np.array(r), 0, 1, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), 0, 1, bins)
        assert len(f_list) == len(r_list)
        assert len(sep) == len(f_list)
        JSD = self.get_js_divergence(r_list, f_list)
        return JSD
    
    def st_loc_jsd_v2(self, p1, p2):
        # TODO:优化排序逻辑
        # TODO:标准化、分桶、绘图过程不统一
        st_act_dict = {}
        order = set()
        for u in p1:
            for i in u:
                if str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1]) not in order:
                    order.add(str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1]))
        for u in p2:
            for i in u:
                if str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1]) not in order:
                    order.add(str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1]))
                    
        st_act_dict = {pair: idx for idx, pair in enumerate(sorted(order))}
        print("STVD_v2 keys nums:" + str(len(st_act_dict)))
        f, r = [], []
        for u in p1:
            for i in u:
                f.append(st_act_dict[str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1])])
        for u in p2:
            for i in u:
                r.append(st_act_dict[str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1])])
        
        self.draw_fr_distribution(f, r, "STVD_v2")
        
        MIN = np.min(r + f)
        MAX = np.max(r + f)
        bins = 400 #TODO 分箱参数
        r = (np.array(r) - MIN) / (MAX - MIN)
        f = (np.array(f) - MIN) / (MAX - MIN)
        r_list, sep = self.arr_to_distribution(np.array(r), 0, 1, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), 0, 1, bins)
        assert len(f_list) == len(r_list)
        assert len(sep) == len(f_list)
        JSD = self.get_js_divergence(r_list, f_list)
        return JSD
    
    def st_loc_jsd_v3(self, p1, p2, rd=2):
        def get_float_round_str(ipt, left):
            return str(round(ipt, left))
        st_act_dict = {}
        order = set()
        
        area_set = set()
        area_dict = set()
        for u in p1:
            for i in u:
                if str(i[0]) + '_' + get_float_round_str(i[2][0], rd) + '_' + get_float_round_str(i[2][1], rd) not in order:
                    order.add(str(i[0]) + '_' + get_float_round_str(i[2][0], rd) + '_' + get_float_round_str(i[2][1], rd))
                    area_set.add(get_float_round_str(i[2][0], rd) + '_' + get_float_round_str(i[2][1], rd))
        for u in p2:
            for i in u:
                if str(i[0]) + '_' + get_float_round_str(i[2][0], rd) + '_' + get_float_round_str(i[2][1], rd) not in order:
                    order.add(str(i[0]) + '_' + get_float_round_str(i[2][0], rd) + '_' + get_float_round_str(i[2][1], rd))
                    area_set.add(get_float_round_str(i[2][0], rd) + '_' + get_float_round_str(i[2][1], rd))
                    
        st_act_dict = {pair: idx for idx, pair in enumerate(sorted(order))}
        area_dict = {ele: idx for idx, ele in enumerate(sorted(area_set))}
        f, r = [], []
        data_to_draw_2d = []
        for u in p1:
            for i in u:
                f.append(st_act_dict[str(i[0]) + '_' + get_float_round_str(i[2][0], rd) + '_' + get_float_round_str(i[2][1], rd)])
                data_to_draw_2d.append([int(i[0]), 
                                        area_dict[get_float_round_str(i[2][0], rd) + '_' + get_float_round_str(i[2][1], rd)],
                                        Evaluation.GENERATED])
        for u in p2:
            for i in u:
                r.append(st_act_dict[str(i[0]) + '_' + get_float_round_str(i[2][0], rd) + '_' + get_float_round_str(i[2][1], rd)])
                data_to_draw_2d.append([int(i[0]), 
                                        area_dict[get_float_round_str(i[2][0], rd) + '_' + get_float_round_str(i[2][1], rd)],
                                        Evaluation.REAL])
        
        st_act_dict_reverse = revert_dict(st_act_dict)
        fcnt = Counter(f)
        rcnt = Counter(r)
        top_n = 10
        print("*"*5 + f"STVD_v3_float_{rd}" + "*"*5)
        print(f"keys num:{len(st_act_dict)}")
        print(f"generated:{len(f)}")
        print(f"real:{len(r)}")
        top_n_f = fcnt.most_common(top_n)
        top_n_r = rcnt.most_common(top_n)
        print("Generated")
        for ele in top_n_f:
            numid = ele[0]
            cnt = ele[1]
            strid = st_act_dict_reverse[numid]
            a, b, c = strid.split("_")
            a = int(float(a))
            b = round(float(b), rd)
            c = round(float(c), rd)
            print(f"Numid:{numid} Time:{a//6}:{(a%6)*10} Lat:{b} Lng:{c} Count:{cnt}")
        print("Real")
        for ele in top_n_r:
            numid = ele[0]
            cnt = ele[1]
            strid = st_act_dict_reverse[numid]
            a, b, c = strid.split("_")
            a = int(float(a))
            b = round(float(b), rd)
            c = round(float(c), rd)
            print(f"Numid:{numid} Time:{a//6}:{(a%6)*10} Lat:{b} Lng:{c} Count:{cnt}")
        print("*"*10)
        
        self.draw_fr_distribution(f, r, f"STVD_v3_float_{rd}")
        self.draw_fr_distribution_2d(data_to_draw_2d, 
                                     f"STVD 2d float {rd}",
                                     ["Time Interval Id (10 mins an interval)", "Geo Location Area Id"])
           
        MIN = np.min(r + f)
        MAX = np.max(r + f)
        bins = 400 #TODO 分箱参数
        r = (np.array(r) - MIN) / (MAX - MIN)
        f = (np.array(f) - MIN) / (MAX - MIN)
        r_list, sep = self.arr_to_distribution(np.array(r), 0, 1, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), 0, 1, bins)
        assert len(f_list) == len(r_list)
        assert len(sep) == len(f_list)
        # self.draw_fr_distribution(f_list, r_list, sep, f"STVD_v3_float_{rd}")
        JSD = self.get_js_divergence(r_list, f_list)
        return JSD
    
    def draw_fr_distribution(self, f, r, figname):
        """_summary_
            f、r、sep长度相同。本质是单变量绘图
        Args:
            f (_type_): _description_
            r (_type_): _description_
            sep (_type_): _description_
        """
        xlabel = {"SD":"Km",
                  "SI":"Min",
                  "DARD":"Time Interval && Activity Type Id",
                  "STVD":"Time Interval && Geo Location AreaId"}
        plt.figure(dpi=400)
        sns.kdeplot(f, label="Generated")
        sns.kdeplot(r, label="Real")
        label = ""
        for k in xlabel.keys():
            if k in figname:
                label = xlabel[k]
                break
        plt.xlabel(label)
        plt.ylabel("Kde Density")
        plt.legend()
        plt.title(f"{figname} Distribution (Kdeplot)", fontsize=20)
        plt.tight_layout()
        plt.savefig(f"{figname} Distribute (Kdeplot).png")
        # plt.show()
        plt.close()
        
    def draw_box_bar(self, f, r, sep, figname, xlabel="Bin Lower Boundary", ylabel="Cnt"):
        df = pd.DataFrame()
        sep = [int(ele) for ele in sep]
        # sns.set_theme(style="darkgrid")
        plt.close()
        df['Cnt'] = f.tolist() + r.tolist()
        df['Tag'] = ["Generated"] * len(f) + ["Real"] * len(r)
        df['Bound'] = sep + sep
        g = sns.barplot(df, x="Bound", y="Cnt", hue="Tag")
        g.set_xlabel(xlabel)
        if len(sep) >= 15:
            ## 部分x轴的值重合，帮我设置间隔
            step = 3
            current_ticks = g.get_xticks()
            new_ticks = current_ticks[::step]
            new_labels = sep[::step]
            g.set_xticks(new_ticks)
            g.set_xticklabels(new_labels)
            
        g.set_ylabel(ylabel)
        g.set_title(f"{figname}", fontsize=20)
        legend = g.get_legend()
        if legend:
            # 将图例标题设置为空字符串
            legend.set_title("")
        plt.tight_layout()
        plt.savefig(f"{figname}.png", dpi=400)
        
    def draw_fr_distribution_2d(self, data, figname, colnames=None):
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # 注意：jointplot 会创建自己的 figure，前面的 plt.figure(dpi=800) 可能不会生效
        # 建议在最后保存时控制 dpi，或者使用 context 设置
        
        # sns.set_theme(style="darkgrid")
        if colnames is None:
            colnames = ["col1", "col2", "Tag"]
        elif type(colnames) == list and len(colnames) == 2:
            colnames.append("Tag")
        else:
            raise ValueError("")
            
        df = pd.DataFrame(data=data, columns=colnames)
        
        g = sns.jointplot(
            data=df,
            x=colnames[0], y=colnames[1], hue=colnames[2],
            kind="kde"
        )
        
        # =======================================================
        # 【修改部分】：去掉图例标题
        # =======================================================
        # 获取主绘图区(ax_joint)的图例对象
        legend = g.ax_joint.get_legend()
        if legend:
            # 将图例标题设置为空字符串
            legend.set_title("")
        # =======================================================

        # 注意：对于 jointplot，建议使用 g.fig.suptitle 来设置总标题，
        # 并需要调整顶部边距(y参数)以防重叠
        g.figure.suptitle(f"{figname} Distribution (Kdeplot)", fontsize=20, y=1.03)
        
        # g.figure.tight_layout() # jointplot 内部已有布局管理，有时 tight_layout 会冲突，视情况保留
        g.figure.set_dpi(400)
        
        plt.savefig(f"{figname} Distribute (Kdeplot).png", dpi=400, bbox_inches='tight')
            
    
    def get_JSD(self, real, fake):
        """_summary_

        Args:
            real (_type_): _description_
            fake (_type_): _description_

        Returns:
            _type_: _description_
            SI SD DARD STVD 
            SI、SD计算方法类似
        """
        #TODO: 时间、活动类型、地点三个的联合分布
        duration_jsd = self.duration_jsd(fake, real) # 时长
        distance_step = self.distance_one_step(fake, real) # 距离
        st_act_jsd = self.st_act_jsd_v2(fake, real) # DVRD:时间 + 活动类型
        stvd1 = self.st_loc_jsd(fake, real) # STVD:时间 + 经纬度
        stvd2 = self.st_loc_jsd_v2(fake, real) # STVD:时间 + 经纬度
        stvd3_float0 = self.st_loc_jsd_v3(fake, real, 0) # STVD:时间 + 经纬度
        st_loc_jsd   = self.st_loc_jsd_v3(fake, real, 1) # STVD:时间 + 经纬度
        stvd3_float2 = self.st_loc_jsd_v3(fake, real, 2) # STVD:时间 + 经纬度
        print(f"STVD:\nV1: {stvd1:.4f}\nV2: {stvd2:.4f}\nV3_float0: {stvd3_float0:.4f}\nV3_float1: {st_loc_jsd:.4f}\nV3_float2: {stvd3_float2:.4f}")
        return duration_jsd, distance_step, st_act_jsd, st_loc_jsd

def llm_as_judge_one_day(id, judge, date_, t, f):
    week_day = date_to_weekday(date_)
    critic_prompt_template_path = r"engine\prompt_template\llmjudge.txt"
    ipt_data = [f"{date_}(is {week_day})", t, f]
    prompt = generate_prompt(ipt_data, critic_prompt_template_path)
    while True:
        contents = execute_prompt(prompt, judge,
                                  objective=f"llm judge...{id}/{date_}")
        try:
            score = int(re.search(r'\d+', contents).group())
        except:
            continue
        break
    print(score)
    return score


def llm_as_judge_preference_rate(id, judge, persona, date_, real_traj, gen_traj, scenario):
    """
    根据用户的个性化特征、真实轨迹、输出轨迹，计算模型的偏好率。
    将三个参数传给llm，问他哪个轨迹更符合个性化特征并给出原因。
    
    Args:
        id: 用户ID
        judge: LLM Judge实例
        persona: 用户的个性化特征
        date_: 日期
        real_traj: 真实轨迹
        gen_traj: 生成轨迹
    
    Returns:
        int: 0表示真实轨迹更符合，1表示生成轨迹更符合
    """
    preference_prompt_template_path = r"engine\prompt_template\preference_rate.txt"
    hint = ""
    if "abnormal" in scenario:
            hint = '''Now it is the pandemic period. The government has asked residents to postpone travel and events and to telecommute as much as possible.'''
            hint = hint.replace("\n", " ").strip()
    ipt_data = [persona, real_traj, gen_traj, hint]
    prompt = generate_prompt(ipt_data, preference_prompt_template_path)
    while True:
        contents = execute_prompt(prompt, judge,
                                  objective=f"preference rate judge...{id}/{date_}")
        try:
            # 从输出中提取数字（0或1）
            print(contents.strip())
            match = re.search(r'\b[01]\b', contents.strip())
            print(match.group())
            if match:
                preference = int(match.group())
                if preference in [0, 1]:
                    break
            # 如果没有找到0或1，尝试从文本中判断
            if "real" in contents.lower() or "真实" in contents:
                preference = 0
                break
            elif "generated" in contents.lower() or "生成" in contents:
                preference = 1
                break
        except Exception as e:
            print(f"Error parsing preference result: {e}, content: {contents}")
            continue
    print(f"Preference for {id}/{date_}: {preference}")
    return preference
        

def eval(dataset='normal', mode=0):
    # Load required data
    mode_name = {0: "llm_l", 1: "llm_e", 2:"llm_nm", 3:"llm_hybrid"}
    mode = mode_name[mode]
    truth = {}
    person_to_test = []
    scenario_tag = {
        '2019': 'normal',
        '2021': 'abnormal',
        '20192021': 'normal_abnormal'
    }
    scenario = scenario_tag[dataset]
    
    # Load persona data if preference rate evaluation is enabled
    persona_dict = dict()
    if args.preference:
        persona_dict = load_persona_mid_result(dataset)
        print(f"Loaded persona data for {len(persona_dict)} users")
    # Define paths
    ground_truth_paths = {
        'normal': f'./result/normal/ground_truth/{mode}/',
        'abnormal': f'./result/abnormal/ground_truth/{mode}/',
        'normal_abnormal': f'./result/normal_abnormal/ground_truth/{mode}/'
    }
    generated_paths = {
        'normal': f'./result/normal/generated/{mode}/',
        'abnormal': f'./result/abnormal/generated/{mode}/',
        'normal_abnormal': f'./result/normal_abnormal/generated/{mode}/'
    }

    # Choose the last defined path
    ground_truth_path = ground_truth_paths[scenario]
    gen_path = generated_paths[scenario]
    folders = [d for d in os.listdir(ground_truth_path) if os.path.isdir(os.path.join(ground_truth_path, d))]
    real_traj = dict()
    # Process ground truth data
    for f in folders:
        person = load_pickle(os.path.join(ground_truth_path, f + '/results.pkl'))
        real_traj[f] = person
        person_id = f
        test_traj_ids, test_lat_lngs, test_act_ts = obtain_analysis_traj(person)

        truth[person_id] = {
            "test": [test_traj_ids, test_lat_lngs, test_act_ts, person]
        }

        person_to_test.append(person_id)

    # Load generated data
    gen_traj = {}
    gen = {}
    for p in person_to_test:
        gen_key = f"{p}_{mode}"
        gen[gen_key] = []
        try:
            result_path = os.path.join(gen_path, p, 'results.pkl')
            res = load_pickle(result_path)
            gen_traj[p] = res 
            res_traj_ids, res_traj_lat_lngs, res_traj_acts = obtain_analysis_traj(res)
            gen[gen_key].append([res_traj_ids, res_traj_lat_lngs, res_traj_acts])
        except FileNotFoundError:
            pass

    # Prepare data for evaluation
    # [时间间隔, 活动类型ID, (纬度, 经度)]
    gen_data, real_data = {}, {}
    for p in person_to_test:
        gen_key = f"{p}_{mode}"
        for i in range(len(gen[gen_key])): ## 一个i是一个人的所有数据
            if mode not in gen_data:
                gen_data[mode] = transfer(gen[gen_key][i])
                real_data[mode] = transfer(truth[p]["test"])
            else:
                gen_data[mode].extend(transfer(gen[gen_key][i]))
                real_data[mode].extend(transfer(truth[p]["test"]))

    # Initialize evaluation results
    evaluation = Evaluation(None)
    duration_jsd_dict, st_act_jsd_dict = {}, {}
    distance_step_dict, st_loc_jsd_dict = {}, {}
    print(len(gen_data[mode])) ## 长度为多少天的轨迹数量
    print(len(real_data[mode]))
    # Compute evaluation metrics
    print(real_data[mode][0])
    print(gen_data[mode][0])
    
    duration_jsd, distance_step, st_act_jsd, st_loc_jsd = evaluation.get_JSD(real_data[mode], gen_data[mode])
    distance_step_dict[mode] = distance_step
    duration_jsd_dict[mode] = duration_jsd
    st_act_jsd_dict[mode] = st_act_jsd
    st_loc_jsd_dict[mode] = st_loc_jsd
    # print(type(distance_step))
    # print(type(duration_jsd))
    # print(type(st_act_jsd))
    # print(type(st_loc_jsd))

    print(f"{scenario}")
    # Print evaluation results
    
    #LLM as judge
    llmjudge = LLMJudge()
    llmjudge.set_model("gpt-3.5-turbo")
    if args.llmjudge:
        for k in real_traj:
            if k not in gen_traj:
                raise ValueError(f"Person {k} in real_traj but not in gen_traj")

        
        llm_judge_result = dict()
        size = 0
        sum_score = 0
        for k in person_to_test:
            llm_judge_result[k] = []
            log_filename = f"chathistory/{k}_llmjudge.txt"
            gpt_structure.set_current_log_file(log_filename)
            for date_ in real_traj[k]:
                t = real_traj[k][date_]
                f = gen_traj[k][date_]
                res = llm_as_judge_one_day(k, llmjudge, date_, t, f)
                llm_judge_result[k].append(res)
                sum_score += res
                size += 1
        
        print("LLM Judge Result:")
        for k in llm_judge_result:
            print(f"{k} mean score {np.mean(llm_judge_result[k]):.4f}")
            with open(f"llm_judge_result.txt", "a") as f:
                f.write(f"{k}\n{llm_judge_result[k]}\n{np.mean(llm_judge_result[k]):.4f}\n")
    
    print(
        f"{mode}: "
        f"SD: {np.mean(distance_step_dict[mode]):.4f}, " # 评估轨迹中相邻点之间地理距离的分布相似性
        f"SI: {np.mean(duration_jsd_dict[mode]):.4f}, "  # 评估轨迹中相邻点之间时间间隔（持续时间）的分布相似性
        f"DARD: {np.mean(st_act_jsd_dict[mode]):.4f}, " # 评估空间-时间-活动的联合分布相似性
        f"STVD: {np.mean(st_loc_jsd_dict[mode]):.4f}" # 评估空间-时间-位置的联合分布相似性。
    )
    print(f"{np.mean(distance_step_dict[mode]):.4f} & {np.mean(duration_jsd_dict[mode]):.4f} & {np.mean(st_act_jsd_dict[mode]):.4f} & {np.mean(st_loc_jsd_dict[mode]):.4f}")

    if args.llmjudge:
        print(f"LLM Judge Result: {sum_score / size:.4f}")
    
    # Preference Rate Evaluation
    if args.preference:
        print("\n" + "="*50)
        print("Starting Preference Rate Evaluation...")
        print("="*50)
        
        # Initialize LLM Judge for preference rate
        pref_judge = LLMJudge()
        pref_judge.set_model("gpt-3.5-turbo")
        
        preference_results = dict()
        total_preference_count = 0
        gen_preference_count = 0  # 生成轨迹被偏好的次数
        
        print(persona_dict)
        for k in person_to_test:
            if k not in real_traj or k not in gen_traj:
                continue
            if int(k) not in persona_dict:
                print(f"Warning: Person {k} not found in persona_dict, skipping...")
                continue
            
            persona = persona_dict[int(k)]
            preference_results[k] = []
            log_filename = f"chathistory/{k}_preference.txt"
            gpt_structure.set_current_log_file(log_filename)
            
            for date_ in real_traj[k]:
                if date_ not in gen_traj[k]:
                    continue
                t = real_traj[k][date_]
                f = gen_traj[k][date_]
                
                # 调用偏好率评估函数
                res = llm_as_judge_preference_rate(k, pref_judge, persona, date_, t, f, scenario)
                preference_results[k].append(res)
                total_preference_count += 1
                if res == 1:  # 1表示生成轨迹更符合
                    gen_preference_count += 1
        
        # 计算偏好率
        if total_preference_count > 0:
            preference_rate = gen_preference_count / total_preference_count
        else:
            preference_rate = 0.0
        
        print("\n" + "="*50)
        print("Preference Rate Evaluation Results:")
        print("="*50)
        for k in preference_results:
            if len(preference_results[k]) > 0:
                user_gen_pref = sum(preference_results[k])
                user_total = len(preference_results[k])
                user_pref_rate = user_gen_pref / user_total
                print(f"Person {k}: {user_gen_pref}/{user_total} = {user_pref_rate:.4f}")
                with open(f"preference_rate_result.txt", "a") as f:
                    f.write(f"{k}\n{preference_results[k]}\n{user_pref_rate:.4f}\n")
        
        print(f"\nOverall Preference Rate: {gen_preference_count}/{total_preference_count} = {preference_rate:.4f}")
        print(f"(Preference Rate = Number of times generated trajectory is preferred / Total number of trajectories)")
        print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Define arguments before parsing
    parser.add_argument('--dataset', type=str, default='2019',
                        help='Specify the dataset: ')
    parser.add_argument('--mode', type=int, default=0,
                        help='Specify the mode type: 0 for llm_l, 1 for llm_e')
    parser.add_argument('--llmjudge', type=int, default=0, help="whether use LLM as a Judge to eval")
    parser.add_argument('--preference', type=int, default=0, help="whether use LLM to evaluate preference rate")
    
    args = parser.parse_args()  # Parse after defining arguments

    # Call the eval function with parsed arguments
    eval(dataset=args.dataset, mode=args.mode)

# 1. SD (Spatial Distance, distance_step)
# 含义: 评估轨迹中相邻点之间地理距离的分布相似性。
# 计算过程:
# 对每个轨迹，计算相邻点之间的距离（使用 geodistance 函数，基于经纬度计算球面距离，单位为公里）。
# 将距离值分箱（bins=10，范围 0-10km），生成直方图分布。
# 使用 JSD 比较真实数据和生成数据的距离分布。
# 示例: 对于您的轨迹数据，计算如 "Baseball Stadium#244 at 12:00" 到 "Convenience Store#7139 at 13:30" 之间的距离。
# 2. SI (Spatial-temporal Interval, duration_jsd)
# 含义: 评估轨迹中相邻点之间时间间隔（持续时间）的分布相似性。
# 计算过程:
# 对每个轨迹，计算相邻点的时间差（分钟），乘以10并四舍五入（duration 函数）。
# 将时间间隔分箱（bins=12，范围 0-12），生成直方图分布。
# 使用 JSD 比较真实数据和生成数据的时间间隔分布。
# 示例: 对于 "12:00" 到 "13:30"，时间差为 90 分钟，转换为 900（乘10后）。
# 3. DARD (Dynamic Activity Representation Distance, st_act_jsd)
# 含义: 评估空间-时间-活动的联合分布相似性。
# 计算过程:
# 为每个轨迹点创建键：str(时间间隔) + '_' + str(活动ID)（活动ID 通过 p2id 映射）。
# 将所有键映射到唯一ID，生成分布（bins=1000，归一化到 0-1）。
# 使用 JSD 比较真实数据和生成数据的联合分布。
# 示例: 对于 "Baseball Stadium#244 at 12:00"，键可能为 "时间间隔_活动ID"，如 "0_244"（假设时间间隔从0开始）。
# 4. STVD (Spatial-temporal Variation Distance, st_loc_jsd)
# 含义: 评估空间-时间-位置的联合分布相似性。
# 计算过程:
# 为每个轨迹点创建键：str(时间间隔) + '_' + str(纬度) + '_' + str(经度)。
# 将所有键映射到唯一ID，生成分布（bins=400，归一化到 0-1）。
# 使用 JSD 比较真实数据和生成数据的联合分布。
# 示例: 对于 "Baseball Stadium#244 at 12:00"，键为 "时间间隔_纬度_经度"，如 "0_39.123_-77.456"。