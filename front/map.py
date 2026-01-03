import pickle
import re
import pandas as pd 
import pydeck as pdk
import streamlit as st
from enum import Enum
import os 
from front.defines import *

view_state = pdk.ViewState(
        latitude=35.69,
        longitude=139.703,
        bearing=0,
        pitch=20,
        zoom=5,
)

class DataTags(Enum):
    ALL = 1
    TRAIN = 2
    TESTGENERATED = 3
    TESTTRUTH = 4

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

def get_acticity_list(id:int, tag:DataTags=DataTags.ALL):
    global available1921, available2019, available2021
    def parse_place_time(text:str):
        place, time = text.split(" at ")
        return (place.strip(), time.strip())
    
    actyls = []
    # [
    # "Activities at 2019-01-02: Convenience Store#2322 at 12:00:00, Historic Site#2176 at 12:10:00, Platform#646 at 12:40:00, Shopping Mall#2228 at 12:50:00, Drugstore#1281 at 13:20:00, Ramen Restaurant#2062 at 13:40:00, Plaza#450 at 15:00:00, Clothing Store#521 at 15:50:00, Discount Store#177 at 16:10:00, Supermarket#627 at 16:50:00.",
    # ......
    # ]
    if tag in [DataTags.ALL, DataTags.TRAIN]:
        if id in available1921:
            person = pickle.load(open(rf"data\20192021\{id}.pkl", "rb"))
        elif id in available2019:
            person = pickle.load(open(rf"data\2019\{id}.pkl", "rb"))
        elif id in available2021:
            person = pickle.load(open(rf"data\2021\{id}.pkl", "rb"))
        else:
            st.error(f"{id} data is not found.")
        if tag == DataTags.ALL:
            actyls = person[0] + person[1]
        else:
            actyls = person[0]
    #     {
    #   "2019-10-21": "Activities at 2019-10-21: Japanese Restaurant#4268 at 10:10:00, Donburi Restaurant#1193 at 10:20:00, Shrine#207 at 10:40:00, Farmers Market#819 at 10:50:00, Pharmacy#1304 at 12:40:00, Supermarket#535 at 12:50:00, Winery#106 at 14:00:00, Rest Area#2152 at 14:10:00, Factory#991 at 16:20:00, Toll Booth#713 at 17:50:00, Toll Plaza#170 at 19:10:00, Shrine#2626 at 20:10:00, Liquor Store#946 at 21:00:00, Park#1775 at 21:20:00, Shrine#1839 at 22:40:00, Convenience Store#10197 at 22:50:00.",
    #   "2019-10-22": "Activities at 2019-10-22: Supermarket#535 at 14:10:00, Convenience Store#358 at 14:20:00, Town Hall#489 at 14:30:00, Drugstore#1370 at 16:20:00, Discount Store#833 at 16:40:00, Buddhist Temple#889 at 18:00:00, Shrine#1277 at 18:20:00, Rest Area#585 at 18:40:00, Cultural Center#114 at 20:30:00, Community Center#39 at 20:50:00, Convenience Store#10197 at 22:30:00."
    # }
    elif tag == DataTags.TESTTRUTH:
        absopath = os.path.join(os.getcwd(), rf"result\normal\ground_truth\llm_e\{id}\results.pkl")
        if os.path.exists(absopath):
            person:dict = pickle.load(open(absopath, "rb"))
            actyls = list(person.values())
            
    elif tag == DataTags.TESTGENERATED:
        absopath = os.path.join(os.getcwd(), rf"result\normal\generated\llm_e\{id}\results.pkl")
        if os.path.exists(absopath):
            person:dict = pickle.load(open(absopath, "rb"))
            actyls = list(person.values())
    
    if not actyls:
        st.error(f"No activities found for {id}.")
    
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

def plot_route(acticity_list:list, loc_map:dict):
    global view_state
    df = []
    for oneday in acticity_list:
        day, acts = oneday[0], oneday[1]
        for i in range(len(acts)-1):
            try:
                begin, end = acts[i], acts[i+1]
                # begin[0] = begin[0].replace(" #", "#")
                # end0] = begin[0].replace(" #", "#")
                
                bpos, bhour = loc_map[begin[0]], begin[1]
                epos, ehour = loc_map[end[0]], end[1]
            except:
                print(f"{acts[i]}, {acts[i+1]}  not in loc_map")
                continue
            df.append([day, bhour, begin[0], bpos[0], bpos[1], 
                            ehour, end[0], epos[0], epos[1]])
    colname = ["day", "b_hour", "b_name", "b_w", "b_j",
                      "e_hour", "e_name", "e_w", "e_j"]
    df = pd.DataFrame(df, columns=colname)
    
    arc_layer = pdk.Layer(
        "ArcLayer",
        data=df,
        get_width="count_std",
        get_source_position=["b_j", "b_w"],
        get_target_position=["e_j", "e_w"],
        get_tilt=15,
        get_source_color=[64, 255, 0, 100],#RED_RGB,
        get_target_color=[0, 128, 200, 100], #GREEN_RGB,
        pickable=True,
        auto_highlight=True,
    )
    TOOLTIP_TEXT = {"html": "{day}<br />起点： {b_hour} {b_name} <br />终点：{e_hour} {e_name}",
                    "style": {"backgroundColor": "steelblue", "color": "white"}}
    return arc_layer
    
def plot_htmap(acticity_list:list, loc_map:dict):
    global view_state
    dcnt = dict()
    dtime = dict()
    for oneday in acticity_list:
        day, acts = oneday[0], oneday[1]
        for i in range(len(acts)):
            pos, hour = acts[i][0], acts[i][1]
            if pos not in dcnt:
                dcnt[pos] = 1
                dtime[pos] = [day+"-"+hour]
            else:
                dcnt[pos] += 1
                dtime[pos].append(day+"-"+hour)
    data = []
    for k, v in dcnt.items():
        try:
            pos_wj = loc_map[k]
            data.append([k, v, pos_wj[0], pos_wj[1], "".join(dtime.get(k, []))])
        except:
            continue
    colnames = ["name", "cnt", "wd", "jd", "time"]
    df = pd.DataFrame(data, columns=colnames)
    COLOR_BREWER_BLUE_SCALE = [
        [240, 249, 232],
        [204, 235, 197],
        [168, 221, 181],
        [123, 204, 196],
        [67, 162, 202],
        [8, 104, 172]
    ]
    hm_layer = pdk.Layer(
        "HeatmapLayer",
        data=df,
        opacity=0.9,
        get_position=["jd", "wd"],
        aggregation=pdk.types.String("MEAN"),
        color_range=COLOR_BREWER_BLUE_SCALE,
        threshold=1,
        get_weight="cnt",
        pickable=True,
    )
    return hm_layer


def draw_map(user_id:int):
    loc_map = load_loc_map()
    
    st.title(f"用户{user_id}的轨迹图ALL")
    acticity_list = get_acticity_list(user_id)  # 使用用户输入的ID
    arc_layer = plot_route(acticity_list, loc_map)
    hm_layer = plot_htmap(acticity_list, loc_map)
    r = pdk.Deck(
        layers=[arc_layer, hm_layer],
        initial_view_state=view_state,
        map_provider="mapbox",
        map_style=pdk.map_styles.CARTO_LIGHT,
        tooltip={"html": "{day}<br />起点： {b_hour} {b_name} <br />终点：{e_hour} {e_name}",
                    "style": {"backgroundColor": "steelblue", "color": "white"}}
    )
    st.pydeck_chart(r)
    
    cols = st.columns(2)
    with cols[0]:
        st.title(f"TESTTRUTH")
        acticity_list = get_acticity_list(user_id, DataTags.TESTTRUTH)  # 使用用户输入的ID
        arc_layer = plot_route(acticity_list, loc_map)
        hm_layer = plot_htmap(acticity_list, loc_map)
        r = pdk.Deck(
            layers=[arc_layer, hm_layer],
            initial_view_state=view_state,
            map_provider="mapbox",
            map_style=pdk.map_styles.CARTO_LIGHT,
            tooltip={"html": "{day}<br />起点： {b_hour} {b_name} <br />终点：{e_hour} {e_name}",
                        "style": {"backgroundColor": "steelblue", "color": "white"}}
        )
        st.pydeck_chart(r)
    
    with cols[1]:
        st.title(f"TESTGENERATED")
        acticity_list = get_acticity_list(user_id, DataTags.TESTGENERATED)  # 使用用户输入的ID
        arc_layer = plot_route(acticity_list, loc_map)
        hm_layer = plot_htmap(acticity_list, loc_map)
        r = pdk.Deck(
            layers=[arc_layer, hm_layer],
            initial_view_state=view_state,
            map_provider="mapbox",
            map_style=pdk.map_styles.CARTO_LIGHT,
            tooltip={"html": "{day}<br />起点： {b_hour} {b_name} <br />终点：{e_hour} {e_name}",
                        "style": {"backgroundColor": "steelblue", "color": "white"}}
        )
        st.pydeck_chart(r)
        
    
    
    