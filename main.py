import pickle
import re
import pandas as pd 
import pydeck as pdk
import streamlit as st

view_state = pdk.ViewState(
        latitude=35.69,
        longitude=139.703,
        bearing=0,
        pitch=20,
        zoom=5,
)

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


def get_acticity_list(id:int):
    def parse_place_time(text:str):
        place, time = text.split(" at ")
        return (place.strip(), time.strip())
    person = pickle.load(open(rf"data\2019\{id}.pkl", "rb"))
    actyls = person[0] + person[1]
    res = []
    for ele in actyls:
        tmp = [None, []]
        p1, p2 = ele.split(": ")
        p1 = p1[-10:]
        tmp[0] = p1
        for pos in p2.split(","):
            tmp[1].append(parse_place_time(pos))
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
    
def plot_bar(acticity_list:list, loc_map:dict):
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
        
    
            
if __name__ == "__main__":
    loc_map = load_loc_map()
    acticity_list = get_acticity_list(6171)
    arc_layer =plot_route(acticity_list, loc_map)
    hm_layer =plot_bar(acticity_list, loc_map)
    r = pdk.Deck(
        layers=[arc_layer, hm_layer],
        initial_view_state=view_state,
        map_provider="mapbox",
        map_style=pdk.map_styles.CARTO_LIGHT,
        tooltip={"html": "{day}<br />起点： {b_hour} {b_name} <br />终点：{e_hour} {e_name}",
                    "style": {"backgroundColor": "steelblue", "color": "white"}}
    )
    st.pydeck_chart(r)
    
    
    
    