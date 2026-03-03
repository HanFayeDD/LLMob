import streamlit as st 
from front import tools
from front.map import plot_route, plot_htmap
import pydeck as pdk

view_state = pdk.ViewState(
        latitude=35.69,
        longitude=139.703,
        bearing=0,
        pitch=20,
        zoom=5,
)

TAG_CHOICES =  ["2019", "2021", "20192021"]
def draw_page1():
    st.title("原始数据集展示")
    
    ## 使用下拉栏获取用户输入的tag
    tag = st.selectbox("选择数据集", TAG_CHOICES)
    ids = tools.ls_dir_with_tag(tag)
    id = st.selectbox("选择数据文件", ids)
    acticity_list = tools.load_pkl_from_data_dir(tag, id)
    
    global loc_map
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
loc_map = tools.load_loc_map()
    
    