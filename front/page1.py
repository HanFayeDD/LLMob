import streamlit as st 
from front import tools
from front.map import plot_route, plot_htmap
from front.map import view_state
from front.tools import loc_map
from front.map import draw_result_map

TAG_CHOICES =  ["2019", "2021", "20192021"]
def draw_page1():
    st.title("原始数据集展示")
    
    ## 使用下拉栏获取用户输入的tag
    tag = st.selectbox("选择数据集", TAG_CHOICES)
    ids = tools.ls_dir_with_tag(tag)
    id = st.selectbox("选择数据文件", ids)
    acticity_list = tools.load_pkl_from_data_dir(tag, id)
    draw_result_map(acticity_list, loc_map)