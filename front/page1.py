import streamlit as st 
from front import tools
from front.tools import loc_map
from front.map import draw_result_map

TAG_CHOICES =  ["2019", "2021", "20192021"]
def draw_page1():
    st.title("原始数据集展示")
    
    ## 使用下拉栏获取用户输入的tag
    cols = st.columns([1, 1, 4])
    with cols[0]:
        tag = st.selectbox("选择数据集", TAG_CHOICES)
    ids = tools.ls_dir_with_tag(tag)
    with cols[1]:
        id = st.selectbox("选择数据文件", ids)
    acticity_list = tools.load_pkl_from_data_dir(tag, id)
    draw_result_map(acticity_list, loc_map)