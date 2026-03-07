import streamlit as st
import os 
from front import tools
from front.tools import loc_map
from front.map import draw_result_map, time_filter, draw_result_radar


def draw_page2():
    st.title("生成轨迹展示")
    
    cols = st.columns([1, 1, 4])
    with cols[0]:
        tag = st.selectbox("选择数据集", ["normal", "normal_abnormal"])

    fs = os.listdir(f"./chathistory/{tag}")
       
    folders = [f for f in fs if os.path.isdir(os.path.join(f"./chathistory/{tag}", f))]
    folders.remove("identify_history")
    
    
    with cols[1]:
        fold = st.selectbox("选择模型与方式", folders)
    
    actyls_r, actyls_g = tools.load_pkl_from_selected_folder(tag, fold)

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Real Trajectories")
        draw_result_map(actyls_r, loc_map)
        
    
    with cols[1]:
        st.subheader("Generated Trajectories")
        draw_result_map(actyls_g, loc_map)
    
    st.subheader("Radar Chart")  
    draw_result_radar(actyls_g, actyls_r)
        
    st.subheader("Real Trajectories")
    cols = st.columns(3)
    with cols[0]:
        st.write("06:00-11:00")
        actyls_r_6_11 = time_filter(actyls_r, 6, 11)
        draw_result_map(actyls_r_6_11, loc_map)
    with cols[1]:
        st.write("11:00-16:00")
        actyls_r_11_16 = time_filter(actyls_r, 11, 16)
        draw_result_map(actyls_r_11_16, loc_map)
    with cols[2]:
        st.write("16:00-24:00")
        actyls_r_16_24 = time_filter(actyls_r, 16, 24)
        draw_result_map(actyls_r_16_24, loc_map)
        
    st.subheader("Generated Trajectories")
    cols = st.columns(3)
    with cols[0]:
        st.write("06:00-11:00")
        actyls_g_6_11 = time_filter(actyls_g, 6, 11)
        draw_result_map(actyls_g_6_11, loc_map)
    with cols[1]:
        st.write("11:00-16:00")
        actyls_g_11_16 = time_filter(actyls_g, 11, 16)
        draw_result_map(actyls_g_11_16, loc_map)    
    with cols[2]:
        st.write("16:00-24:00")
        actyls_g_16_24 = time_filter(actyls_g, 16, 24)
        draw_result_map(actyls_g_16_24, loc_map)
        
        
    st.subheader("Radar Chart By Time Slot")
    cols = st.columns(3)
    with cols[0]:
        st.write("06:00-11:00")
        draw_result_radar(actyls_g_6_11, actyls_r_6_11)
    
    with cols[1]:
        st.write("11:00-16:00")
        draw_result_radar(actyls_g_11_16, actyls_r_11_16)
            
    with cols[2]:
        st.write("16:00-24:00")
        draw_result_radar(actyls_g_16_24, actyls_r_16_24)
        