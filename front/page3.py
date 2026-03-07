import streamlit as st
import os 
from front import tools
from front.map import draw_result_map, time_filter, draw_result_radar
from front.tools import loc_map


def draw_page3():
    global persona_map
    st.title("个体追踪")
    
    cols = st.columns([1, 1, 1, 3])
    with cols[0]:
        tag = st.selectbox("选择数据集", ["normal", "normal_abnormal"])

    fs = os.listdir(f"./chathistory/{tag}")  
    folders = [f for f in fs if os.path.isdir(os.path.join(f"./chathistory/{tag}", f))]
    with cols[1]:
        fold = st.selectbox("选择模型与方式", folders)
        
    idls = tools.ls_id_from_history_dir(tag, fold)
    with cols[2]:
        id = st.selectbox("选择ID", idls)

    st.subheader("活动模式识别结果")
    with st.expander("活动模式识别", expanded=False):
        st.text(persona_map[tag + id])
    
    st.subheader("活动轨迹")
    actyls_r, actyls_g = tools.load_pkl_from_id(tag, fold, id)
    
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Real Trajectories")
        draw_result_map(actyls_r, loc_map)
        
    
    with cols[1]:
        st.subheader("Generated Trajectories")
        draw_result_map(actyls_g, loc_map)
        
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
    
    st.subheader("对话历史")
    with st.expander("活动模式识别", expanded=False):
        tools.show_identify_history(tag, fold, id)
    
    with st.expander("活动轨迹生成", expanded=False):
        tools.show_traj_gen_history(tag, fold, id)
        
    with st.expander("LLM轨迹评分", expanded=False):
        tools.show_llm_judge_history(tag, fold, id)
    
content = None 
persona_map = dict()
with open(r"persona_result\results2019.txt", "r") as f:
    content = f.read()
lines = [line for line in content.split("\n") if line.strip()]
for i in range(0, len(lines), 2):
    persona_map["normal" + lines[i].split("-")[1].strip()] = lines[i+1]
    
    
with open(r"persona_result\results20192021.txt", "r") as f:
    content = f.read()
lines = [line for line in content.split("\n") if line.strip()]
for i in range(0, len(lines), 2):
    persona_map["normal_abnormal" + lines[i].split("-")[1].strip()] = lines[i+1]
print("loading persona")