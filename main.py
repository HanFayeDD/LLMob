import pandas as pd 
import pydeck as pdk
import streamlit as st
from enum import Enum
from front.map import draw_map, draw_result
from front.defines import *
from front.conversations import draw_conversations


st.set_page_config(layout="wide")

def draw_selected():
    global user_id
    draw_map(user_id)
    draw_conversations(user_id)
    
if __name__ == "__main__":
    # 添加侧边栏
    with st.sidebar:
        st.title("用户ID输入")
        user_id = st.number_input(
            "请输入用户ID", 
            min_value=1, 
            value=2575, 
            step=1
        )

    with st.sidebar:
        st.write(f"2019:\n{available2019}\n2021\n{available2021}\n20192021\n{available1921} ")
    
    pg = st.navigation([
        st.Page(draw_selected, title="draw selected", icon="🤗"),
        st.Page(draw_result, title="draw all", icon="🤗")
    ])
    
    pg.run()