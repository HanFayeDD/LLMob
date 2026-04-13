import streamlit as st
from front.defines import *
from front import page1, page2, page3, page4, page5

st.set_page_config(layout="wide")

    
if __name__ == "__main__":
    
    pg = st.navigation([
        st.Page(page1.draw_page1, title="原始数据集展示", icon="🤗"),
        st.Page(page2.draw_page2, title="生成轨迹展示", icon="🤗"),
        st.Page(page3.draw_page3, title="个体追踪", icon="🤗"),
        st.Page(page4.draw_page4, title="疫情分析", icon="🤗"),
        st.Page(page5.draw_page5, title="轨迹生成工具", icon="🔧"),
    ])
    
    pg.run()