import streamlit as st
from front.defines import *
from front import page1, page2, page3

st.set_page_config(layout="wide")

    
if __name__ == "__main__":
    
    pg = st.navigation([
        st.Page(page1.draw_page1, title="原始数据集展示", icon="🤗"),
        st.Page(page2.draw_page2, title="生成轨迹展示", icon="🤗"),
        st.Page(page3.draw_page3, title="个体追踪", icon="🤗")
        
    ])
    
    pg.run()