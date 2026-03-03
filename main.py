import pandas as pd 
import pydeck as pdk
import streamlit as st
from enum import Enum
from front.map import draw_map, draw_result
from front.defines import *
from front.conversations import draw_conversations
from front import page1, page2

st.set_page_config(layout="wide")

def draw_selected():
    global user_id
    draw_map(user_id)
    draw_conversations(user_id)
    
if __name__ == "__main__":
    
    pg = st.navigation([
        st.Page(draw_selected, title="draw selected", icon="🤗"),
        st.Page(draw_result, title="draw all", icon="🤗"),
        st.Page(page1.draw_page1, title="page1", icon="🤗"),
        st.Page(page2.draw_page2, title="page2", icon="🤗"),
        
    ])
    
    pg.run()