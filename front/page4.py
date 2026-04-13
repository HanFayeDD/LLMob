import streamlit as st
import os 
from front import tools
from front.tools import loc_map
from front.map import draw_result_map, time_filter, draw_result_radar, draw_result_radar_combination



def draw_page4():
    st.title("epidemic")
    
    tag = "normal_abnormal"
    fold = "gemini_PMC_no_pendemic_prompt"
    gemini_r_np, gemini_g_np = tools.load_pkl_from_selected_folder(tag, fold)
    
    fold = "gemini_PMC_pendemic_prompt"
    gemini_r_p, gemini_g_p = tools.load_pkl_from_selected_folder(tag, fold)
    
    fold = "gpt3.5turbo_PMC_pendemic_prompt"
    gpt3_r_p, gpt3_g_p = tools.load_pkl_from_selected_folder(tag, fold)
    
    _, gpt3_g_np = tools.load_pkl_from_selected_folder("normal_abnormal", "gpt3.5turbo_PMC_no_pendemic_prompt")
    
    draw_result_radar_combination(      
        [gemini_g_np, gemini_g_p, gemini_r_np],
        ["GPT3.5 No Prompt", "GPT3.5 Prompt", "Ground Truth(2021)"]
    )