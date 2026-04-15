import streamlit as st
import pickle
import os
import random
import engine.llm_configs.gpt_structure as gpt_structure
from engine.llm_configs.poe_api import PoeAPI
from engine.trajectory_generate import mob_gen
from engine.persona_identify import identify
from engine.agent import Person
from front.defines import available2019, available1921
from front.tools import (
    get_person_choices, get_max_days, save_results_to_frontdata,
    load_results_from_frontdata, has_results_in_frontdata, load_persona_mid_result,
    parser_actyls, loc_map
)
from front.map import draw_result_map, draw_result_radar, time_filter

# 常量定义
DATASET_CHOICES = ["2019", "20192021"]
MODE_CHOICES = {"基于检索": 0, "基于演化": 1}
MODEL_CHOICES = ["gemini-2.5-flash-lite", "gpt-3.5-turbo"]
DEFAULT_CRITIC = True

# scenario_tag映射
SCENARIO_TAG = {
    '2019': 'normal',
    '20192021': 'normal_abnormal'
}

def draw_page5():
    st.title("轨迹生成工具")
    
    # === 第一部分：参数选择 ===
    st.subheader("参数配置")
    cols = st.columns([1, 1, 1, 3])
    with cols[0]:
        dataset = st.selectbox("选择数据集", DATASET_CHOICES, key="dataset_select")
    with cols[1]:
        mode_display = st.selectbox("选择模式", list(MODE_CHOICES.keys()), key="mode_select")
        mode = MODE_CHOICES[mode_display]
    with cols[2]:
        model = st.selectbox("选择模型", MODEL_CHOICES, key="model_select")
    # with cols[3]:
    #     critic_check = st.toggle("启用critic", value=DEFAULT_CRITIC, key="critic_toggle")
    colss = st.columns([1, 5])
    with colss[0]:
        critic_check = st.toggle("启用critic", value=DEFAULT_CRITIC, key="critic_toggle")
    # === 第二部分：个体选择 ===
    st.subheader("个体选择")
    person_choices = get_person_choices(dataset)
    person_id = st.selectbox("选择个体ID", person_choices, key="person_select")
    
    # === 第三部分：天数选择 ===
    st.subheader("生成天数配置")
    max_days = get_max_days(dataset, person_id)
    default_days = min(14, max_days)
    g_days = st.slider(
        "生成天数", 
        min_value=1, 
        max_value=max_days, 
        value=default_days,
        help=f"可选择范围: 1 到 {max_days} 天"
    )
    
    # 显示当前配置信息
    st.info(f"当前配置: 数据集={dataset}, 个体ID={person_id}, 模式={mode_display}, 模型={model}, 天数={g_days}, critic={'开启' if critic_check else '关闭'}")
    
    # === 第四部分：执行按钮 ===
    st.subheader("轨迹生成")
    
    # 检查是否已有结果
    has_result = has_results_in_frontdata(person_id, dataset, mode, model)
    
    col_btn = st.columns([1, 4])
    with col_btn[0]:
        generate_btn = st.button("开始生成", type="primary", disabled=False)
    
    # 如果已有结果，显示加载按钮
    if has_result:
        with col_btn[0]:
            load_btn = st.button("加载已有结果", type="secondary")
        if load_btn:
            st.session_state['load_result'] = True
    
    # === 执行生成逻辑 ===
    if generate_btn:
        # 创建进度条和状态文本
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("正在初始化...")
        
        try:
            # 初始化日志
            from utils.logger import init_log
            init_log()
            
            # 设置随机种子
            random.seed(123)
            
            # 加载persona结果
            persona_dict = load_persona_mid_result(dataset)
            
            # 设置日志文件
            log_filename = f"chathistory/{person_id}.txt"
            gpt_structure.set_current_log_file(log_filename)
            
            # 加载pkl数据
            folder = f"./data/{dataset}/"
            with open(folder + str(person_id) + ".pkl", "rb") as f:
                att = pickle.load(f)
            
            status_text.text("正在创建Person对象...")
            
            # 创建Person对象
            P = Person(name=person_id, model=PoeAPI(), person_id=person_id)
            if model != "":
                P.llm.set_model(model)
            
            # 加载属性
            P.train_routine_list, P.test_routine_list, P.attribute, P.cat, P.domain_knowledge, P.neg_routines, P.activity_area, P.area_freq, P.loc_cat = \
                att[0], att[1], att[2], att[4], att[5], att[6], att[7], att[8], att[11]
            
            # 加载persona
            if person_id in persona_dict:
                P.attribute = persona_dict[person_id]
            else:
                status_text.text("正在识别persona...")
                P = identify(P)
            
            # 初始化retriever (仅mode=0需要)
            if mode == 0:
                status_text.text("正在初始化检索器...")
                P.init_retriever(model_type="DeepModel")
            
            status_text.text("开始生成轨迹...")
            
            # 定义进度回调函数
            def progress_callback(current_idx, total_days, message):
                progress = (current_idx + 1) / total_days
                progress_bar.progress(progress)
                status_text.text(message)
            
            # 执行轨迹生成
            scenario_tag = SCENARIO_TAG[dataset]
            mob_gen(P, mode=mode, scenario_tag=scenario_tag, critic_check=critic_check,
                    g_days=g_days, progress_callback=progress_callback)
            
            # 从result目录读取生成结果
            mode_name = {0: "llm_l", 1: "llm_e", 2: "llm_nm"}
            generation_path = f"./result/{scenario_tag}/generated/{mode_name[mode]}/{person_id}/results.pkl"
            ground_truth_path = f"./result/{scenario_tag}/ground_truth/{mode_name[mode]}/{person_id}/results.pkl"
            
            with open(generation_path, "rb") as f:
                gen_traj = pickle.load(f)
            with open(ground_truth_path, "rb") as f:
                real_traj = pickle.load(f)
            
            # 保存到frontdata
            save_dir = save_results_to_frontdata(real_traj, gen_traj, person_id, dataset, mode, model)
            
            progress_bar.progress(1.0)
            status_text.text("生成完成！")
            
            st.success(f"轨迹生成完成！结果已保存到: {save_dir}")
            
            # 设置session state以便显示结果
            st.session_state['generation_complete'] = True
            st.session_state['current_person_id'] = person_id
            st.session_state['current_dataset'] = dataset
            st.session_state['current_mode'] = mode
            st.session_state['current_model'] = model
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"生成过程中出现错误: {str(e)}")
            st.exception(e)
    
    # === 第五部分：结果展示 ===
    # 检查是否需要显示结果
    show_result = False
    if 'generation_complete' in st.session_state and st.session_state['generation_complete']:
        show_result = True
        person_id = st.session_state['current_person_id']
        dataset = st.session_state['current_dataset']
        mode = st.session_state['current_mode']
        model = st.session_state['current_model']
    elif 'load_result' in st.session_state and st.session_state['load_result']:
        show_result = True
    
    if show_result:
        st.subheader("结果可视化")
        
        # 加载结果
        real_actyls, gen_actyls = load_results_from_frontdata(person_id, dataset, mode, model)
        
        if real_actyls is None or gen_actyls is None:
            st.warning("无法加载结果数据")
        else:
            # 显示轨迹地图对比
            st.subheader("轨迹地图对比")
            cols = st.columns(2)
            with cols[0]:
                st.write("**真实轨迹**")
                draw_result_map(real_actyls, loc_map)
            
            with cols[1]:
                st.write("**生成轨迹**")
                draw_result_map(gen_actyls, loc_map)
            
            # 显示雷达图对比
            st.subheader("频率对比雷达图")
            draw_result_radar(gen_actyls, real_actyls)
            
            # 按时间段展示
            st.subheader("分时段轨迹展示")
            
            # 真实轨迹分时段
            st.write("**真实轨迹**")
            cols = st.columns(3)
            with cols[0]:
                st.write("06:00-11:00")
                real_6_11 = time_filter(real_actyls, 6, 11)
                draw_result_map(real_6_11, loc_map)
            with cols[1]:
                st.write("11:00-16:00")
                real_11_16 = time_filter(real_actyls, 11, 16)
                draw_result_map(real_11_16, loc_map)
            with cols[2]:
                st.write("16:00-24:00")
                real_16_24 = time_filter(real_actyls, 16, 24)
                draw_result_map(real_16_24, loc_map)
            
            # 生成轨迹分时段
            st.write("**生成轨迹**")
            cols = st.columns(3)
            with cols[0]:
                st.write("06:00-11:00")
                gen_6_11 = time_filter(gen_actyls, 6, 11)
                draw_result_map(gen_6_11, loc_map)
            with cols[1]:
                st.write("11:00-16:00")
                gen_11_16 = time_filter(gen_actyls, 11, 16)
                draw_result_map(gen_11_16, loc_map)
            with cols[2]:
                st.write("16:00-24:00")
                gen_16_24 = time_filter(gen_actyls, 16, 24)
                draw_result_map(gen_16_24, loc_map)
            
            # 分时段雷达图
            st.subheader("分时段频率对比")
            cols = st.columns(3)
            with cols[0]:
                st.write("06:00-11:00")
                draw_result_radar(gen_6_11, real_6_11)
            with cols[1]:
                st.write("11:00-16:00")
                draw_result_radar(gen_11_16, real_11_16)
            with cols[2]:
                st.write("16:00-24:00")
                draw_result_radar(gen_16_24, real_16_24)