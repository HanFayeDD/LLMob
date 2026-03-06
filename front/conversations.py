import os
import streamlit as st

BASE_DIR = "chathistory"

def get_abso_path(user_id: int):
    # 确保目录存在，防止报错
    path = os.path.join(os.getcwd(), BASE_DIR)
    if not os.path.exists(path):
        os.makedirs(path)
    # 假设每个用户的记录是一个单独的文件，例如 "1001.txt"
    # 如果是一个文件夹下的多个文件，逻辑需要微调
    return os.path.join(path, f"{user_id}.txt")

def parse_conversation_log(file_content):
    """
    解析特定格式的日志文件内容。
    返回一个包含字典的列表，每个字典代表一条记录。
    """
    conversations = []
    # 使用分隔符切分记录
    raw_records = file_content.split("==================== START RECORD ====================")

    for raw in raw_records:
        if not raw.strip():
            continue
            
        # 初始化字段
        objective = ""
        prompt = ""
        result = ""

        # 定位标签位置
        idx_obj = raw.find("[Objective]:")
        idx_prompt = raw.find("[Prompt]:")
        idx_result = raw.find("[Result]:")
        idx_end = raw.find("==================== END RECORD ====================")

        # 提取 Objective (如果有)
        if idx_obj != -1:
            end = idx_prompt if idx_prompt != -1 else len(raw)
            objective = raw[idx_obj + len("[Objective]:"):end].strip()

        # 提取 Prompt
        if idx_prompt != -1:
            end = idx_result if idx_result != -1 else len(raw)
            prompt = raw[idx_prompt + len("[Prompt]:"):end].strip()

        # 提取 Result
        if idx_result != -1:
            end = idx_end if idx_end != -1 else len(raw)
            result = raw[idx_result + len("[Result]:"):end].strip()

        if prompt or result:
            conversations.append({
                "objective": objective,
                "prompt": prompt,
                "result": result
            })
            
    return conversations

def draw_conversations(content):    
    # 解析内容
    records = parse_conversation_log(content)
    
    if not records:
        st.warning("File found but no valid records parsed.")
        return

    # 遍历并展示记录
    for record in records:
        # 1. 展示 Objective (通常是系统指令或元数据，可以用折叠框隐藏)
        if record['objective']:
            with st.expander(record['objective']):
                # 2. 展示 Prompt (用户输入)
                # 使用 st.chat_message 模拟对话气泡
                with st.chat_message("user"):
                    st.write("**Prompt:**")
                    st.markdown(record['prompt'])

                # 3. 展示 Result (LLM 输出)
                with st.chat_message("assistant"):
                    st.write("**Result:**")
                    st.markdown(record['result'])
        
