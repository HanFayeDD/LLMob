from engine.prompt_template.prompt_paths import *
from engine.utilities.process_tools import *
from engine.llm_configs.gpt_structure import *
from engine.utilities.retrieval_helper import *
import logging
import os
import pickle
from collections import defaultdict

# ==============================================================================
# [新增] Critic 模块：用于语义逻辑检查
# ==============================================================================
def semantic_critic(llm, plan_json, date_str):
    """
    使用 LLM 检查轨迹的语义逻辑合理性（软约束）。
    返回: None (通过) 或 错误描述字符串 (不通过)
    """
    critic_prompt_template_path = r"engine\prompt_template\critic.txt"
    
    ipt_data = [date_str, plan_json]
    
    prompt = generate_prompt(ipt_data, critic_prompt_template_path)
    
    # 格式化 plan 为易读字符串
    # todo:是不是可以添加星期几
    
    try:
        # 调用 LLM 进行检查，这里复用传入的 llm 对象
        feedback = execute_prompt(prompt, llm, objective="Critic_Check", temperature=0.0)
        if "Pass" in feedback:
            return None
        else:
            return feedback
    except Exception as e:
        print(f"Critic Error: {e}")
        return None # 如果 Critic 挂了，默认放行，避免卡死

# ==============================================================================
# 主生成逻辑
# ==============================================================================

def mob_gen(person, mode=0, scenario_tag="normal", critic_check=True):
    infer_template = "./engine/prompt_template/one-shot_infer_mot.txt"
    # mode = 0 for learning based retrieval, 1 for evolving based retrieval
    describe_mot_template = "./engine/" + motivation_infer_prompt_paths[mode]
    motivation_ways = ["Following are the motivation that you want to achieve:",
                       "Following are the thing you focus in the last few days:"
                       ]
    mode_name = {0: "llm_l", 1: "llm_e"}
    generation_path = f"./result/{scenario_tag}/generated/{mode_name[mode]}/{str(person.id)}/"
    ground_truth_path = f"./result/{scenario_tag}/ground_truth/{mode_name[mode]}/{str(person.id)}/"
    if os.path.exists(generation_path) is False:
        os.makedirs(generation_path)
    if os.path.exists(ground_truth_path) is False:
        os.makedirs(ground_truth_path)

    results = {}
    reals = {}
    his_routine = person.train_routine_list[-person.top_k_routine:]
    cho = defaultdict(int)
    cho["generated"] = 0 
    cho["from_demo"] = 0
    
    MAX_DAYS = 14
    try_times = 0
    for idx, test_route in enumerate(person.test_routine_list[:]):
        date_ = test_route.split(": ")[0].split(" ")[-1]
        week_day = date_to_weekday(date_)
        # get motivation
        consecutive_past_days = check_consecutive_dates(his_routine, date_)
        if mode == 0:
            # learning based retrieved
            retrieve_route = person.retriever.retrieve(date_)
            demo = retrieve_route[0]
        else:
            # evolving based retrieved
            demo = his_routine[-1]

        hint = ""  # add condition prompt for conditional generation, i.e., pandemic condition
        curr_input = [person.attribute, "Go to " + demo.split(": ")[-1], consecutive_past_days, hint]
        ## 动机推断prompt
        prompt_mot = generate_prompt(curr_input, describe_mot_template)
        ## 根据demo，结合地点类型信息，获取相似地点推荐
        area = retrieve_loc(person, demo)
        motivation = execute_prompt(prompt_mot, person.llm, objective=f"Think about motivation")
        motivation = first2second(motivation)
        his_routine = his_routine[1:] + [test_route] ## 会更新his_routine，从而导致跟新demo
        weekday = find_detail_weekday(date_)
        hint = ""
        
        ## 动机驱动生成轨迹
        if motivation is not None:
            curr_input = [person.attribute, motivation, date_, ',  '.join(area), weekday, demo,
                          motivation_ways[mode],
                          hint, week_day]
        
        # 生成基础 Prompt
        base_prompt = generate_prompt(curr_input, infer_template)
        
        logging.info(f"base_prompt\n{base_prompt}")
        
        max_trial = 5 # 适当减少尝试次数，因为现在有了修正机制，效率应该更高
        trial = 0
        running = True
        
        # [新增] 用于存储反思反馈的变量
        feedback_instruction = "" 
        
        while running and trial < max_trial:
            # [修改] 构造当前轮次的 Prompt：基础 Prompt + (可能的) 反思修正指令
            current_prompt = base_prompt + feedback_instruction
            
            contents = execute_prompt(current_prompt, person.llm,
                                      objective=f"one_shot_infer_response_{len(results) + 1}/{len(person.test_routine_list)}_{trial}/{max_trial}")
            
            try_times += 1
            
            if len(feedback_instruction) > 0:
                logging.info(f"current_prompt\n{current_prompt}")    
        
            
            error_msgs = [] # 收集本轮的所有错误
            
            try:
                if trial == 0:
                    print(f"Initial prompt sent.")
                
                contents = filter_json_part(contents)
                logging.info(f"contents after\n{contents}")
                res = json.loads(contents)
                
                # ==========================================
                # [修改] 增强校验与 Critic 介入 (Reflexion Loop)
                # ==========================================
                
                # 1. 硬约束检查 (Hard Constraints)
                if (res_place := valid_place_return(res["plan"], person.area_freq)):
                    error_msgs.append(f"Error: Some locations({res_place}) in the plan are invalid or do not match the area frequency.")

                if (res_time := valid_time_return(res["plan"])):
                    error_msgs.append(f"Error: Time sequence{res_time} is invalid (e.g., end time is earlier than start time, or format is wrong).")

                # if  valid_place(res["plan"], person.area_freq):
                #     error_msgs.append("Error: Some locations in the plan are invalid or do not match the area frequency.")
                
                # if not valid_time(res["plan"]):
                #     error_msgs.append("Error: Time sequence is invalid (e.g., end time is earlier than start time, or format is wrong).")
                
                # 2. 软约束检查 (Soft Constraints / Semantic Critic)
                # 只有当硬约束通过时，才进行昂贵的语义检查，节省 Token
                if not error_msgs and critic_check:
                    semantic_error = semantic_critic(person.llm, res["plan"], date_)
                    if semantic_error:
                        error_msgs.append(f"Logic Error: {semantic_error}")

                # 3. 判定结果
                if not error_msgs:
                    # 所有检查通过
                    valid_generation_v2(person, f"Activities at {date_}: " + ', '.join(res["plan"]))
                    running = False # 退出循环
                else:
                    # 发现错误，触发异常以进入 except 块进行反思处理
                    raise ValueError(" | ".join(error_msgs))

            except Exception as e:
                logging.info(f"Trial {trial}/{max_trial} Failed. Reason: {str(e)}")
                
                # [新增] 生成反思指令 (Reflexion)
                # 告诉 LLM 它上次生成了什么，以及为什么错了
                if critic_check:
                    feedback_instruction = f"\n\n[System Feedback - Self-Correction Required]\n" \
                                       f"Your previous generated plan was: {contents}\n" \
                                       f"It contained the following errors: {str(e)}\n" \
                                       f"Please re-generate the plan. Fix these errors specifically. Ensure valid JSON format."
                
                    logging.info(f"feedback:{feedback_instruction}")
                trial += 1
                continue
        
        # 循环结束后的处理
        if trial >= max_trial:
            # 如果多次修正仍失败，回退到 Demo
            res = {"plan": demo.split(": ")[-1]}
            logging.warning(f"Max retries reached for {date_}. Fallback to demo.")

        logging.info(contents)
        print("Motivation: ", motivation)
        print("Real: ", test_route)
        reals[date_] = test_route
        logging.info(f"\ndemo:{demo}")
        
        if trial < max_trial:
            results[date_] = f"Activities at {date_}: " + ', '.join(res["plan"])
            logging.info(f"result is generated")
            cho["generated"] += 1
        else:
            results[date_] = f"Activities at {date_}: " + demo.split(": ")[-1]
            logging.info(f"result is from demo")
            cho["from_demo"] += 1
            
        if mode == 0:
            person.retriever.nodes.append(reals[date_])
            
        logging.info(f"Generated{date_}: {results[date_]}")
        # break
        if idx == MAX_DAYS-1:
            break
        
    with open(r"./cho.txt", "a") as f:
        f.write(f"{person.name}-{MAX_DAYS}-{try_times}/{max_trial}-{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\ngenerated:{cho['generated']}\nfrom_demo:{cho['from_demo']}\n")
            
    logging.info(f"{cho}")
    # dump pkl
    with open(generation_path + "results.pkl", "wb") as f:
        pickle.dump(results, f)
    with open(ground_truth_path + "results.pkl", "wb") as f:
        pickle.dump(reals, f)
    print(generation_path)
    print(ground_truth_path)