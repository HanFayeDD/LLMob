from engine.prompt_template.prompt_paths import *
from engine.utilities.process_tools import *
from engine.llm_configs.gpt_structure import *
from engine.utilities.retrieval_helper import *
import logging
import os
import pickle


def mob_gen(person, mode=0, scenario_tag="normal"):
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
    for idx, test_route in enumerate(person.test_routine_list[:]):
        date_ = test_route.split(": ")[0].split(" ")[-1]
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
        prompt = generate_prompt(curr_input, describe_mot_template)
        ## 根据demo，结合地点类型信息，获取相似地点推荐
        area = retrieve_loc(person, demo)
        motivation = execute_prompt(prompt, person.llm, objective=f"Think about motivation")
        motivation = first2second(motivation)
        his_routine = his_routine[1:] + [test_route]
        weekday = find_detail_weekday(date_)
        hint = ""
        ## 动机驱动生成轨迹
        if motivation is not None:
            curr_input = [person.attribute, motivation, date_, ',  '.join(area), weekday, demo,
                          motivation_ways[mode],
                          hint]
        prompt = generate_prompt(curr_input, infer_template)
        max_trial = 10
        trial = 0
        running = True
        while running and trial < max_trial:
            contents = execute_prompt(prompt, person.llm,
                                      objective=f"one_shot_infer_response_{len(results) + 1}/{len(person.test_routine_list)}_{trial}")
            try:
                if trial == 0:
                    print(f"prompt\n{prompt}")
                print(f"contents before\n{contents}")
                contents = filter_json_part(contents)
                print(f"contents after\n{contents}")
                res = json.loads(contents)
                ## 一些清洗与校验
                # valid_generation(person, f"Activities at {date_}: " + ', '.join(res["plan"]))
                valid_generation_v2(person, f"Activities at {date_}: " + ', '.join(res["plan"]))
                running = False
            except Exception as e:
                print("Error: " + str(e))
                trial += 1
                continue
            break
        if trial >= max_trial:
            res = {"plan": demo.split(": ")[-1]}
        logging.info(contents)
        print("Motivation: ", motivation)
        print("Real: ", test_route)
        reals[date_] = test_route
        results[date_] = f"Activities at {date_}: " + ', '.join(res["plan"])
        if mode == 0:
            person.retriever.nodes.append(reals[date_])
        ## 只产生一天的
        if idx == 1:
            break
    # dump pkl
    with open(generation_path + "results.pkl", "wb") as f:
        pickle.dump(results, f)
    with open(ground_truth_path + "results.pkl", "wb") as f:
        pickle.dump(reals, f)
    print(generation_path)
    print(ground_truth_path)
