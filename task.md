现在我想基于偏好率实现llmjudge思路，主要代码修改在evaluate.py中。
主要实现思路如下：
根据用户的个性化特征、真实轨迹、输出轨迹，计算模型的偏好率。
将三个参数传给llm，问他哪个轨迹更符合个性化特征并给出原因。
最后偏好的真实轨迹的数目/总轨迹数目即为偏好率。

给你的提示如下：
（1）可以参考evaluate.py中llm_as_judge_one_day的实现，
给出对应的prompt_template提示词模板
（2）用户的个性化特征可以参考generate.py中的load_persona_mid_result方法。注意：需要根据dataset进行分文件读取。

不要修改与任务无关的一切代码。
先指定计划，再进行代码撰写。遇到问题问我。