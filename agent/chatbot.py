# 建议的完整代码
from langchain_openai import ChatOpenAI  # <-- 使用新的导入方式
from langchain.messages import AIMessage
from utils.config import get_env
from pydantic import BaseModel, Field
import os

class ChatBot():
    def __init__(self):
        """
        初始化 ChatBot。
        通过显式传递参数给 ChatOpenAI，确保配置的准确性。
        这是目前推荐的最佳实践。
        """
   
        self.api_key = get_env("OPENAI_API_KEY")
        self.api_model = get_env("OPENAI_API_MODEL")
        self.api_url = get_env("OPENAI_API_BASE")

        print(f"API Key loaded: ...{self.api_key[-4:]}") # 打印最后四位以确认
        print(f"API Model: {self.api_model}")
        print(f"API Base URL: {self.api_url}")

        self.model = ChatOpenAI(
            model=self.api_model,
            openai_api_key=self.api_key,
            openai_api_base=self.api_url,
            # temperature=0.7 # 也可以在这里设置其他参数
        )

    def ask_one(self, q: str) -> str:
        """
        向模型提问并获取回答。
        LangChain v0.1.0 之后，推荐使用 .invoke() 方法。
        """
        # 在新版 LangChain 中，输入通常被包装成消息对象
        # 但为了简化，直接传入字符串通常也能工作
        return self.model.invoke(q).content
    
    def ask_one_with_struct_output(self, q:str, struct:BaseModel)->BaseModel:
        model_with_structure = self.model.with_structured_output(struct)
        response = model_with_structure.invoke(q)
        print(type(response))
        return response
    
    
    

# --- 使用示例 ---
# 假设您的 get_env 和 .env 文件设置正确
# if __name__ == '__main__':
#     try:
#         bot = ChatBot()
#         question = "你好，请介绍一下你自己。"
#         response = bot.ask_one(question)
#         print("\n模型回答:")
#         print(response.content)
#     except Exception as e:
#         print(f"\n发生错误: {e}")