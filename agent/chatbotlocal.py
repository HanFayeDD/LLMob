from typing import Optional
from langchain.messages import AIMessage
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel, Field

from utils.config import get_env


class ChatBot:
    def __init__(self) -> None:
        """
        初始化 ChatBot，连接到本地 Ollama 服务。

        通过显式传递参数给 ChatOllama，确保配置的准确性。
        本示例默认为本地 127.0.0.1:11434 上的 Ollama 服务，模型为 gemma3:1b。
        可通过环境变量覆盖默认值：
            - OLLAMA_MODEL         默认值：gemma3:1b
            - OLLAMA_BASE_URL      默认值：http://127.0.0.1:11434
            - OLLAMA_TEMPERATURE   默认值：0.7
        """
        self.model_name: str = get_env("OLLAMA_MODEL", default="gemma3:1b")
        self.base_url: str = get_env("OLLAMA_BASE_URL", default="http://127.0.0.1:11434")
        temperature_env: Optional[str] = get_env("OLLAMA_TEMPERATURE", default=None)
        self.temperature: float = float(temperature_env) if temperature_env else 0.7

        print(f"Ollama model: {self.model_name}")
        print(f"Ollama base URL: {self.base_url}")
        print(f"Ollama temperature: {self.temperature}")

        self.model = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
        )

    def ask_one(self, q: str) -> str:
        """
        向模型提问并获取回答。
        在 LangChain 中调用本地 Ollama 模型。
        """
        response = self.model.invoke(q)
        if isinstance(response, AIMessage):
            return response.content
        return str(response)

    def ask_one_with_struct_output(self, q: str, struct: BaseModel) -> BaseModel:
        """
        使用 LangChain 的结构化输出能力，返回 Pydantic 模型实例。
        """
        model_with_structure = self.model.with_structured_output(struct)
        response = model_with_structure.invoke(q)
        print(type(response))
        return response


# 示例：定义一个结构化返回的 Pydantic 模型
class AnswerSchema(BaseModel):
    title: str = Field(..., description="答案摘要标题")
    answer: str = Field(..., description="详细回答内容")


if __name__ == "__main__":
    bot = ChatBot()
    question = "给我一个关于生成式人工智能应用案例的简要介绍。"
    print("普通回答：")
    print(bot.ask_one(question))

    print("\n结构化回答：")
    structured_response = bot.ask_one_with_struct_output(question, AnswerSchema)
    print(structured_response)