# -*- coding: utf-8 -*-
import openai
from engine.llm_configs.base_gpt_api import BaseGPTAPI
from engine.llm_configs.config import CONFIG
import logging

class PoeAPI(BaseGPTAPI):
    """
    与 Poe API 交互的实现。
    继承自 BaseGPTAPI，复用 ask, ask_batch 等通用逻辑。
    """
    def __init__(self):
        self.model = CONFIG.poe_api_model
        
        # --- 修改点 1: 初始化 OpenAI Client ---
        # openai.OpenAI() 内部维护了一个 httpx.Client 实例和连接池。
        # 将其保存在 self.client 中，确保在整个对象生命周期内复用 TCP/TLS 连接。
        self.client = openai.OpenAI(
            api_key=CONFIG.poe_api_key,
            base_url=CONFIG.poe_api_base,
        )
        print(f"{CONFIG.poe_api_model} loaded")

    def __del__(self):
        """
        析构函数：当对象被销毁时，显式关闭 client 内部的 http 连接
        """
        try:
            if hasattr(self, 'client'):
                self.client.close()
        except Exception:
            pass

    def completion(self, messages: list[dict]) -> dict:
        """
        实现对 Poe API 的调用。
        
        Args:
            messages: OpenAI 格式的消息列表 [{"role": "user", "content": "..."}]
        
        Returns:
            符合 OpenAI 响应结构的字典。
        """
        try:
            # --- 修改点 2: 复用 self.client 发起请求 ---
            # 这里调用的是同一个 client 实例，底层的 connection pool 会自动处理 Keep-Alive
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )

            # --- 关键步骤：格式转换 ---
            # openai v1 库返回的是一个 Pydantic 对象。
            # 为了配合 BaseGPTAPI 的通用逻辑（通常期望 dict），我们将其转换为字典。
            # model_dump() 是 Pydantic v2 的标准方法，如果你的环境较老可能需要用 dict() 或手动构造。
            
            # 方法 A: 直接转为字典 (推荐，如果依赖库版本较新)
            # return response.model_dump()

            # 方法 B: 如果上面的报错，可以使用手动构造（类似你的 Ollama 例子），确保万无一失：
            return {
                "choices": [
                    {
                        "message": {
                            "role": response.choices[0].message.role,
                            "content": response.choices[0].message.content
                        }
                    }
                ],
                "usage": {
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }

        except openai.APIConnectionError as e:
            print(f"The server could not be reached: {e.__cause__}")
            return self._generate_error_response(f"Connection Error: {e}")
        except openai.RateLimitError as e:
            print(f"A 429 status code was received; we should back off: {e}")
            return self._generate_error_response(f"Rate Limit Error: {e}")
        except openai.APIStatusError as e:
            print(f"Another non-200-range status code was received: {e.status_code}")
            print(e.response)
            return self._generate_error_response(f"API Error {e.status_code}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return self._generate_error_response(f"Unexpected Error: {e}")

    def _generate_error_response(self, error_msg: str) -> dict:
        """
        辅助方法：生成统一的错误响应格式
        """
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": error_msg
                    }
                }
            ]
        }