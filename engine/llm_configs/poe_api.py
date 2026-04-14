# -*- coding: utf-8 -*-
import openai
from engine.llm_configs.base_gpt_api import BaseGPTAPI
from engine.llm_configs.config import CONFIG
import logging
from engine.utilities.api_limiter import SimpleRateLimiter


# ==========================================
# 修改后的 PoeAPI 类
# ==========================================
class PoeAPI(BaseGPTAPI):
    """
    与 Poe API 交互的实现。
    继承自 BaseGPTAPI，复用 ask, ask_batch 等通用逻辑。
    """
    def __init__(self):
        self.model = CONFIG.poe_api_model
        
        # 初始化 OpenAI Client
        self.client = openai.OpenAI(
            api_key=CONFIG.poe_api_key,
            base_url=CONFIG.poe_api_base,
        )
        print(f"{CONFIG.poe_api_model} loaded")
        
    def set_model(self, model_name: str):
        """
        允许动态切换模型。
        """
        self.model = model_name
        print(f"PoeAPI model switched to {self.model}")

    # --- 修改点: 应用装饰器 ---
    # 示例：限制每 60 秒最多请求 10 次
    @SimpleRateLimiter(max_calls=10, period=10)
    def completion(self, messages: list[dict]) -> dict:
        """
        实现对 Poe API 的调用。
        已增加速率限制（单线程版）。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                timeout=30
            )

            # 格式转换
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
            return self._generate_error_response(f"API Error {e.status_code}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return self._generate_error_response(f"Unexpected Error: {e}")

    def _generate_error_response(self, error_msg: str) -> dict:
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