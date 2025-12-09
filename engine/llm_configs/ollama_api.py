# -*- coding: utf-8 -*-
import requests
import json
from engine.llm_configs.base_gpt_api import BaseGPTAPI
from engine.llm_configs.config import CONFIG

class OllamaAPI(BaseGPTAPI):
    """
    与本地Ollama模型交互的实现。
    它继承了BaseGPTAPI，因此拥有ask, ask_batch等方法。
    我们只需要实现核心的 completion 方法。
    """
    def __init__(self):
        self.model = CONFIG.ollama_api_model
        self.base_url = CONFIG.ollama_api_base
        
        # --- 修改点 1: 初始化 Session 对象 ---
        # 使用 Session 对象可以实现连接池和连接复用（Keep-Alive）
        # 这对于频繁调用本地 API 能够显著减少 TCP 握手开销
        self.session = requests.Session()
        
        # 可以选择性地设置一些 headers，这样后续所有通过 session 发出的请求都会带上
        self.session.headers.update({"Content-Type": "application/json"})

        # Ollama通常在本地运行，不需要复杂的速率限制，但如果需要也可以实现
        # RateLimiter.__init__(self, rpm=...) 
        print(f"ollama {CONFIG.ollama_api_model} laoded")

    def __del__(self):
        """
        析构函数：当对象被销毁时，关闭 session 连接
        """
        try:
            if hasattr(self, 'session'):
                self.session.close()
        except Exception:
            pass
        
    def completion(self, messages: list[dict]) -> dict:
        """
        实现对Ollama /api/chat端点的调用。
        
        Args:
            messages: OpenAI格式的消息列表。
        
        Returns:
            一个模拟OpenAI响应格式的字典，以便上层代码可以统一处理。
        """
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False  # 为了与现有结构兼容，我们先使用非流式响应
        }
        
        try:
            # --- 修改点 2: 使用 self.session 发送 POST 请求 ---
            # 这里的 headers 已经在 init 中设置过 Content-Type，但显式传递也不会出错
            # json=payload 参数会自动将字典转为 json 字符串并设置 Content-Type
            response = self.session.post(url, json=payload)
            
            response.raise_for_status()  # 如果HTTP状态码是4xx或5xx，则抛出异常
            
            ollama_response = response.json()
            
            # --- 关键步骤：将Ollama的响应格式转换为OpenAI的格式 ---
            # Ollama响应: {'model': 'llama3', 'created_at': '...', 'message': {'role': 'assistant', 'content': '...'}, ...}
            # OpenAI响应: {'choices': [{'message': {'role': 'assistant', 'content': '...'}}], ...}
            
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": ollama_response.get("message", {}).get("content", "")
                        }
                    }
                ],
                # 如果需要，可以模拟其他字段
                "usage": {
                    "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
                    "completion_tokens": ollama_response.get("eval_count", 0),
                    "total_tokens": ollama_response.get("prompt_eval_count", 0) + ollama_response.get("eval_count", 0)
                }
            }

        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            # 返回一个空的、但结构一致的错误响应
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"Error: Failed to connect to Ollama. {e}"
                        }
                    }
                ]
            }

    # 注意：ask, ask_batch, ask_code, get_choice_text等方法都继承自BaseGPTAPI，
    # 它们会自动调用我们上面实现的 completion 方法，所以我们无需重复编写！