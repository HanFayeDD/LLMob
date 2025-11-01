import os
from dataclasses import dataclass
import toml
import logging

CONF_FILE = ["text", "config", "key.toml"]

@dataclass
class Conf():
    pass

@dataclass
class LLMAPIConf(Conf):
    api_key: str 
    model_name: str 
    base_url: str

class ConfigLoader():
    def __init__(self):
        self.path = os.path.join(os.getcwd(), *CONF_FILE)
        
    def load(self)->Conf:
        with open(self.path, "r", encoding='utf-8') as f:
            conf = toml.load(f)
        
        llmapiconf = LLMAPIConf(api_key=conf["LLMAPIConf"]["OPENAI_API_KEY"],
                                model_name=conf["LLMAPIConf"]["OPENAI_API_MODEL"],
                                base_url=conf["LLMAPIConf"]["OPENAI_API_BASE"])   
        
        logging.info("加载配置")
        
        return llmapiconf
         


