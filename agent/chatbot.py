from langchain.agents import create_agent
from utils.config import get_env



class ChatBot():
    def __init__(self):
        print(get_env("OPENAI_API_KEY"))
    
    
