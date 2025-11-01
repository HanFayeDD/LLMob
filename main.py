from utils.logger import init_log
from utils.config import load_env
from agent.chatbot import ChatBot

if __name__ == "__main__":
    init_log()
    load_env()
    ChatBot()


    