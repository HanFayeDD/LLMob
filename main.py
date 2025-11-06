from utils.logger import init_log
from utils.config import load_env
from agent.chatbot import ChatBot
from agent.chatbotlocal import ChatBot as ChatBotOllama
from pydantic import BaseModel, Field
import os
import logging


if __name__ == "__main__":
    init_log()
    load_env()
    cb = ChatBotOllama()
    # print(cb.ask_one("1+1等于几"))

    class Book(BaseModel):
        """the struct of a book"""
        title:str = Field(..., description="The title of the book")
        author:str = Field(..., description="The author of the book")
        year:int = Field(..., description="The year the book was published")

    print(cb.ask_one_with_struct_output("the book on the shelf is 'The Hitchhiker's Guide to the Galaxy' by Douglas Adams in 1989. I had read it five times in 2022", Book))
    