from utils.logger import init_log
from utils.config import load_env
from agent.chatbot import ChatBot
from pydantic import BaseModel, Field
import os


if __name__ == "__main__":
    init_log()
    load_env()
    cb = ChatBot()
    print(cb.ask_one("1+1等于几"))


    class Movie(BaseModel):
        """A movie with details."""
        title: str = Field(..., description="The title of the movie")
        year: int = Field(..., description="The year the movie was released")
        director: str = Field(..., description="The director of the movie")
        rating: float = Field(..., description="The movie's rating out of 10")

    print(cb.ask_one_with_struct_output("Provide details about the movie Inception", Movie))
    