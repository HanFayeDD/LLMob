import os
from dotenv import load_dotenv

def load_env()->None:
    load_dotenv()

def get_env(k:str)->str:
    return os.getenv(k)
