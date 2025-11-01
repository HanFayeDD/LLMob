import os
from dotenv import load_dotenv

def load_env()->None:
    load_dotenv()

def get_env(k:str)->str:
    res = os.getenv(k)
    if res is None:
        raise ValueError(f"{k} not found in env")
    return res 
