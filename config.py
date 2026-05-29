import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("TOP_K", "3"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
DATA_DIR = os.environ.get("DATA_DIR", "data")
INDEX_PATH = os.environ.get("INDEX_PATH", "data/recipes.index")
CSV_PATH = os.environ.get("CSV_PATH", "data/recipes.csv")
