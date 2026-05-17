import os
from dotenv import load_dotenv


load_dotenv()


class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
    RETRIEVAL_DISTANCE_THRESHOLD: float = 1.4


settings = Settings()