import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm(temperature: float = 0.0):
    token = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()

    if not token:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_key=token,
        openai_api_base=base_url,
        temperature=temperature,
    )
