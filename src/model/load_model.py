import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

dotenv_path = os.path.join(os.path.dirname(__file__), '../../config/.env')
load_dotenv(dotenv_path)
api_key = os.getenv('API_KEY')
model_name = "llama3-70b-8192"


def get_model(temperature=0.8):
    llm_model = ChatGroq(groq_api_key=api_key, model_name=model_name,
                         temperature=temperature)
    return llm_model
