import os
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

dotenv_path = os.path.join(os.path.dirname(__file__), '../../config/.env')
load_dotenv(dotenv_path)
api_key = os.getenv('API_KEY')
model_name = "llama3-70b-8192"


def get_model(temperature=0.8):
    llm_model = ChatGroq(groq_api_key=api_key, model_name=model_name,
                         temperature=temperature)
    return llm_model


def get_generate_model(temperature=0.8):
    llm_model = get_model(temperature=temperature)
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "context"],
    )

    return prompt | llm_model | StrOutputParser()
