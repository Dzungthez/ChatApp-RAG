from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.pipeline.utils import pattern, extract_answer_from_model

# Define the prompt
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    It is really important to keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)


class Str_Output_Parser(StrOutputParser):
    def __init__(self):
        super().__init__()

    def parse(self, output: str, ):
        return extract_answer_from_model(output)
