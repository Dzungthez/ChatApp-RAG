from pydantic import BaseModel
from pydantic import Field

from src.pipeline.text_loader import Loader
from src.model.vectorstore import MyDatabase


class InputChat(BaseModel):
    question: str = Field(..., title="Question to ask the model")


class OutputChat(BaseModel):
    answer: str = Field(..., title="Answer from the model")


def build_chain(model, data_dir):
    docs = Loader().load_dir(data_dir)
    retriever = MyDatabase(documents=docs).get_retriever()
