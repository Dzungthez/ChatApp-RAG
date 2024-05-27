import os
from dotenv import load_dotenv
from model.grader import get_retrieval_grader, get_hallucination_grader, get_answer_grader
from model.load_model import get_generate_model
from model.vectorstore import MyDatabase
from pipeline.construct_workflow import WorkFlow
from model.websearch_tool import get_websearch_tool
from src.pipeline.text_loader import Loader

dotenv_path = os.path.join(os.path.dirname(__file__), '../../config/.env')
load_dotenv(dotenv_path)
init_filepath = os.getenv('INIT_FILEPATH')


def call_workflow():
    docs = Loader().load(init_filepath)
    retriever = MyDatabase(documents=docs).get_retriever()
    generate_model = get_generate_model()
    web_search_tool = get_websearch_tool()
    retrieval_grader = get_retrieval_grader()
    hallucination_grader = get_hallucination_grader()
    answer_grader = get_answer_grader()

    workflow = WorkFlow(
        retriever=retriever,
        generate_model=generate_model,
        web_search_tool=web_search_tool,
        retrieval_grader=retrieval_grader,
        hallucination_grader=hallucination_grader,
        answer_grader=answer_grader
    )

    return workflow


def main():
    workflow = call_workflow()
    question = "Which city is considered the city of light?"
    result = workflow.invoke(question)
    print(result['generation'])


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
