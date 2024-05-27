import unittest
from pprint import pprint

from src.model import load_model, websearch_tool, vectorstore
from src.model.grader import get_retrieval_grader, get_hallucination_grader, get_answer_grader
from src.pipeline.construct_workflow import WorkFlow
from src.pipeline.text_loader import Loader


class MyTestCase(unittest.TestCase):
    def test_workflow(self):
        docs = Loader().load("test.pdf")
        retriever = vectorstore.MyDatabase(documents=docs).get_retriever()
        llm_model = load_model.get_generate_model()
        web_search_tool = websearch_tool.get_websearch_tool()
        retrieval_grader = get_retrieval_grader()
        hallucination_grader = get_hallucination_grader()
        answer_grader = get_answer_grader()
        self.assertTrue(True)
        workflow = WorkFlow(retriever, llm_model, web_search_tool,
                            retrieval_grader, hallucination_grader, answer_grader)
        questions = {"question": "Which city is considered the city of light?"}
        for output in workflow.app.stream(questions):
            for key, value in output.items():
                pprint(f"Finished running: {key}")
        pprint(value["generation"])
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
