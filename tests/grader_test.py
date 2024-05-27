import unittest

from src.model.grader import get_hallucination_grader, get_retrieval_grader, get_answer_grader


class MyTestCase(unittest.TestCase):
    def test_retriever_grader(self):
        grader = get_retrieval_grader()
        question = "What is the capital of Vietnam?"
        docs = ["Hai Phong is a seaside city in Vietnam.",
                "Ho Chi Minh City is the largest city in Vietnam.",
                "Hanoi, a city in Vietnam, is the capital of the country."]
        response = grader.invoke({"question": question, "document": docs})
        print(response)
        self.assertIn('yes', response['score'])

    def test_hallucination_grader(self):
        grader = get_hallucination_grader()
        generation = "Hanoi is not the capital of Vietnam."
        docs = ["Hai Phong is a seaside city in Vietnam.",
                "Ho Chi Minh City is the largest city in Vietnam.",
                "Hanoi, a city in Vietnam, is the metropolis of the country."]
        response = grader.invoke({"generation": generation, "documents": docs})
        print(response)
        self.assertIn('yes', response['score'])

    def test_answer_grader(self):
        grader = get_answer_grader()
        generation = "Hanoi is the capital of Vietnam."
        question = "What is the capital of Vietnam?"
        response = grader.invoke({"generation": generation, "question": question})
        print(response)
        self.assertIn('yes', response['score'])


if __name__ == '__main__':
    unittest.main()
