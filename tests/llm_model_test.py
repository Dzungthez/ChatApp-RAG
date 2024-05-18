import unittest
from src.model.load_model import get_model


class TestChatGroqModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model()

    def test_basic_response(self):
        question = "What is the capital of Vietnam?"
        response = self.model.invoke(question)

        expected_output = ["Hanoi", "Ha Noi"]
        self.assertTrue(any(city in response.content for city in expected_output))

    def test_numerical_question(self):
        question = "what is 20 + 55?"
        response = self.model.invoke(question)
        self.assertIn("75", response.content)


if __name__ == '__main__':
    unittest.main()
